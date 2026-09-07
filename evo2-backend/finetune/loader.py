"""Constructing Evo2 under PyTorch's ``weights_only`` unpickler.

Loading the published evo2 checkpoint trips ``weights_only=True`` in two
independent places, neither of them ours:

* ``vortex.model.utils.load_checkpoint`` passes ``weights_only=True``
  explicitly, so this is not something a torch version pin can avoid.
* ``transformer_engine``'s ``set_extra_state`` calls ``torch.load`` on each
  module's serialised extra state during ``load_state_dict``.

The checkpoint legitimately contains objects outside the default allowlist
(``_codecs.encode``, ``transformer_engine.common.recipe._OverrideLinearPrecision``,
and however many more sit behind those), and which one surfaces first shifts
with the torch version, so allowlisting them one at a time is a treadmill.

``weights_only`` exists to stop an *untrusted* pickle from executing code on
load. These weights come from Arc Institute's official Hugging Face repo, which
is the same artifact the rest of the pipeline is built around and is already
trusted to run as model code — so the trust condition is met and the check is
relaxed for the duration of the load only.
"""

import contextlib
import importlib

# Bump on every change to this file. Printed by load_evo2 so a run's log makes
# clear which version of loader.py is actually embedded in the notebook -- Kaggle
# has repeatedly run a stale copy after a re-upload.
LOADER_REVISION = 7


@contextlib.contextmanager
def trusted_torch_load():
    """Force ``weights_only=False`` on every ``torch.load`` inside the block.

    Patching the module attribute is what makes this reach ``torch.load`` calls
    inside vortex and transformer_engine, which take no arguments from us. The
    original is always restored, so nothing outside the block is affected.
    """
    import torch

    original = torch.load

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def _gpu_supports_fp8() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


def _gpu_supports_bf16() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 0)


def _cast_to_fp16(evo2_model) -> None:
    """Run the model in float16 because the GPU has no bfloat16.

    ``evo2_7b`` loads in bfloat16. On a pre-Ampere GPU (compute capability < 8.0,
    e.g. a T4 at 7.5) ``ptxas`` rejects the ``.bf16`` PTX that vortex's Triton
    kernels (rotary embedding, etc.) emit for bf16 tensors:

        Feature '.bf16' requires .target sm_80 or higher

    Casting the whole module to float16 makes those kernels compile the ``.f16``
    path instead, which every CUDA GPU since Pascal supports. float16 has bf16's
    precision but a much smaller exponent range, so this can overflow to inf/nan
    where bf16 would not -- another reason this is a last-resort fallback, only
    taken on GPUs that cannot run the model any other way.
    """
    evo2_model.model.half()


def _neuter_fp8_autocast() -> None:
    """Make ``te.fp8_autocast`` a no-op context manager, process-wide.

    ``evo2_7b``'s checkpoint config sets ``use_fp8_input_projections=True``. FP8
    execution needs an Ada/Hopper GPU (compute capability >= 8.9); on a T4
    (CC 7.5) the ``with te.fp8_autocast(enabled=True, ...)`` in
    ``vortex.model.layers`` asserts at forward time. The body of that ``with``
    is ``out = super().forward(x)`` -- an ordinary ``nn.Linear`` call -- so if
    the context manager simply does nothing, the projection runs in the model's
    normal precision and everything downstream works.

    Patching per-module ``use_fp8_input_projections`` flags proved unreliable
    (the flag lives on a closure-defined class and the walk can miss it);
    replacing the one function every FP8 path funnels through cannot be missed.
    vortex does ``import transformer_engine.pytorch as te`` and calls
    ``te.fp8_autocast`` by attribute, so overwriting the module attribute takes
    effect for calls made after this runs.

    This is an approximation of the intended numerics -- the projection weights
    were calibrated for FP8 -- not a bit-exact substitute. Only applied on GPUs
    that cannot run the model any other way.
    """
    import transformer_engine.pytorch as te

    if getattr(te.fp8_autocast, "_evo2_neutered", False):
        return

    @contextlib.contextmanager
    def _noop_fp8_autocast(*args, **kwargs):
        yield

    _noop_fp8_autocast._evo2_neutered = True
    te.fp8_autocast = _noop_fp8_autocast


def _patch_flash_attention_to_sdpa() -> None:
    """Route vortex's attention through torch SDPA instead of FlashAttention-2.

    FlashAttention-2 (the ``flash_attn`` wheel) only supports Ampere and newer
    (sm_80+). On a T4 (sm_75) the kernel raises ``FlashAttention only supports
    Ampere GPUs or newer`` at forward time. The installed vortex funnels its
    dense-attention path through the module-level
    ``local_flash_attn_qkvpacked_func`` (and its varlen sibling), so replace
    those with a ``torch.nn.functional.scaled_dot_product_attention`` equivalent
    -- SDPA's math / mem-efficient backends run on every CUDA GPU. Patching the
    function rather than a class name survives vortex refactors.

    Full, non-windowed attention only (evo2_7b's attention layers are global);
    ``softcap`` and ``alibi_slopes`` are not supported by SDPA and are asserted
    to be unset rather than silently dropped.
    """
    import math

    import torch

    def _sdpa_qkvpacked(
        qkv, dropout_p=0.0, softmax_scale=None, causal=False,
        window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
        deterministic=False, return_attn_probs=False, **_,
    ):
        assert not softcap, "SDPA attention fallback does not support softcap"
        assert alibi_slopes is None, "SDPA attention fallback does not support alibi"
        if tuple(window_size) not in ((-1, -1),):
            print(f"  !! attention window_size={window_size} ignored by SDPA fallback")
        q, k, v = qkv.unbind(dim=2)               # each (B, S, H, D)
        q = q.transpose(1, 2)                     # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        d = q.shape[-1]
        scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=bool(causal), scale=scale,
        )
        return out.transpose(1, 2).contiguous()   # (B, S, H, D)

    def _sdpa_varlen_qkvpacked(qkv, cu_seqlens, max_seqlen, dropout_p=0.0,
                               softmax_scale=None, causal=False, **kw):
        # features._forward never pads, so every batch is fixed-length: fold the
        # single sequence back to (1, S, 3, H, D) and reuse the dense path.
        if qkv.dim() == 4:
            qkv = qkv.unsqueeze(0)
        return _sdpa_qkvpacked(qkv, dropout_p, softmax_scale, causal)[0]

    _sdpa_qkvpacked._evo2_sdpa = True
    patched = 0
    for modname in ("vortex.ops.attn_interface", "vortex.model.attention"):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for attr, repl in (
            ("local_flash_attn_qkvpacked_func", _sdpa_qkvpacked),
            ("local_flash_attn_varlen_qkvpacked_func", _sdpa_varlen_qkvpacked),
        ):
            if hasattr(mod, attr) and not getattr(getattr(mod, attr), "_evo2_sdpa", False):
                setattr(mod, attr, repl)
                patched += 1
    print(f"  routed {patched} vortex attention entrypoint(s) through SDPA")


def load_evo2(model_name: str, allow_fp8_fallback: bool = True):
    """Build an ``Evo2`` with the checkpoint's own pickle contents permitted.

    When ``allow_fp8_fallback`` and the current GPU lacks FP8 support, FP8
    autocast is disabled so the input projections run in normal precision.
    """
    from evo2 import Evo2

    import torch

    print(
        f"load_evo2: loader.py revision {LOADER_REVISION}; "
        f"{torch.cuda.device_count()} CUDA device(s) visible"
    )

    if allow_fp8_fallback and not _gpu_supports_fp8():
        _neuter_fp8_autocast()
        print(
            "load_evo2: GPU lacks FP8 support (compute capability < 8.9) -- "
            "disabled FP8 autocast; projection results are an approximation of "
            "the FP8-calibrated model."
        )

    if allow_fp8_fallback and not _gpu_supports_bf16():
        _patch_flash_attention_to_sdpa()
        print(
            "load_evo2: GPU predates FlashAttention-2 (compute capability < 8.0) "
            "-- routing attention through torch SDPA."
        )

    with trusted_torch_load():
        model = Evo2(model_name)

    if allow_fp8_fallback and not _gpu_supports_bf16():
        _cast_to_fp16(model)
        print(
            "load_evo2: GPU lacks bfloat16 (compute capability < 8.0) -- cast "
            "model to float16; watch for inf/nan from the reduced exponent range."
        )

    return model
