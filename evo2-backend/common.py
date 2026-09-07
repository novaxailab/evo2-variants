"""Modal image and volumes shared by the inference endpoint and the training jobs.

Both apps build from the same image so a head trained in one runs under exactly
the same Evo2 / vortex / transformer-engine versions that will serve it.
"""

import modal

from finetune import config

# The build steps are unchanged from the original inference image, so switching
# to this module reuses the existing layer cache rather than rebuilding.
evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        ["build-essential", "cmake", "ninja-build",
            "libcudnn8", "libcudnn8-dev", "git", "gcc", "g++"]
    )
    .env({
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        # Scoring 8 kb windows on a 24 GB card leaves little headroom, and the
        # allocator was stranding ~6 GB as reserved-but-unallocated between
        # variants, which OOMs a 2 GB allocation on a card with 22 GB free in
        # principle. Expandable segments let those blocks be reused instead.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    # Pin torch *before* installing evo2, so evo2's unpinned torch dependency is
    # already satisfied and pip doesn't pull in a newer wheel that mismatches
    # this image's CUDA toolkit. The exact minor version matters only because
    # the flash-attn wheel below must be built against the same one.
    .run_commands("pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124")
    # vortex's attention module imports flash_attn_2_cuda at import time, but
    # nothing in the evo2/vortex dependency chain installs it. Three things are
    # pinned here, and each fails differently if wrong:
    #   - version >= 2.7.0. flash-attn 2.7.0 reduced the C++ `fwd` return from 8
    #     values to 4, and vortex's vendored attn_interface unpacks 4. Older
    #     builds raise "too many values to unpack" on the first forward pass.
    #     Note vortex *warns* that it supports <= 2.6.3; that warning is stale
    #     and contradicts its own code, so believe the code.
    #   - torch2.4 and abiFALSE, matching the torch pinned above and the
    #     pre-C++11 ABI of PyTorch's Linux wheels. Either being wrong shows up
    #     as an `undefined symbol` ImportError rather than an install failure.
    # Installed as a prebuilt wheel rather than `pip install flash-attn`, which
    # compiles for ~1h — long enough that a laptop sleeping kills the build.
    .run_commands(
        "pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'"
    )
    # transformer_engine below still builds from source with --no-build-isolation,
    # so it only sees packages already installed here.
    .run_commands("pip install packaging wheel")
    .run_commands("git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && cd evo2 && pip install .")
    .run_commands("pip uninstall -y transformer-engine transformer_engine")
    .run_commands("pip install 'transformer_engine[pytorch]==1.13' --no-build-isolation")
    .pip_install_from_requirements("requirements.txt")
    # Mounted at runtime, so editing these files does not invalidate the image.
    .add_local_python_source("common", "finetune")
)

hf_cache_volume = modal.Volume.from_name(config.HF_CACHE_VOLUME, create_if_missing=True)
data_volume = modal.Volume.from_name(config.DATA_VOLUME, create_if_missing=True)

# Model weights and pipeline artifacts. The endpoint mounts both: the second one
# is where it finds a published head.
VOLUMES = {
    config.HF_CACHE_PATH: hf_cache_volume,
    config.DATA_PATH: data_volume,
}
