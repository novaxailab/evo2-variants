"""Reference-genome access for building variant windows.

Training needs tens of thousands of 8kb windows, which is far too many round
trips for the UCSC REST API, so bulk work reads from a local FASTA on the Modal
volume. The REST path is kept for one-off lookups on the inference endpoint.
"""

from typing import Dict, Optional, Tuple

from . import config


class ReferenceGenome:
    """Random access to a reference assembly, indexed with pyfaidx.

    Chromosome names are normalised to the UCSC ``chrN`` convention on lookup,
    because ClinVar's VCF uses bare ``N`` while the UCSC FASTAs use ``chrN``.
    """

    def __init__(self, fasta_path: str, sole_contig: Optional[str] = None):
        """Open ``fasta_path``.

        ``sole_contig`` declares that this FASTA holds exactly one sequence and
        names which chromosome it is. Needed for the vendored GRCh37 chr17
        file, whose sequence is headed with an NCBI accession
        (``NC_000017.10``) rather than ``chr17``, so no amount of chr-prefix
        normalisation would match it. Naming the chromosome explicitly, rather
        than resolving anything at all to "the only contig present", keeps a
        lookup for some *other* chromosome an error instead of silently
        returning chr17 sequence.
        """
        from pyfaidx import Fasta

        self.fasta_path = fasta_path
        self._fasta = Fasta(fasta_path, sequence_always_upper=True, as_raw=True)
        keys = list(self._fasta.keys())
        self._name_map = self._build_name_map(keys)

        if sole_contig is not None:
            if len(keys) != 1:
                raise ValueError(
                    f"{fasta_path} was declared single-contig ({sole_contig!r}) "
                    f"but holds {len(keys)} sequences"
                )
            bare = (
                sole_contig[3:] if sole_contig.startswith("chr") else sole_contig
            )
            self._name_map.setdefault(bare, keys[0])
            self._name_map.setdefault(f"chr{bare}", keys[0])

    @staticmethod
    def _build_name_map(keys) -> Dict[str, str]:
        """Map both ``1`` and ``chr1`` spellings onto whatever the FASTA uses."""
        mapping = {}
        for key in keys:
            mapping[key] = key
            bare = key[3:] if key.startswith("chr") else key
            mapping.setdefault(bare, key)
            mapping.setdefault(f"chr{bare}", key)
            # ClinVar writes the mitochondrial contig as MT, UCSC as chrM.
            if bare in ("M", "MT"):
                mapping.setdefault("MT", key)
                mapping.setdefault("chrMT", key)
        return mapping

    def close(self) -> None:
        """Release the underlying FASTA handle.

        Modal reuses a container across map inputs, and ``Volume.reload()``
        refuses to run while any file on the volume is still open. pyfaidx holds
        the FASTA open for the life of the object, so a shard that leaves its
        genomes open makes the *next* shard on that container fail before it
        starts.
        """
        fasta = getattr(self, "_fasta", None)
        if fasta is not None:
            fasta.close()
            self._fasta = None

    def resolve(self, chromosome: str) -> Optional[str]:
        return self._name_map.get(chromosome)

    def length(self, chromosome: str) -> int:
        key = self.resolve(chromosome)
        if key is None:
            raise KeyError(f"Chromosome {chromosome!r} not in {self.fasta_path}")
        return len(self._fasta[key])

    def window(
        self,
        chromosome: str,
        position: int,
        window_size: int,
    ) -> Tuple[str, int]:
        """Return a window centred on 1-based ``position``.

        Returns ``(sequence, start)`` where ``start`` is the 0-based offset of
        the first base, so the variant sits at ``position - 1 - start``.
        """
        key = self.resolve(chromosome)
        if key is None:
            raise KeyError(f"Chromosome {chromosome!r} not in {self.fasta_path}")

        contig = self._fasta[key]
        contig_len = len(contig)
        p = position - 1
        if p < 0 or p >= contig_len:
            raise ValueError(
                f"Position {position} is outside {chromosome} (length {contig_len})"
            )

        half = window_size // 2
        start = max(0, p - half)
        end = min(contig_len, p + half)
        return str(contig[start:end]), start


def fetch_window_ucsc(
    position: int,
    genome: str,
    chromosome: str,
    window_size: int = 8192,
) -> Tuple[str, int]:
    """Fetch a single window from the UCSC REST API.

    Used by the inference endpoint, where one HTTP call per request is fine.
    Returns ``(sequence, start)`` with a 0-based ``start``, matching
    :meth:`ReferenceGenome.window`.
    """
    import requests

    half = window_size // 2
    start = max(0, position - 1 - half)
    end = position - 1 + half

    api_url = (
        f"https://api.genome.ucsc.edu/getData/sequence"
        f"?genome={genome};chrom={chromosome};start={start};end={end}"
    )
    response = requests.get(api_url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"UCSC API returned {response.status_code} for {chromosome}:{start}-{end}"
        )

    payload = response.json()
    if "dna" not in payload:
        raise RuntimeError(f"UCSC API error: {payload.get('error', 'unknown error')}")

    sequence = payload["dna"].upper()
    if len(sequence) != end - start:
        print(
            f"Warning: UCSC returned {len(sequence)} bases, expected {end - start}"
        )
    return sequence, start


def build_variant_window(
    ref_window: str,
    relative_position: int,
    alternative: str,
    expected_reference: Optional[str] = None,
) -> Tuple[str, str]:
    """Substitute a single base, returning ``(variant_window, reference_base)``.

    If ``expected_reference`` is given it is checked against the assembly. A
    mismatch means the coordinate, assembly or strand is wrong, and silently
    scoring it would poison the training set, so it raises.
    """
    if not 0 <= relative_position < len(ref_window):
        raise ValueError(
            f"Relative position {relative_position} outside window of "
            f"length {len(ref_window)}"
        )

    reference_base = ref_window[relative_position]
    if expected_reference is not None and reference_base != expected_reference.upper():
        raise ValueError(
            f"Reference mismatch: assembly has {reference_base!r}, "
            f"record claims {expected_reference!r}"
        )

    variant_window = (
        ref_window[:relative_position]
        + alternative.upper()
        + ref_window[relative_position + 1:]
    )
    return variant_window, reference_base


def ensure_hg38(force: bool = False) -> str:
    """Download and index hg38 onto the data volume. Returns the FASTA path."""
    import gzip
    import os
    import shutil
    import time

    import requests

    os.makedirs(config.GENOMES_DIR, exist_ok=True)
    fasta = config.HG38_FASTA

    if os.path.exists(fasta) and not force:
        print(f"hg38 already present at {fasta}")
    else:
        tmp_gz = fasta + ".gz.part"
        # Two failure modes to survive: a mid-transfer reset, where restarting
        # would throw away up to 950 MB, and a host refusing connections
        # outright, where only a different mirror helps. So retries both resume
        # from whatever already landed and rotate through the mirror list.
        urls = config.HG38_URLS
        attempts = 3 * len(urls)
        for attempt in range(1, attempts + 1):
            url = urls[(attempt - 1) % len(urls)]
            done = os.path.getsize(tmp_gz) if os.path.exists(tmp_gz) else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            print(
                f"Downloading hg38 from {url} (~950 MB)"
                + (f", resuming at {done / 1e6:.0f} MB" if done else "")
                + " ..."
            )
            try:
                with requests.get(
                    url, stream=True, timeout=600, headers=headers
                ) as response:
                    response.raise_for_status()
                    # 206 means the range was honoured; a server that ignores it
                    # replies 200 and restarts at byte 0, where appending would
                    # silently corrupt the file.
                    mode = "ab" if response.status_code == 206 else "wb"
                    with open(tmp_gz, mode) as handle:
                        shutil.copyfileobj(
                            response.raw, handle, length=8 * 1024 * 1024
                        )
                break
            except requests.exceptions.RequestException as exc:
                if attempt == attempts:
                    raise
                print(f"  attempt {attempt}/{attempts} failed ({exc}); retrying ...")
                time.sleep(5 * attempt)

        print("Decompressing (~3.1 GB) ...")
        tmp_fa = fasta + ".part"
        with gzip.open(tmp_gz, "rb") as src, open(tmp_fa, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        os.replace(tmp_fa, fasta)
        os.remove(tmp_gz)

    _ensure_index(fasta)
    return fasta


def ensure_hg19_chr17(force: bool = False) -> str:
    """Decompress and index the vendored GRCh37 chr17 FASTA. Returns its path."""
    import gzip
    import os
    import shutil

    os.makedirs(config.GENOMES_DIR, exist_ok=True)
    fasta = config.HG19_CHR17_FASTA

    if not os.path.exists(fasta) or force:
        if not os.path.exists(config.HG19_CHR17_FASTA_GZ):
            raise FileNotFoundError(
                f"{config.HG19_CHR17_FASTA_GZ} not found. It ships with the evo2 "
                "repo that the image clones; check the image build."
            )
        print(f"Decompressing {config.HG19_CHR17_FASTA_GZ} ...")
        tmp = fasta + ".part"
        with gzip.open(config.HG19_CHR17_FASTA_GZ, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        os.replace(tmp, fasta)

    _ensure_index(fasta)
    return fasta


def _ensure_index(fasta: str) -> None:
    import os

    from pyfaidx import Faidx

    if not os.path.exists(fasta + ".fai"):
        print(f"Building .fai index for {fasta} ...")
        Faidx(fasta)
    print(f"Ready: {fasta}")


def open_genome(assembly: str) -> ReferenceGenome:
    """Open the local FASTA for ``assembly``, preparing it if necessary."""
    if assembly == "hg38":
        return ReferenceGenome(ensure_hg38())
    if assembly == "hg19":
        # Only chr17 is vendored; the BRCA1 benchmark never leaves that locus.
        return ReferenceGenome(ensure_hg19_chr17(), sole_contig="chr17")
    raise ValueError(
        f"Unsupported assembly {assembly!r}. Expected 'hg38' or 'hg19'."
    )
