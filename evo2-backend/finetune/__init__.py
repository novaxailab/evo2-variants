"""Fine-tuning pipeline for Evo2 variant effect prediction.

Evo2 stays frozen. The pipeline caches its embeddings and per-token
log-probabilities for labelled variants, then trains a small classifier head on
top of them, replacing the hand-fitted delta-likelihood threshold that the
zero-shot endpoint uses.

Stages, in order:

1. ``data``      - build labelled SNV tables from ClinVar and the BRCA1 DMS set
2. ``features``  - one GPU pass per variant, sharded and resumable
3. ``train``     - fit the head, pick a threshold, score against the zero-shot baseline
4. ``head``      - the serialised artifact the endpoint loads

See ``FINETUNING.md`` for the runbook.
"""

from . import config  # noqa: F401

__all__ = ["config", "data", "features", "head", "sequences", "train"]
