"""pv — pitch variation (Thomas et al. 2018 §3.2, "Expressive phonology").

"Pitch variation, measured as the variance in F0 at those times when there is a speech
signal according to OpenSMILE. Again this is per-person, per-task, i.e., this is the
variance across the entire recording." F0 in Hz from librosa.pyin on the in-house
pitch.py settings (50–400 Hz, ≈64 ms window / 16 ms hop at 8 kHz — ``_prosody``); the
variance is in Hz², as in the paper ("on the order of 1000 Hz²"), and therefore
register-dependent (the in-house rising-terminal redesign records why a semitone scale
would not be — a replication keeps the paper's scale). Per utterance this module writes
the voiced-frame count, F0 mean and F0 variance (population); the paper's pv is the
variance of ALL the side's voiced frames pooled (``aggregate``: N = Σn, μ = Σnμᵢ/N,
Var = Σn(vᵢ + μᵢ²)/N − μ²), never the mean of per-utterance variances. Loading on
involvement: +0.09.

Output: utterances_v2/features/pv.csv   (shares the _prosody cache; see --workers/--limit/--overwrite)
Header: Utterance File Name,pv n,pv mean,pv var
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ._io import Row, fmt, pooled_variance
from ._prosody import Prosody, write_projection

FEATURE_NAME = "pv"
HEADER = ("Utterance File Name", "pv n", "pv mean", "pv var")


def project(p: Prosody | None) -> list[str]:
    return ["", "", ""] if p is None else [str(p[0]), fmt(p[1]), fmt(p[2])]


def aggregate(rows: list[Row]) -> float | None:
    """F0 variance (Hz²) over all voiced frames of the group, pooled."""
    return pooled_variance(rows, "pv n", "pv mean", "pv var")


def write_pv(manifest_csv: Path, output_csv: Path, out_root: Path, **kw) -> int:
    return write_projection(manifest_csv, output_csv, out_root, HEADER, project, **kw)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_pv(
        manifest_path(out_root), out_root / "features" / f"{FEATURE_NAME}.csv", out_root,
        workers=args.workers, limit=args.limit, overwrite=args.overwrite,
    )
    print(f"wrote {n} pv rows")
    return 0
