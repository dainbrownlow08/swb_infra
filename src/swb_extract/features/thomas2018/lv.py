"""lv — loudness variation (Thomas et al. 2018 §3.2, "Expressive phonology").

"Loudness variation, measured the same way" as pv: the variance of the loudness signal
over frames with a speech signal (voiced frames), pooled over the participant-task.
Loudness here is frame RMS amplitude on the same frame grid as the F0 track (librosa —
the in-house loudness.py measure; OpenSMILE's loudness contour is not a dependency of
this pipeline): a monotone substitute whose units differ (RMS², not loudness²). The
paper z-scores every variable before use, so the scale is immaterial to its
involvement sum — but not to cross-study comparison of raw values. Per utterance this
module writes the voiced-frame count, RMS mean and RMS variance; ``aggregate`` pools
exactly as pv does. The in-house telephone-line-gain caveat (loudness as recorded, not
vocal effort) applies unchanged. Loading on involvement: +0.21.

Output: utterances_v2/features/lv.csv   (shares the _prosody cache; see --workers/--limit/--overwrite)
Header: Utterance File Name,lv n,lv mean,lv var
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ._io import Row, fmt, pooled_variance
from ._prosody import Prosody, write_projection

FEATURE_NAME = "lv"
HEADER = ("Utterance File Name", "lv n", "lv mean", "lv var")


def project(p: Prosody | None) -> list[str]:
    return ["", "", ""] if p is None else [str(p[0]), fmt(p[3]), fmt(p[4])]


def aggregate(rows: list[Row]) -> float | None:
    """RMS variance over all voiced frames of the group, pooled."""
    return pooled_variance(rows, "lv n", "lv mean", "lv var")


def write_lv(manifest_csv: Path, output_csv: Path, out_root: Path, **kw) -> int:
    return write_projection(manifest_csv, output_csv, out_root, HEADER, project, **kw)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_lv(
        manifest_path(out_root), out_root / "features" / f"{FEATURE_NAME}.csv", out_root,
        workers=args.workers, limit=args.limit, overwrite=args.overwrite,
    )
    print(f"wrote {n} lv rows")
    return 0
