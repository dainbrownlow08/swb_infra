"""wps — speech rate in words per second (Thomas et al. 2018 §3.2, "Rate of speech").

"Speech rate, in words per second. This is an overall (micro-) average, calculated as
the number of words in the transcript divided by the total duration of utterances in
the transcript. An 'utterance' here is simply a line of transcript." Per utterance this
module writes the utterance's duration in seconds — word-tight (``_bounds``: first-word
onset to last-word offset; trans bounds only without word rows), so "the duration of an
utterance" means the same thing here as in olap and poplen. The paper's wps for a side
is Σwpu / Σ(wps sec) (``aggregate``), which is NOT the mean of per-utterance rates (the
in-house word_rate column is that per-utterance rate, over the padded trans span).
Loading on involvement: +0.39.

Output: utterances_v2/features/wps.csv
Header: Utterance File Name,wps sec
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ._bounds import build_bounds
from ._io import Row, fmt, key_of, ratio_of_sums, write_feature_csv

FEATURE_NAME = "wps"
HEADER = ("Utterance File Name", "wps sec")


def aggregate(rows: list[Row]) -> float | None:
    """Words per second over the group: Σwpu / Σ(wps sec)."""
    return ratio_of_sums(rows, "wpu", "wps sec")


def write_wps(manifest_csv: Path, output_csv: Path, transcript_root: Path) -> int:
    bounds = build_bounds(transcript_root)

    def cells(rel: str, _text: str) -> list[str]:
        key = key_of(rel)
        b = bounds.get(key) if key else None
        return [fmt(b[1] - b[0] if b else None)]

    return write_feature_csv(manifest_csv, output_csv, HEADER, cells)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_wps(
        manifest_path(out_root),
        out_root / "features" / f"{FEATURE_NAME}.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} wps rows")
    return 0
