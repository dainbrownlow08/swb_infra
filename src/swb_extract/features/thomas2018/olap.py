"""olap — overlap rate (Thomas et al. 2018 §3.2, "Overlap").

"The proportion of utterances which initiate an overlap: that is, the proportion of
one participant's utterances which begin while the other participant is still talking.
This need not be an interruption in the usual sense, as overlaps commonly include
utterances such as 'uh-huh' which indicate agreement but let the partner continue."
Per utterance this module writes the 0/1 flag (``_bounds.onset_timing``: word-tight
bounds, the partner's running end still ahead of this start; every transcript line
counted, backchannels included, exactly as the paper notes); the paper's olap is the
mean over the side (``aggregate``). Loading on involvement: −0.01 — the one sign
reversal in the paper's Table 2, which it reads as "practically zero ... overlap
effectively carries no signal". The in-house overlap / overlap_split columns are the
duration-based and cooperative/obstructive measures.

Output: utterances_v2/features/olap.csv
Header: Utterance File Name,olap
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ._bounds import build_onset_timing
from ._io import Row, key_of, mean_of, write_feature_csv

FEATURE_NAME = "olap"
HEADER = ("Utterance File Name", "olap")


def aggregate(rows: list[Row]) -> float | None:
    return mean_of(rows, "olap")


def write_olap(manifest_csv: Path, output_csv: Path, transcript_root: Path) -> int:
    timing = build_onset_timing(transcript_root)

    def cells(rel: str, _text: str) -> list[str]:
        key = key_of(rel)
        t = timing.get(key) if key else None
        return [str(t[0]) if t else ""]

    return write_feature_csv(manifest_csv, output_csv, HEADER, cells)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_olap(
        manifest_path(out_root),
        out_root / "features" / f"{FEATURE_NAME}.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} olap rows")
    return 0
