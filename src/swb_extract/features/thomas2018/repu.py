"""repu — fraction of utterances with a repeated term (Thomas et al. 2018 §3.2, "Re-statement").

"The fraction of utterances which included at least one repeated term, defined as
above" — i.e. rept's preprocessing (``_terms``: stopwords + um/uh/uh-huh removed,
Porter-stemmed, same-side previous utterance). Per utterance this module writes
1 if the utterance shares ≥ 1 term with the speaker's own previous line, else 0; blank
for a side's first utterance. The paper's repu is the mean over defined rows
(``aggregate``). Loading on involvement: +0.39; repu was also the only one of the
eleven that correlated with time on task in MISC (r = −0.27).

Output: utterances_v2/features/repu.csv
Header: Utterance File Name,repu
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ..inhouse._turn_index import build_text_index
from ._io import Row, key_of, mean_of, write_feature_csv
from ._terms import build_repeat_index

FEATURE_NAME = "repu"
HEADER = ("Utterance File Name", "repu")


def aggregate(rows: list[Row]) -> float | None:
    return mean_of(rows, "repu")


def write_repu(manifest_csv: Path, output_csv: Path, transcript_root: Path) -> int:
    repeats = build_repeat_index(build_text_index(transcript_root))

    def cells(rel: str, _text: str) -> list[str]:
        key = key_of(rel)
        v = repeats.get(key) if key else None
        return ["" if v is None else str(int(v >= 1))]

    return write_feature_csv(manifest_csv, output_csv, HEADER, cells)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_repu(
        manifest_path(out_root),
        out_root / "features" / f"{FEATURE_NAME}.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} repu rows")
    return 0
