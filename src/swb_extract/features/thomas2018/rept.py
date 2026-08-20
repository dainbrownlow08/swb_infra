"""rept — terms repeated from the same person's previous utterance (Thomas et al. 2018 §3.2, "Re-statement").

"Mean number of terms which are repeated from the same person's previous utterance.
Before counting repeats we removed stopwords as well as 'um', 'uh', and 'uh-huh', and
stemmed what remained." Per utterance this module writes the number of distinct
stemmed terms shared with the speaker's OWN previous transcript line (``_terms``:
frozen NLTK stopword list + the paper's three fillers removed, Porter stemmer; a term
is a type, so a word repeated twice counts once); blank for a side's first utterance.
The paper's rept is the mean over defined rows (``aggregate``). Loading on
involvement: +0.39. Construct note: this is SELF-repetition across a speaker's own
consecutive lines — the paper's "persistence" / re-statement — not the
other-repetition of Tannen dim 6 (AUDIT.md §4E-h; the in-house
repetitions_in_previous column mixes both predecessors).

Output: utterances_v2/features/rept.csv
Header: Utterance File Name,rept
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ..inhouse._turn_index import build_text_index
from ._io import Row, fmt, key_of, mean_of, write_feature_csv
from ._terms import build_repeat_index

FEATURE_NAME = "rept"
HEADER = ("Utterance File Name", "rept")


def aggregate(rows: list[Row]) -> float | None:
    return mean_of(rows, "rept")


def write_rept(manifest_csv: Path, output_csv: Path, transcript_root: Path) -> int:
    repeats = build_repeat_index(build_text_index(transcript_root))

    def cells(rel: str, _text: str) -> list[str]:
        key = key_of(rel)
        return [fmt(repeats.get(key) if key else None)]

    return write_feature_csv(manifest_csv, output_csv, HEADER, cells)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_rept(
        manifest_path(out_root),
        out_root / "features" / f"{FEATURE_NAME}.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} rept rows")
    return 0
