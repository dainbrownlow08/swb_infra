"""wpu — mean words per utterance (Thomas et al. 2018 §3.2, "Pauses, turn-taking").

"wpu Mean words per utterance." Per utterance this module writes the word count —
the shared in-house tokenizer's tokens (markup dropped, laughed words unwrapped), the
Switchboard analogue of the speech-to-text words the paper counted. It is also the
word total that ppron, wps and wpp divide (their aggregates read this column). The
paper's wpu for a side is the mean over its utterances (``aggregate``). Loading on
involvement: +0.45, the largest of the eleven.

Output: utterances_v2/features/wpu.csv
Header: Utterance File Name,wpu
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ..inhouse._text import tokenize
from ._io import Row, mean_of, write_feature_csv

FEATURE_NAME = "wpu"
HEADER = ("Utterance File Name", "wpu")


def count_words(text: str) -> int:
    return len(tokenize(text))


def aggregate(rows: list[Row]) -> float | None:
    return mean_of(rows, "wpu")


def write_wpu(manifest_csv: Path, output_csv: Path) -> int:
    return write_feature_csv(
        manifest_csv, output_csv, HEADER, lambda _rel, text: [str(count_words(text))]
    )


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_wpu(manifest_path(out_root), out_root / "features" / f"{FEATURE_NAME}.csv")
    print(f"wrote {n} wpu rows")
    return 0
