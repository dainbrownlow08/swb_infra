"""Token Count per utterance — raw integer count of tokens.

Legacy `FETokenCount.py` returned `[token_count, word_rate]` using
`gensim.utils.tokenize()`, which conflicted with `FEWordRate.py`'s
`text.split()` — two different definitions of 'word' producing two columns
both named `word_rate`. We split the responsibilities cleanly: `token_count`
returns only the count, and `word_rate.py` returns only the rate. Both share
the bracket-stripped tokenizer used by the other linguistic extractors.

Output: utterances_v2/features/token_count.csv
Header: Utterance File Name,token_count
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path

FEATURE_NAME = "token_count"
HEADER = ("Utterance File Name", "token_count")


def tokenize(text: str) -> list[str]:
    return [
        w for w in text.lower().split()
        if not (w.startswith("[") and w.endswith("]"))
    ]


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def write_token_counts(
    manifest_csv: Path,
    output_csv: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(manifest_csv, encoding="utf-8", newline="") as fin, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest_csv}: {header!r}"
            )
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for row in reader:
            if not row:
                continue
            rel, text = row[0], row[1]
            writer.writerow([rel, count_tokens(text)])
            n += 1
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_token_counts(
        manifest_path(out_root),
        out_root / "features" / "token_count.csv",
    )
    print(f"wrote {n} token count rows")
    return 0
