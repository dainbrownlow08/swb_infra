"""Repetitions In Previous Utterance — cross-utterance pair matches.

For each utterance, count token-position pair matches (i, j) where
current[i] == previous[j], where "previous" is the immediately preceding
utterance in the conversation's cross-speaker chronological merge (same
"previous" definition used by Turn Gap). Equivalently:

    sum over words w of (count_in_current[w] * count_in_previous[w])

This preserves the legacy `FERepeats` cross-utterance metric semantics while
fixing the bugs:

- Whole-bracket tokens (`[noise]`, `[laughter]`, …) are stripped — they are
  noises, not words, and would otherwise manufacture fake matches.
- First utterance of a conversation has no previous → empty cell, not 0.
  (The legacy collapsed this into 0, an analogue of the t=0 inflation we
  fixed for Turn Gap.)
- Tokenization, indexing, and "previous" come from the same trusted source
  as the existing extractors — no `gensim` / NLTK lemmatizer dependency.

Output: utterances_v2/features/repetitions_in_previous.csv
Header: Utterance File Name,Repetitions In Previous Utterance
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from .repetition_rate import tokenize
from .turn_gap import (
    TextIndex,
    TurnGapIndex,
    build_text_index,
    build_turn_gap_index,
)

FEATURE_NAME = "repetitions_in_previous"
HEADER = ("Utterance File Name", "Repetitions In Previous Utterance")


def count_cross_pair_matches(cur: list[str], prev: list[str]) -> int:
    """sum over w of count_in_cur[w] * count_in_prev[w]."""
    a = Counter(cur)
    b = Counter(prev)
    # Iterate the smaller counter for efficiency.
    if len(a) > len(b):
        a, b = b, a
    return sum(a[w] * b[w] for w in a if w in b)


def lookup_repetitions_in_previous(
    merged_idx: TurnGapIndex,
    text_idx: TextIndex,
    rel_path: str,
) -> int | None:
    call_id, side, utt_num = parse_rel_path(rel_path)
    merged = merged_idx.get(call_id)
    if not merged:
        return None
    cur_pos: int | None = None
    for i, e in enumerate(merged):
        if e[0] == side and e[1] == utt_num:
            cur_pos = i
            break
    if cur_pos is None or cur_pos == 0:
        return None
    prev_side, prev_utt, _, _ = merged[cur_pos - 1]
    cur_text = text_idx.get((call_id, side, utt_num))
    prev_text = text_idx.get((call_id, prev_side, prev_utt))
    if cur_text is None or prev_text is None:
        return None
    return count_cross_pair_matches(tokenize(cur_text), tokenize(prev_text))


def _fmt(v: int | None) -> str:
    return "" if v is None else str(v)


def write_repetitions_in_previous(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_idx = build_turn_gap_index(transcript_root)
    text_idx = build_text_index(transcript_root)
    n = 0
    n_first = 0
    n_missing = 0
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
            rel = row[0]
            v = lookup_repetitions_in_previous(merged_idx, text_idx, rel)
            if v is None:
                if _is_first_of_conversation(merged_idx, rel):
                    n_first += 1
                else:
                    n_missing += 1
            writer.writerow([rel, _fmt(v)])
            n += 1
    if n_first or n_missing:
        print(
            f"  empty cells: first-of-conversation={n_first}, missing={n_missing}"
        )
    return n


def _is_first_of_conversation(idx: TurnGapIndex, rel_path: str) -> bool:
    try:
        call_id, side, utt_num = parse_rel_path(rel_path)
    except ValueError:
        return False
    merged = idx.get(call_id)
    if not merged:
        return False
    return merged[0][0] == side and merged[0][1] == utt_num


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_repetitions_in_previous(
        manifest_path(out_root),
        out_root / "features" / "repetitions_in_previous.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} repetitions-in-previous rows")
    return 0
