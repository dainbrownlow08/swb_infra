"""Question features — interrogative form & echo questions.

Tannen ref: Ch.7 dim 4 (PDF p.202) "Use of questions"; Ch.4 echo questions as
back-channel (p.87, "You stayed at the Plaza?"). High-Involvement speakers ask
more questions and echo the other's words back as engaged questions.

Switchboard trans carries NO punctuation, so we detect interrogative FORM from
onset syntax (cheap, transparent):

  Question Flag = 1 if the bracket-stripped utterance begins with
     - a wh-word: what/where/when/who/whom/whose/why/how/which, OR
     - a subject-aux inversion starter: do/does/did/is/are/am/was/were/
       can/could/will/would/shall/should/have/has/had/may/might/must
     0 otherwise; "" if the utterance has no content tokens.

  Echo Question Flag = 1 if Question Flag AND the utterance is short
     (<= ECHO_MAX_TOKENS content tokens) AND >= ECHO_MIN_OVERLAP of its content
     tokens also appear in the PREVIOUS (cross-speaker) turn — throwing the
     other speaker's words back as a question. 0 otherwise; "" when Question
     Flag is undefined.

Onset detection is deliberately conservative: it misses declarative-form
questions (caught by the prosodic rising-terminal feature) and will occasionally
false-fire on "how ..." exclamations. Validate on a sample before downstream
use.

Output: utterances_v2/features/question_flags.csv
Header: Utterance File Name,Question Flag,Echo Question Flag
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from .turn_gap import (
    TextIndex,
    TurnGapIndex,
    build_text_index,
    build_turn_gap_index,
)

FEATURE_NAME = "question_flags"
HEADER = ("Utterance File Name", "Question Flag", "Echo Question Flag")

WH_WORDS = frozenset(
    {"what", "where", "when", "who", "whom", "whose", "why", "how", "which"}
)
AUX_STARTERS = frozenset(
    {
        "do", "does", "did", "is", "are", "am", "was", "were", "can", "could",
        "will", "would", "shall", "should", "have", "has", "had", "may",
        "might", "must",
    }
)

ECHO_MAX_TOKENS = 6
ECHO_MIN_OVERLAP = 0.5

_PUNCT = "\"'.,!?;:()[]"


def strip_bracket_tokens(text: str) -> str:
    """Drop whole-bracket tokens ([noise], [laughter]); keep 'i[t]-' fragments."""
    return " ".join(
        t for t in text.split() if not (t.startswith("[") and t.endswith("]"))
    )


def _content_tokens(text: str) -> list[str]:
    out: list[str] = []
    for t in strip_bracket_tokens(text).split():
        w = t.strip(_PUNCT).lower()
        if w:
            out.append(w)
    return out


def question_flag(text: str) -> int | None:
    """1 if the utterance opens with a wh-word or an aux/modal; 0 otherwise.

    None when there are no content tokens (pure brackets / empty).
    """
    toks = _content_tokens(text)
    if not toks:
        return None
    first = toks[0]
    return 1 if (first in WH_WORDS or first in AUX_STARTERS) else 0


def is_echo_question(cur_text: str, prev_text: str) -> int:
    """1 if a short current question reuses the previous turn's content words."""
    cur = _content_tokens(cur_text)
    if not cur or len(cur) > ECHO_MAX_TOKENS:
        return 0
    prev = set(_content_tokens(prev_text))
    if not prev:
        return 0
    overlap = sum(1 for t in cur if t in prev)
    return 1 if overlap / len(cur) >= ECHO_MIN_OVERLAP else 0


def _previous_text(
    merged_idx: TurnGapIndex,
    text_idx: TextIndex,
    call_id: int,
    side: str,
    utt_num: int,
) -> str | None:
    """Text of the cross-speaker chronological predecessor, or None."""
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
    return text_idx.get((call_id, prev_side, prev_utt))


def _fmt(v: int | None) -> str:
    return "" if v is None else str(v)


def write_question_flags(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_idx = build_turn_gap_index(transcript_root)
    text_idx = build_text_index(transcript_root)

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
            rel = row[0]
            text = row[1] if len(row) > 1 else ""
            q = question_flag(text)
            echo: int | None
            if q != 1:
                # Not a question (or undefined) → echo is 0 (or "" if undefined).
                echo = None if q is None else 0
            else:
                try:
                    call_id, side, utt_num = parse_rel_path(rel)
                except ValueError:
                    echo = 0
                else:
                    prev = _previous_text(
                        merged_idx, text_idx, call_id, side, utt_num
                    )
                    echo = is_echo_question(text, prev) if prev is not None else 0
            writer.writerow([rel, _fmt(q), _fmt(echo)])
            n += 1
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_question_flags(
        manifest_path(out_root),
        out_root / "features" / "question_flags.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} question-flag rows")
    return 0
