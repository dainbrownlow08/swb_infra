"""Floor-Transfer Offset (FTO) per utterance — turn-level response timing.

The corrected response-timing feature (AUDIT.md §3 fix 1). The existing
``Turn Gap`` measures current.start − previous.end against whatever utterance
started most recently in the cross-speaker chronology — including the
listener's backchannels — so 60% of its values are negative and the median is
−0.49 s. That is segmentation arithmetic, not response timing. ``Turn Gap``
is kept as-is for paper replication; analyses should use FTO.

FTO (Heldner & Edlund 2010) is defined at genuine floor transfers:

- A **turn** is a maximal run of one speaker's talk, in the conversation's
  chronological merge, that the other speaker does not take over. Three kinds
  of intervening material do NOT end a turn:

  * the other speaker's **backchannels** (``backchannels.is_backchannel``) —
    a listener's "uh-huh" does not take the floor;
  * the other speaker's substantive utterances **fully contained** within the
    holder's ongoing speech (utterance end ≤ the turn's running end) — the
    holder talked through them, so the floor never changed hands. These are
    flagged as interjections;
  * the holder's own lexically backchannel-like utterances ("yeah" mid-turn
    is the speaker's own token, not a listener response) — they extend the
    turn like any other same-side talk.

- The floor transfers when the other speaker produces a substantive utterance
  that extends beyond the holder's running end. At that boundary,
  FTO = incoming.start − turn.end. Negative = started in overlap; positive =
  silent gap. A same-speaker resumption after silence is a continuation (no
  transfer happened), so it gets no FTO.

- Turn edges are **word-tight**: each utterance's start/end is its first-word
  onset / last-word offset from the sibling ``*-word.text`` files, falling
  back to trans-line bounds when word rows are missing. Trans-line bounds
  include silence padding that biases FTO negative.

Columns (empty cell = no measurement):

- ``FTO Sec``           — set on the first utterance of each turn that has a
                          previous turn; empty otherwise.
- ``Onset Gap Sec``      — speech-onset latency for every substantive
                          utterance: for turn-initials this equals FTO; for
                          same-side continuations it is the gap from the
                          speaker's own previous substantive utterance
                          (rapid-series pacing); for interjections it is the
                          (negative) offset into the holder's ongoing turn;
                          empty for backchannels and conversation-initial
                          turns.
- ``Turn Initial Flag``  — 1 = first utterance of a turn; 0 = anything else;
                          empty = unplaceable.
- ``Backchannel Flag``   — 1/0, purely lexical (the shared allowlist), so it
                          matches the analysis notebooks. Computed from the
                          trans-file text; falls back to the manifest
                          transcript for unplaceable rows, so never empty.
- ``Interjection Flag``  — 1 = substantive other-speaker utterance contained
                          within the holder's turn (no transfer); 0 = anything
                          else; empty = unplaceable.

Output: utterances_v2/features/fto.csv
Header: Utterance File Name,FTO Sec,Onset Gap Sec,Turn Initial Flag,Backchannel Flag,Interjection Flag
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from .backchannels import is_backchannel
from .turn_gap import build_text_index, build_turn_gap_index
from .word_align import WordRow, build_word_index

FEATURE_NAME = "fto"
HEADER = (
    "Utterance File Name",
    "FTO Sec",
    "Onset Gap Sec",
    "Turn Initial Flag",
    "Backchannel Flag",
    "Interjection Flag",
)

# Per-utterance result:
# (fto_sec, onset_gap_sec, turn_initial, backchannel, interjection);
# None = no measurement.
Result = tuple[float | None, float | None, int, int, int]
FtoIndex = dict[tuple[int, str, int], Result]

# Turn-walk event kinds. The walk itself (build_turn_events) is the single
# source of truth for the merged-turn state machine; build_fto_index and the
# overlap_split extractor are both thin views over it, so their turn logic
# cannot drift apart.
FIRST = "first"  # first substantive utterance of the conversation
CONT = "cont"  # same-side continuation (incl. the holder's own bc-like tokens)
BC = "bc"  # listener backchannel — never takes, extends, or breaks a turn
INTERJ = "interj"  # contained substantive utterance the holder talked through
TRANSFER = "transfer"  # genuine floor transfer

# Event = (kind, start, end, ref_end, lex_bc): word-tight bounds, plus the
# reference time the classification was made against — the holder's running
# turn end at onset (BC/INTERJ/TRANSFER; None for a conversation-initial BC),
# the speaker's own previous substantive end (CONT; None if none), None for
# FIRST. lex_bc = the lexical-allowlist flag for same-side rows.
Event = tuple[str, float, float, float | None, int]
EventIndex = dict[tuple[int, str, int], Event]


def _word_tight_bounds(
    rows: list[WordRow] | None, trans_start: float, trans_end: float
) -> tuple[float, float]:
    if not rows:
        return trans_start, trans_end
    return rows[0][0], max(end for _start, end, _tok in rows)


def build_turn_events(transcript_root: Path) -> EventIndex:
    """Walk each conversation's merged-turn chronology, classifying every utterance."""
    chrono = build_turn_gap_index(transcript_root)
    texts = build_text_index(transcript_root)
    words = build_word_index(transcript_root)
    idx: EventIndex = {}
    for call_id, merged in chrono.items():
        # The chrono merge is ordered by trans bounds, but every turn decision
        # below uses word-tight bounds — and trans bounds carry silence padding
        # that can reorder utterances (a long leading silence makes an utterance
        # sort early though its speech starts late). Resolve word-tight bounds
        # first, then re-order by word-time so the turn logic sees true onsets.
        placed = []
        for side, utt_num, t_start, t_end in merged:
            key = (call_id, side, utt_num)
            start, end = _word_tight_bounds(words.get(key), t_start, t_end)
            placed.append((start, end, side, utt_num, key))
        placed.sort(key=lambda p: (p[0], p[1], p[3]))

        cur_side: str | None = None  # speaker holding the floor
        cur_turn_end: float = 0.0  # running max word-tight end of the open turn
        last_sub_end: dict[str, float] = {}  # per-side last substantive end
        for start, end, side, utt_num, key in placed:
            bc = is_backchannel(texts.get(key, ""))
            if bc and side != cur_side:
                # Listener backchannel (incl. conversation-initial): never
                # starts, extends, or breaks a turn.
                ref = cur_turn_end if cur_side is not None else None
                idx[key] = (BC, start, end, ref, 1)
                continue
            if cur_side is None:
                # First substantive utterance of the conversation: opens a
                # turn, but there is no previous turn to measure against.
                idx[key] = (FIRST, start, end, None, 0)
                cur_side, cur_turn_end = side, end
            elif side == cur_side:
                idx[key] = (CONT, start, end, last_sub_end.get(side), 1 if bc else 0)
                cur_turn_end = max(cur_turn_end, end)
            elif end <= cur_turn_end:
                # Substantive but the holder talked through it: within-speech
                # interjection, not a floor transfer.
                idx[key] = (INTERJ, start, end, cur_turn_end, 0)
            else:
                idx[key] = (TRANSFER, start, end, cur_turn_end, 0)
                cur_side, cur_turn_end = side, end
            last_sub_end[side] = max(last_sub_end.get(side, 0.0), end)
    return idx


def build_fto_index(transcript_root: Path) -> FtoIndex:
    """Classify every well-formed utterance and time the floor transfers."""
    events = build_turn_events(transcript_root)
    idx: FtoIndex = {}
    for key, (kind, start, _end, ref, lex_bc) in events.items():
        if kind == BC:
            idx[key] = (None, None, 0, 1, 0)
        elif kind == FIRST:
            idx[key] = (None, None, 1, 0, 0)
        elif kind == CONT:
            onset = start - ref if ref is not None else None
            idx[key] = (None, onset, 0, lex_bc, 0)
        elif kind == INTERJ:
            # Onset is the (negative) offset into the holder's ongoing turn.
            idx[key] = (None, start - ref, 0, 0, 1)
        else:  # TRANSFER
            fto = start - ref
            idx[key] = (fto, fto, 1, 0, 0)
    return idx


def lookup_fto(idx: FtoIndex, rel_path: str) -> Result | None:
    return idx.get(parse_rel_path(rel_path))


def _fmt_float(v: float | None) -> str:
    return "" if v is None else repr(v)


def write_ftos(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    idx = build_fto_index(transcript_root)
    n = 0
    n_fto = 0
    n_onset = 0
    n_initial = 0
    n_bc = 0
    n_interj = 0
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
            result = lookup_fto(idx, rel)
            if result is None:
                # Unplaceable in the trans chronology; the backchannel flag is
                # still derivable from the manifest's own transcript column.
                n_missing += 1
                bc = 1 if is_backchannel(row[1] if len(row) > 1 else "") else 0
                writer.writerow([rel, "", "", "", str(bc), ""])
                n_bc += bc
            else:
                fto, onset, initial, bc, interj = result
                writer.writerow(
                    [
                        rel,
                        _fmt_float(fto),
                        _fmt_float(onset),
                        str(initial),
                        str(bc),
                        str(interj),
                    ]
                )
                n_fto += fto is not None
                n_onset += onset is not None
                n_initial += initial
                n_bc += bc
                n_interj += interj
            n += 1
    print(
        f"  turn-initial={n_initial}, fto-defined={n_fto}, onset-defined={n_onset}, "
        f"backchannels={n_bc}, interjections={n_interj}, unplaceable={n_missing}"
    )
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_ftos(
        manifest_path(out_root),
        out_root / "features" / "fto.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} fto rows")
    return 0
