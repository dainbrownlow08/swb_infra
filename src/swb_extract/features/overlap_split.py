"""Cooperative vs obstructive overlap — the Tannen-diagnostic split (§4E-a).

Tannen ref: Ch.4 "Overlap and Pace" (PDF pp.113-122); Ch.7 dim 5a. Cooperative
overlap — chiming in to show engagement — is the signature High-Involvement
device; an interruption that seizes the floor is its obstructive opposite. The
existing ``overlap`` extractor measures how MUCH simultaneous speech there is;
this one classifies each overlap EVENT by what it did to the floor (AUDIT.md
§4E-a; submission plan Delta 7d).

Events are defined on the merged FTO turn walk (``fto.build_turn_events`` — the
same state machine behind ``FTO Sec``, so the two features cannot disagree
about turns). Every OTHER-speaker utterance whose word-tight start falls before
the holder's running word-tight turn end initiates exactly one overlap event,
attributed to the incoming (overlapping) utterance:

- **listener backchannel** — cooperative BY DEFINITION (Delta 7d): "uh-huh"
  over ongoing speech is support, not a floor claim.
- **interjection** (contained; the holder talked through it and kept the
  floor) — cooperative: the overlapper never held the floor.
- **floor-taking overlap** (a transfer with FTO < 0): **obstructive** iff the
  overlapped speaker's turn terminates within ``OBSTRUCTIVE_WINDOW_SEC`` of
  the overlap onset (turn_end − start ≤ W) — the incomer drove them off the
  floor; **cooperative** otherwise — the holder talked on past W and finished
  their turn despite the overlap (anticipatory/enthusiastic completion, the
  classic involvement pattern).

W = ``OBSTRUCTIVE_WINDOW_SEC`` = 1.0 s, fixed in advance (Delta 7d, recorded
2026-07-09) — the standard "successful interruption" operationalization:
speaker switch close to overlap onset. Gold validity checks (overlapping gold
``b`` events ~all cooperative; ``+``-continuation floor retention) run in NB07
Step 14, which adjudicates the two columns' trust status.

Columns (0 = placeable utterance, no overlap event; empty = unplaceable):

- ``Cooperative Overlap Count`` — 1 if this utterance initiated a cooperative
  overlap event, else 0. Sums to unit-level event counts.
- ``Obstructive Overlap Count`` — likewise for obstructive events. At most one
  of the two is 1 per utterance (each utterance starts at most one event).

Output: utterances_v2/features/overlap_split.csv
Header: Utterance File Name,Cooperative Overlap Count,Obstructive Overlap Count
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from .fto import BC, CONT, FIRST, INTERJ, Event, EventIndex, build_turn_events

FEATURE_NAME = "overlap_split"
HEADER = (
    "Utterance File Name",
    "Cooperative Overlap Count",
    "Obstructive Overlap Count",
)

OBSTRUCTIVE_WINDOW_SEC = 1.0


def classify_event(event: Event) -> tuple[int, int]:
    """(cooperative, obstructive) counts for one turn-walk event."""
    kind, start, _end, ref, _lex_bc = event
    if kind in (FIRST, CONT):
        # Same-side material (ref is the speaker's OWN previous end, not the
        # other speaker's turn): no cross-speaker overlap event by definition.
        return (0, 0)
    if ref is None or start >= ref:
        # No open turn to overlap (conversation-initial backchannel), or the
        # utterance began at/after the holder's running end — no overlap.
        return (0, 0)
    if kind == BC:
        return (1, 0)  # backchannel-only overlap: cooperative by definition
    if kind == INTERJ:
        return (1, 0)  # the holder talked through it: floor retained
    # TRANSFER in overlap: obstructive iff the holder ceded within the window.
    return (0, 1) if (ref - start) <= OBSTRUCTIVE_WINDOW_SEC else (1, 0)


def lookup_split(idx: EventIndex, rel_path: str) -> tuple[int, int] | None:
    event = idx.get(parse_rel_path(rel_path))
    return None if event is None else classify_event(event)


def write_overlap_split(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    idx = build_turn_events(transcript_root)
    n = 0
    n_coop = 0
    n_obstr = 0
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
            split = lookup_split(idx, rel)
            if split is None:
                n_missing += 1
                writer.writerow([rel, "", ""])
            else:
                coop, obstr = split
                n_coop += coop
                n_obstr += obstr
                writer.writerow([rel, str(coop), str(obstr)])
            n += 1
    n_events = n_coop + n_obstr
    print(
        f"  overlap events={n_events} (cooperative={n_coop}, obstructive={n_obstr}"
        + (f", obstructive share={100 * n_obstr / n_events:.1f}%" if n_events else "")
        + f"), unplaceable={n_missing}"
    )
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_overlap_split(
        manifest_path(out_root),
        out_root / "features" / "overlap_split.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} overlap split rows")
    return 0
