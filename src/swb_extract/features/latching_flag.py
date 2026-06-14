"""Latching Flag — near-zero-gap floor transfers (FTO-based).

Tannen ref: Ch.4 (PDF p.119) — "rapid rate of speech, overlap, and latching ...
show solidarity"; Ch.7 dim 5b (timing of contribution). Latching is a canonical
High-Involvement signal: the next speaker takes the floor with no audible gap
and no overlap, exactly as the prior speaker finishes.

Redefined on Floor-Transfer Offsets (AUDIT.md §3 fix 1). The original version
measured against the chronological predecessor — including listener
backchannels — whose negative-gap dominance suppressed the flag to 1.15%
positives. Latching is a property of a *floor transfer*, so it is defined only
where a transfer happened (``fto.FTO Sec``):

    Latching Flag =
      1   turn-initial and 0 <= FTO <= LATCH_MAX_SEC
      0   turn-initial and FTO outside that window (longer gap, or overlap)
      ""  everything else — continuations, backchannels, interjections, the
          first turn of a conversation (no transfer to time), unplaceable rows

The column mean is therefore the latch rate per floor transfer. For explicit
denominators use ``fto.csv``'s Turn Initial Flag. Overlap (FTO < 0) is
deliberately NOT latching — it is measured by the overlap extractor.

LATCH_MAX_SEC = 0.2 s: the conversation-analytic "beat" — tighter than a normal
between-turn pause. Documented constant; easy to sweep.

Output: utterances_v2/features/latching_flag.csv
Header: Utterance File Name,Latching Flag
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from .fto import FtoIndex, build_fto_index

FEATURE_NAME = "latching_flag"
HEADER = ("Utterance File Name", "Latching Flag")

LATCH_MAX_SEC = 0.2


def lookup_latching(idx: FtoIndex, rel_path: str) -> int | None:
    result = idx.get(parse_rel_path(rel_path))
    if result is None:
        return None
    fto, _onset, initial, _bc, _interj = result
    if not initial or fto is None:
        return None
    return 1 if 0.0 <= fto <= LATCH_MAX_SEC else 0


def _fmt(v: int | None) -> str:
    return "" if v is None else str(v)


def write_latching_flags(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    idx = build_fto_index(transcript_root)
    n = 0
    n_latched = 0
    n_defined = 0
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
            v = lookup_latching(idx, rel)
            if v is not None:
                n_defined += 1
                n_latched += v
            writer.writerow([rel, _fmt(v)])
            n += 1
    print(
        f"  defined (floor transfers)={n_defined}, latched={n_latched}"
        + (f" ({100 * n_latched / n_defined:.1f}% of transfers)" if n_defined else "")
    )
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_latching_flags(
        manifest_path(out_root),
        out_root / "features" / "latching_flag.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} latching flag rows")
    return 0
