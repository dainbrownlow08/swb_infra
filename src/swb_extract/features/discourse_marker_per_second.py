"""Discourse Markers per Second per utterance — the {D}-class half of the filler split.

Sibling of ``filled_pause_per_second`` (2026-08-20 split of the pooled Filler
Words per Second column, audit §4E-c — see that module's docstring for the
cancellation rationale). Counts only ``_text.DISCOURSE_MARKERS`` (so / well /
like / you know / i mean / i guess / basically).

Known construct caveat carried from the combined column: so / like / well are
lexical homographs ("i like dogs", "well water") and together are 34.5% of
all combined-column hits — this column OVERCOUNTS discourse-marker use by an
unquantified-per-speaker amount. Gold disambiguation study pending against
``corpus/annotations/treebank/dysfl/*.dff`` ({D}/{C}/{F} markup, 36 convs) at
the project's 0.8 precision bar; tokens that fail may be excised from
``DISCOURSE_MARKERS`` (which would break the sum identity with the legacy
column — re-gate on any list change).

  count = _text.count_filler_hits over DISCOURSE_MARKERS only
  duration = utterance.end - utterance.start (guarded; None → blank)
  rate = count / duration   (None if duration <= 0)

Output: utterances_v2/features/discourse_marker_per_second.csv
Header: Utterance File Name,Discourse Markers per Second
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path
from ._duration_lookup import build_duration_index, lookup_duration
from ._text import DISCOURSE_MARKERS, count_filler_hits, tokenize

FEATURE_NAME = "discourse_marker_per_second"
HEADER = ("Utterance File Name", "Discourse Markers per Second")


def compute_rate_per_second(text: str, duration: float | None) -> float | None:
    if duration is None or duration <= 0:
        return None
    words = tokenize(text)
    if not words:
        return 0.0
    return count_filler_hits(words, DISCOURSE_MARKERS) / duration


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def write_discourse_markers_per_second(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    durations = build_duration_index(transcript_root)
    n = 0
    missing = 0
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
            d = lookup_duration(durations, rel)
            if d is None:
                missing += 1
            writer.writerow([rel, _fmt(compute_rate_per_second(text, d))])
            n += 1
    if missing:
        print(f"  warning: {missing} rows had no transcript-derived duration")
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_discourse_markers_per_second(
        manifest_path(out_root),
        out_root / "features" / "discourse_marker_per_second.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} discourse-markers-per-second rows")
    return 0
