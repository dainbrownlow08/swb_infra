"""poplen — mean length of post-other pauses (Thomas et al. 2018 §3.2, "Pauses, turn-taking").

"A post-other pause is the gap between the end of one participant's utterance and the
start of their partner's next. In marking post-other pauses, we allowed for cases where
participants did not strictly alternate." Figure 2: the pause between P5's "thing" and
P6's "I'll". Per utterance this module writes that gap in seconds, attributed to the
utterance it precedes (the partner's next), on word-tight bounds via
``_bounds.onset_timing``: blank when the utterance starts in overlap (no pause exists —
that is an olap event), when the speaker's own talk is more recent than the partner's
(an own-own gap), or before the partner has spoken. Every transcript line counts —
backchannels included, as in the paper; the in-house FTO family is the
backchannel-aware response-timing measure. The paper's poplen is the mean over defined
rows (``aggregate``). Loading on involvement: −0.27.

Output: utterances_v2/features/poplen.csv
Header: Utterance File Name,poplen
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ._bounds import build_onset_timing
from ._io import Row, fmt, key_of, mean_of, write_feature_csv

FEATURE_NAME = "poplen"
HEADER = ("Utterance File Name", "poplen")


def aggregate(rows: list[Row]) -> float | None:
    return mean_of(rows, "poplen")


def write_poplen(manifest_csv: Path, output_csv: Path, transcript_root: Path) -> int:
    timing = build_onset_timing(transcript_root)

    def cells(rel: str, _text: str) -> list[str]:
        key = key_of(rel)
        t = timing.get(key) if key else None
        return [fmt(t[1]) if t else ""]

    return write_feature_csv(manifest_csv, output_csv, HEADER, cells)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_poplen(
        manifest_path(out_root),
        out_root / "features" / f"{FEATURE_NAME}.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} poplen rows")
    return 0
