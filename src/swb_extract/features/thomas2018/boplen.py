"""boplen — mean length of between-own pauses (Thomas et al. 2018 §3.2, "Pauses, turn-taking").

"A between-own pause is a period during a single utterance where there is no speech
signal (periods where OpenSMILE reports no F0)" — Figure 2: the pause between "see" and
"the" inside one speaker's utterance. Here: a maximal run of unvoiced frames bounded on
both sides by voiced frames of the same utterance slice (librosa.pyin on the in-house
pitch settings — ``_prosody``; leading/trailing silence is slice padding, not a pause
between the speaker's own words). Taken literally, as the paper did, every unvoiced run
counts — voiceless-consonant stretches included — which is why the paper's boplen is
"on the order of 0.1 s" (golden utterance sw2001A-U0002: 19 pauses, mean 0.16 s). Per
utterance this module writes the pause count and their total seconds; the paper's
boplen is Σsec / Σn over the side (``aggregate``), not a mean of per-utterance means.
Loading on involvement: −0.10. The in-house ``Within Pause`` family is the
alignment-based (silence ≥ 0.25 s between aligned words) alternative.

Output: utterances_v2/features/boplen.csv   (shares the _prosody cache; see --workers/--limit/--overwrite)
Header: Utterance File Name,boplen n,boplen sec
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ._io import Row, fmt, ratio_of_sums
from ._prosody import Prosody, write_projection

FEATURE_NAME = "boplen"
HEADER = ("Utterance File Name", "boplen n", "boplen sec")


def project(p: Prosody | None) -> list[str]:
    return ["", ""] if p is None else [str(p[5]), fmt(p[6])]


def aggregate(rows: list[Row]) -> float | None:
    """Mean between-own pause length over the group: Σ(boplen sec) / Σ(boplen n)."""
    return ratio_of_sums(rows, "boplen sec", "boplen n")


def write_boplen(manifest_csv: Path, output_csv: Path, out_root: Path, **kw) -> int:
    return write_projection(manifest_csv, output_csv, out_root, HEADER, project, **kw)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_boplen(
        manifest_path(out_root), out_root / "features" / f"{FEATURE_NAME}.csv", out_root,
        workers=args.workers, limit=args.limit, overwrite=args.overwrite,
    )
    print(f"wrote {n} boplen rows")
    return 0
