"""wpp — words per between-own pause (Thomas et al. 2018 §3.2, "Pauses, turn-taking").

"Words per between-own pauses; approximately the length of each spoken phrase. Again
this is an overall average, calculated as the number of words in the transcript divided
by the total number of between-own pauses, and not a per-utterance value." Per
utterance this module writes the number of between-own pauses — unvoiced runs bounded
by voiced frames inside the utterance (``_prosody``: the paper's "periods where
OpenSMILE reports no F0"); the paper's wpp is Σwpu / Σ(wpp pauses) (``aggregate``).
The count is identical by construction to ``boplen n`` — each acoustic CSV carries what
its own statistic needs. Taken literally every unvoiced run counts, so wpp is closer to
"words per voiced stretch" than to phrase length; the paper's own caveat is the
"approximately". Loading on involvement: +0.44.

Output: utterances_v2/features/wpp.csv   (shares the _prosody cache; see --workers/--limit/--overwrite)
Header: Utterance File Name,wpp pauses
"""
from __future__ import annotations

from pathlib import Path

from ...manifest import manifest_path
from ._io import Row, ratio_of_sums
from ._prosody import Prosody, write_projection

FEATURE_NAME = "wpp"
HEADER = ("Utterance File Name", "wpp pauses")


def project(p: Prosody | None) -> list[str]:
    return [""] if p is None else [str(p[5])]


def aggregate(rows: list[Row]) -> float | None:
    """Words per between-own pause over the group: Σwpu / Σ(wpp pauses)."""
    return ratio_of_sums(rows, "wpu", "wpp pauses")


def write_wpp(manifest_csv: Path, output_csv: Path, out_root: Path, **kw) -> int:
    return write_projection(manifest_csv, output_csv, out_root, HEADER, project, **kw)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_wpp(
        manifest_path(out_root), out_root / "features" / f"{FEATURE_NAME}.csv", out_root,
        workers=args.workers, limit=args.limit, overwrite=args.overwrite,
    )
    print(f"wrote {n} wpp rows")
    return 0
