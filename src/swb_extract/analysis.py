"""Single entry point for loading the canonical feature table in analysis.

Trustworthy / experimental notebooks import from here instead of hardcoding a
``read_csv`` path, so the data location lives in exactly one place and two guards
run on every load:

  * stale-data guard — errors if any ``features/*.csv`` or ``manifest.csv`` is
    newer than the built table, telling you to ``swb-extract table``. This is what
    makes "you're on stale data" a hard failure instead of silent confusion.
  * registry guard — errors if the table has an unregistered column (or the
    registry names a column the table lacks), forcing every new feature to be
    registered with a trust status before it can be used.

``include`` selects by trust level so a notebook can ask for only ``validated``
data, or ``provisional`` (validated + unconfirmed, the experimentation default,
with a printed warning naming the provisional columns).
"""
from __future__ import annotations

from pathlib import Path

from . import registry as R

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "utterances_v2"
TABLE_RELPATH = Path("derived") / "features_table.csv"

# trust ladder: include="X" keeps every status at or above X.
_INCLUDE_LEVELS = {
    "validated": ("validated",),
    "provisional": ("validated", "provisional"),
    "all": ("validated", "provisional", "deprecated"),
}


def table_path(out_root: Path | str | None = None) -> Path:
    root = Path(out_root) if out_root else DEFAULT_OUT_ROOT
    return root / TABLE_RELPATH


def assert_table_fresh(out_root: Path | str | None = None) -> None:
    """Raise if the canonical table is missing or older than any of its inputs."""
    root = Path(out_root) if out_root else DEFAULT_OUT_ROOT
    tbl = root / TABLE_RELPATH
    if not tbl.exists():
        raise FileNotFoundError(
            f"canonical table missing: {tbl}\n  run:  swb-extract table"
        )
    t = tbl.stat().st_mtime
    inputs = [root / "manifest.csv", *sorted((root / "features").glob("*.csv"))]
    newer = [p.name for p in inputs if p.exists() and p.stat().st_mtime > t]
    if newer:
        raise RuntimeError(
            f"STALE canonical table — these inputs are newer than {tbl.name}:\n"
            f"  {newer}\n"
            f"  run:  swb-extract table   (then reload)"
        )


def load_features_table(
    *,
    include: str = "provisional",
    family: str | None = None,
    with_meta: bool = True,
    out_root: Path | str | None = None,
    check_fresh: bool = True,
):
    """Load the canonical per-utterance table as a DataFrame.

    include   : trust floor — "validated" | "provisional" (default) | "all".
    family    : optional filter — volume|interactional|prosody|tannen|meta.
    with_meta : keep reserved key/text columns (Utterance File Name, Transcript).
    """
    import pandas as pd

    if include not in _INCLUDE_LEVELS:
        raise ValueError(f"include must be one of {tuple(_INCLUDE_LEVELS)}, got {include!r}")
    root = Path(out_root) if out_root else DEFAULT_OUT_ROOT
    tbl = root / TABLE_RELPATH
    if check_fresh:
        assert_table_fresh(root)
    R.validate_against_table(tbl)

    statuses = _INCLUDE_LEVELS[include]
    keep = [
        c for c in R.REGISTRY
        if R.status_of(c) in statuses and (family is None or R.family_of(c) == family)
    ]
    usecols = (list(R.KEY_COLUMNS) if with_meta else []) + keep
    df = pd.read_csv(tbl, usecols=lambda c: c in usecols)

    prov = [c for c in keep if R.status_of(c) == "provisional"]
    if prov:
        shown = ", ".join(prov[:6]) + ("  …" if len(prov) > 6 else "")
        print(
            f"⚠ loaded {len(keep)} feature cols including {len(prov)} PROVISIONAL "
            f"(unconfirmed) — trust with care:\n    {shown}"
        )
    return df
