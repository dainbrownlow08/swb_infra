"""Feature trust registry — parsed from ``docs/FEATURES.md`` (the living document).

``docs/FEATURES.md`` is the single, human-editable source of truth: every feature
column is a row in a markdown table under a ``## Trusted`` / ``## WIP`` /
``## Deprecated`` section, and the section it lives under *is* its trust status.
To change a feature's trust you move its row between sections — nothing else.

This module parses that doc once on import into ``REGISTRY`` and exposes the
selectors the loader (``swb_extract.analysis``) uses. Parsing is strict: a bad
family, a duplicate column, or an empty doc raises immediately, so the doc can
never silently disagree with the code.
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs" / "FEATURES.md"

# Reserved columns that are not features (manifest key + transcript text).
KEY_COLUMNS = ("Utterance File Name", "Transcript")

VALID_STATUSES = ("validated", "provisional", "deprecated")
VALID_FAMILIES = ("volume", "interactional", "prosody", "tannen", "meta")

# First word of a "## <Heading>" (lowercased) -> status. Other sections (title,
# Dashboard, …) map to None and their rows are ignored.
_SECTION_STATUS = {"trusted": "validated", "wip": "provisional", "deprecated": "deprecated"}


def _parse_doc(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"feature registry doc missing: {path}")
    registry: dict[str, dict] = {}
    status: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            words = line.lstrip("#").strip().lower().split()
            status = _SECTION_STATUS.get(words[0]) if words else None
            continue
        if status is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        column = cells[0]
        if column.lower() == "column" or set(column) <= set("-: "):  # header / separator
            continue
        extractor, family = cells[1], cells[2]
        note = cells[3] if len(cells) > 3 else ""
        if family not in VALID_FAMILIES:
            raise ValueError(
                f"{path.name}: column {column!r} has unknown family {family!r}; "
                f"valid: {VALID_FAMILIES}"
            )
        if column in registry:
            raise ValueError(f"{path.name}: duplicate column {column!r}")
        registry[column] = {
            "family": family, "status": status, "extractor": extractor, "note": note,
        }
    if not registry:
        raise ValueError(f"{path.name}: parsed no feature rows — is the table format intact?")
    return registry


REGISTRY: dict[str, dict] = _parse_doc(DOC_PATH)


def cols(family: str | None = None, status: str | None = None) -> list[str]:
    """Column names matching the given family and/or status (doc order)."""
    if family is not None and family not in VALID_FAMILIES:
        raise ValueError(f"unknown family {family!r}; valid: {VALID_FAMILIES}")
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"unknown status {status!r}; valid: {VALID_STATUSES}")
    return [
        c for c, e in REGISTRY.items()
        if (family is None or e["family"] == family)
        and (status is None or e["status"] == status)
    ]


def status_of(col: str) -> str:
    return REGISTRY[col]["status"]


def family_of(col: str) -> str:
    return REGISTRY[col]["family"]


def validate_against_table(table_csv: Path) -> None:
    """Assert the registry (docs/FEATURES.md) and the canonical table agree.

    Raises ValueError naming any column in the table that is unregistered, or any
    registered column absent from the table. Run after every ``swb-extract table``.
    """
    with open(table_csv, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), [])
    table_features = [c for c in header if c not in KEY_COLUMNS]
    table_set, reg_set = set(table_features), set(REGISTRY)
    unregistered = [c for c in table_features if c not in reg_set]
    missing = [c for c in REGISTRY if c not in table_set]
    problems = []
    if unregistered:
        problems.append(f"in table but NOT in docs/FEATURES.md (add a row): {unregistered}")
    if missing:
        problems.append(f"in docs/FEATURES.md but NOT in table (stale row, or rebuild): {missing}")
    if problems:
        raise ValueError("registry / table mismatch:\n  " + "\n  ".join(problems))


def summary() -> str:
    """Counts by family x status — the at-a-glance trust dashboard."""
    lines = [f"{'family':<14}" + "".join(f"{s:>13}" for s in VALID_STATUSES) + f"{'total':>8}"]
    for fam in VALID_FAMILIES:
        row = [len(cols(fam, s)) for s in VALID_STATUSES]
        lines.append(f"{fam:<14}" + "".join(f"{n:>13}" for n in row) + f"{sum(row):>8}")
    tot = [len(cols(None, s)) for s in VALID_STATUSES]
    lines.append(f"{'ALL':<14}" + "".join(f"{n:>13}" for n in tot) + f"{sum(tot):>8}")
    return "\n".join(lines)
