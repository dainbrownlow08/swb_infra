"""Demographic features for each utterance, derived from LDC tables.

Source of truth (shipped by LDC as part of Switchboard-1):
- tables/caller_tab.csv : per-speaker sex, birth_year, dialect_area, education
- tables/conv_tab.csv   : per-conversation caller_from (A side), caller_to (B side)

Output: utterances_v2/features/demographics.csv
Header: Utterance File Name,Gender,Region,Year Born,Generation,Decade,Education
        (matches the legacy paper's features.csv first 8 columns verbatim)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path

FEATURE_NAME = "demographics"
HEADER = (
    "Utterance File Name",
    "Gender",
    "Region",
    "Year Born",
    "Generation",
    "Decade",
    "Education",
)

# Columns in caller_tab.csv (per tables/caller_doc.txt)
CALLER_COL_SEX = 3
CALLER_COL_BIRTH_YEAR = 4
CALLER_COL_DIALECT = 5
CALLER_COL_EDUCATION = 6
# Columns in conv_tab.csv (per tables/conv_doc.txt)
CONV_COL_NO = 0
CONV_COL_FROM = 2  # A side
CONV_COL_TO = 3    # B side


@dataclass(frozen=True)
class Demographics:
    gender: str
    region: str
    year_born: int
    generation: str
    decade: str
    education: str  # numeric code as string ("0"-"3" or "9" per LDC)


def normalize_sex(raw: str) -> str:
    s = raw.strip().upper()
    if s == "MALE":
        return "male"
    if s == "FEMALE":
        return "female"
    raise ValueError(f"unrecognized sex: {raw!r}")


def normalize_region(raw: str) -> str:
    s = raw.strip().upper()
    if not s or s == "UNK":
        return "unk"
    return s.lower().replace(" ", "_")


def derive_generation(birth_year: int) -> str:
    """Cutoffs match those used in the legacy Switchboard paper.
    The legacy file_listing labels show: GI=1924, Silent=[1927-1944],
    Baby_Boomer=[1945-1964], Generation_X=[1965-1975].
    """
    if birth_year <= 1926:
        return "GI"
    if birth_year <= 1944:
        return "Silent"
    if birth_year <= 1964:
        return "Baby_Boomer"
    return "Generation_X"


def derive_decade(birth_year: int) -> str:
    return f"{(birth_year // 10) * 10}s"


def _strip_csv_field(s: str) -> str:
    return s.strip().strip('"')


def load_caller_tab(path: Path) -> dict[int, Demographics]:
    out: dict[int, Demographics] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) <= CALLER_COL_EDUCATION:
                continue
            try:
                caller_no = int(_strip_csv_field(row[0]))
            except ValueError:
                continue
            sex = _strip_csv_field(row[CALLER_COL_SEX])
            year_str = _strip_csv_field(row[CALLER_COL_BIRTH_YEAR])
            dialect = _strip_csv_field(row[CALLER_COL_DIALECT])
            education = _strip_csv_field(row[CALLER_COL_EDUCATION])
            if not year_str.isdigit():
                continue
            year = int(year_str)
            out[caller_no] = Demographics(
                gender=normalize_sex(sex),
                region=normalize_region(dialect),
                year_born=year,
                generation=derive_generation(year),
                decade=derive_decade(year),
                education=education,
            )
    return out


def load_conv_tab(path: Path) -> dict[int, tuple[int, int]]:
    """call_id → (A-side caller, B-side caller)."""
    out: dict[int, tuple[int, int]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) <= CONV_COL_TO:
                continue
            try:
                cid = int(_strip_csv_field(row[CONV_COL_NO]))
                a = int(_strip_csv_field(row[CONV_COL_FROM]))
                b = int(_strip_csv_field(row[CONV_COL_TO]))
            except ValueError:
                continue
            out[cid] = (a, b)
    return out


def build_speaker_demos(
    caller_path: Path, conv_path: Path
) -> dict[tuple[int, str], Demographics]:
    callers = load_caller_tab(caller_path)
    convs = load_conv_tab(conv_path)
    out: dict[tuple[int, str], Demographics] = {}
    for cid, (a, b) in convs.items():
        if a in callers:
            out[(cid, "A")] = callers[a]
        if b in callers:
            out[(cid, "B")] = callers[b]
    return out


def write_demographics(
    manifest_csv: Path,
    output_csv: Path,
    demos_by_callside: dict[tuple[int, str], Demographics],
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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
        missing: list[tuple[int, str]] = []
        for row in reader:
            if not row:
                continue
            rel = row[0]
            call_id, side, _utt = parse_rel_path(rel)
            d = demos_by_callside.get((call_id, side))
            if d is None:
                missing.append((call_id, side))
                continue
            writer.writerow([
                rel,
                d.gender,
                d.region,
                d.year_born,
                d.generation,
                d.decade,
                d.education,
            ])
            n += 1
    if missing:
        raise RuntimeError(
            f"missing demographics for {len(missing)} (call,side) pairs; "
            f"first few: {missing[:5]}"
        )
    return n


def run(args) -> int:
    caller = Path(args.caller_tab)
    conv = Path(args.conv_tab)
    out_root = Path(args.out_root)
    demos = build_speaker_demos(caller, conv)
    n = write_demographics(
        manifest_path(out_root),
        out_root / "features" / "demographics.csv",
        demos,
    )
    print(f"wrote {n} demographic rows for {len(demos)} (call,side) speakers")
    return 0
