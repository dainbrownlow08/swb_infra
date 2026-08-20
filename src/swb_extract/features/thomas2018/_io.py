"""Manifest-ordered CSV plumbing + the three pooled statistics shared by the eleven.

Every extractor writes one row per manifest utterance in manifest order (the
``features_table`` zip-merge contract), blank = not measurable. The ``aggregate``
helpers take any re-iterable sequence of row mappings (csv.DictReader rows, canonical
table records, ``DataFrame.to_dict("records")``) and skip blank / None / NaN cells.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from ...manifest import MANIFEST_HEADER, parse_rel_path

Key = tuple[int, str, int]  # (call_id, side, utt_num)
Row = Mapping[str, object]


def iter_manifest(manifest_csv: Path) -> Iterator[tuple[str, str]]:
    """(rel_path, transcript) per manifest row, in order; header checked."""
    with open(manifest_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest_csv}: {header!r}"
            )
        for row in reader:
            if row:
                yield row[0], (row[1] if len(row) > 1 else "")


def write_feature_csv(
    manifest_csv: Path,
    output_csv: Path,
    header: tuple[str, ...],
    cells_for: Callable[[str, str], list[str]],
) -> int:
    """Write ``[rel, *cells_for(rel, transcript)]`` per manifest row; returns the count."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for rel, text in iter_manifest(manifest_csv):
            writer.writerow([rel, *cells_for(rel, text)])
            n += 1
    return n


def key_of(rel: str) -> Key | None:
    try:
        return parse_rel_path(rel)
    except ValueError:
        return None


def fmt(v: int | float | None) -> str:
    """Cell text: ints verbatim, floats via repr (round-trip exact), None blank."""
    if v is None:
        return ""
    return str(v) if isinstance(v, int) else repr(float(v))


def num(cell: object) -> float | None:
    """Cell → float; None for blank / None / NaN."""
    if cell is None or cell == "":
        return None
    v = float(cell)  # type: ignore[arg-type]
    return None if math.isnan(v) else v


def mean_of(rows: Iterable[Row], col: str) -> float | None:
    vals = [v for v in (num(r.get(col)) for r in rows) if v is not None]
    return sum(vals) / len(vals) if vals else None


def ratio_of_sums(rows: Iterable[Row], numerator: str, denominator: str) -> float | None:
    """Σnumerator / Σdenominator over rows where BOTH are measured (a micro-average)."""
    top = bottom = 0.0
    for r in rows:
        a, b = num(r.get(numerator)), num(r.get(denominator))
        if a is None or b is None:
            continue
        top += a
        bottom += b
    return top / bottom if bottom > 0 else None


def pooled_variance(
    rows: Iterable[Row], n_col: str, mean_col: str, var_col: str
) -> float | None:
    """Population variance of all frames pooled across rows, from per-row (n, mean, var):
    N = Σnᵢ, μ = Σnᵢμᵢ/N, Var = Σnᵢ(vᵢ + μᵢ²)/N − μ²."""
    total_n = sum_x = sum_x2 = 0.0
    for r in rows:
        n, m, v = num(r.get(n_col)), num(r.get(mean_col)), num(r.get(var_col))
        if not n or m is None or v is None:
            continue
        total_n += n
        sum_x += n * m
        sum_x2 += n * (v + m * m)
    if total_n <= 0:
        return None
    mu = sum_x / total_n
    return max(sum_x2 / total_n - mu * mu, 0.0)
