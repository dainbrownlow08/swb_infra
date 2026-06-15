"""Canonical per-utterance feature table — manifest + every feature CSV.

The version-controlled replacement for the unscripted assembly of
``merge_test.csv`` (AUDIT.md §3 fix 2). ``merge_test.csv`` stays frozen so the
existing notebooks remain replicable; new analyses load ``features_table.csv``
and select whatever ingredient columns they need. ``tannen_features.csv`` is a
separate track (pure-Tannen pipeline) and is not touched here.

The table is a pure function of its inputs, rebuilt from scratch on every run
(~seconds): ``manifest.csv`` provides the key + Transcript, then every
``features/*.csv`` contributes its value columns, in sorted filename order so
the column order is deterministic.

Implementation is a **zip-merge with per-row key assertions**, not a join:
extractors write rows in exact manifest order, so all files are streamed in
lockstep and every row's key is checked against the manifest's. Any
misalignment, duplicate, missing row, or trailing row in any feature CSV is an
immediate error naming the file and row — the silent row-scrambling that
poisoned the legacy pipeline cannot happen here. Duplicate value-column names
across feature CSVs are likewise refused. stdlib csv only; runs under either
python environment; O(1) memory.

Output: utterances_v2/features_table.csv
"""
from __future__ import annotations

import csv
from contextlib import ExitStack
from pathlib import Path

from .manifest import MANIFEST_HEADER, manifest_path

KEY = MANIFEST_HEADER[0]  # "Utterance File Name"
OUTPUT_NAME = "features_table.csv"


class TableBuildError(RuntimeError):
    pass


def build_table(
    manifest_csv: Path, features_dir: Path, output_csv: Path
) -> tuple[int, int]:
    """Build the table; returns (n_rows, n_cols). Raises TableBuildError."""
    feature_paths = sorted(p for p in features_dir.glob("*.csv"))
    if not feature_paths:
        raise TableBuildError(f"no feature CSVs in {features_dir}")

    with ExitStack() as stack:
        man_reader = csv.reader(
            stack.enter_context(open(manifest_csv, encoding="utf-8", newline=""))
        )
        man_header = next(man_reader, None)
        if tuple(man_header or ()) != MANIFEST_HEADER:
            raise TableBuildError(
                f"unexpected manifest header in {manifest_csv}: {man_header!r}"
            )

        readers: list[csv.reader] = []
        widths: list[int] = []  # value columns per feature file
        out_header: list[str] = list(man_header)
        for path in feature_paths:
            reader = csv.reader(
                stack.enter_context(open(path, encoding="utf-8", newline=""))
            )
            header = next(reader, None)
            if not header or header[0] != KEY:
                raise TableBuildError(f"{path.name}: bad header {header!r}")
            for col in header[1:]:
                if col in out_header:
                    raise TableBuildError(
                        f"{path.name}: duplicate column name {col!r}"
                    )
            out_header.extend(header[1:])
            readers.append(reader)
            widths.append(len(header) - 1)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(output_csv, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(out_header)
            for man_row in man_reader:
                if not man_row:
                    continue
                key = man_row[0]
                out_row = list(man_row)
                for path, reader, width in zip(feature_paths, readers, widths):
                    frow = next(reader, None)
                    if frow is None or frow[0] != key:
                        got = frow[0] if frow else "<end of file>"
                        raise TableBuildError(
                            f"{path.name} out of sync at manifest row {n + 1}: "
                            f"expected key {key!r}, got {got!r} — re-run the "
                            f"extractor to regenerate it in manifest order"
                        )
                    vals = frow[1:]
                    if len(vals) > width:  # malformed extractor — never truncate
                        raise TableBuildError(
                            f"{path.name} at manifest row {n + 1} (key {key!r}): "
                            f"{len(vals)} value cols, header declares {width} — "
                            f"re-run the extractor"
                        )
                    if len(vals) < width:  # trailing empty cells dropped by csv
                        vals = vals + [""] * (width - len(vals))
                    out_row.extend(vals)
                writer.writerow(out_row)
                n += 1
            for path, reader in zip(feature_paths, readers):
                if next(reader, None) is not None:
                    raise TableBuildError(
                        f"{path.name} has rows beyond the end of the manifest"
                    )
    return n, len(out_header)


def run(args) -> int:
    out_root = Path(args.out_root)
    output_csv = out_root / OUTPUT_NAME
    n_rows, n_cols = build_table(
        manifest_path(out_root), out_root / "features", output_csv
    )
    print(f"wrote {output_csv}: {n_rows} rows x {n_cols} cols")
    return 0
