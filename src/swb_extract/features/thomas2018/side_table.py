"""Per-(call, side) table of the eleven variables — the paper's unit of analysis.

Zip-merges manifest.csv with features/{ppron,…,repu}.csv (per-row key assertions, as
features_table does), groups utterances by conversation side, and applies each
module's ``aggregate``. Output: utterances_v2/derived/thomas2018_side.csv with columns
call_id, side, n_utts, ppron, wps, wpu, wpp, boplen, poplen, pv, lv, olap, rept, repu
in raw units (no z-scoring: the paper rescales each to zero mean / unit SD before
weighting by ``PAPER_LOADINGS``). ``aggregate_group(rows)`` is the same computation
for any grouping of canonical-table rows — e.g. caller-level pooling in NB08.
"""
from __future__ import annotations

import csv
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator

from ...manifest import manifest_path
from . import EXTRACTORS, VARIABLES
from ._io import Row, fmt, iter_manifest, key_of

OUTPUT_RELPATH = Path("derived") / "thomas2018_side.csv"
HEADER = ("call_id", "side", "n_utts", *VARIABLES)


def aggregate_group(rows: list[Row]) -> dict[str, float | None]:
    """The eleven paper statistics over one group of utterance rows."""
    return {m.FEATURE_NAME: m.aggregate(rows) for m in EXTRACTORS}


def iter_merged(manifest_csv: Path, features_dir: Path) -> Iterator[dict[str, str]]:
    """Manifest rows joined with the eleven CSVs, in lockstep, keys asserted per row."""
    with ExitStack() as stack:
        readers = []
        for m in EXTRACTORS:
            path = features_dir / f"{m.FEATURE_NAME}.csv"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing {path} — run `swb-extract features {m.FEATURE_NAME}` first"
                )
            reader = csv.reader(stack.enter_context(open(path, encoding="utf-8", newline="")))
            header = next(reader, None)
            if tuple(header or ()) != m.HEADER:
                raise RuntimeError(
                    f"{path.name}: header {header!r} != {m.HEADER} — re-run the extractor"
                )
            readers.append((path.name, m.HEADER[1:], reader))
        for i, (rel, _text) in enumerate(iter_manifest(manifest_csv), 1):
            row = {"Utterance File Name": rel}
            for name, cols, reader in readers:
                frow = next(reader, None)
                if frow is None or frow[0] != rel:
                    got = frow[0] if frow else "<end of file>"
                    raise RuntimeError(
                        f"{name} out of sync at manifest row {i}: expected {rel!r}, "
                        f"got {got!r} — re-run the extractor"
                    )
                vals = frow[1:] + [""] * (len(cols) - len(frow) + 1)
                row.update(zip(cols, vals))
            yield row


def write_side_table(manifest_csv: Path, features_dir: Path, output_csv: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n_sides = 0
    n_unplaceable = 0
    seen: set[tuple[int, str]] = set()
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        current: tuple[int, str] | None = None
        rows: list[Row] = []

        def flush() -> None:
            nonlocal n_sides
            if current is None:
                return
            agg = aggregate_group(rows)
            writer.writerow([current[0], current[1], len(rows), *(fmt(agg[v]) for v in VARIABLES)])
            n_sides += 1

        for row in iter_merged(manifest_csv, features_dir):
            key = key_of(row["Utterance File Name"])
            if key is None:
                n_unplaceable += 1
                continue
            side_key = (key[0], key[1])
            if side_key != current:
                flush()
                if side_key in seen:  # manifest rows of one side are contiguous by construction
                    raise RuntimeError(f"side {side_key} appears twice in manifest order")
                seen.add(side_key)
                current, rows = side_key, []
            rows.append(row)
        flush()
    if n_unplaceable:
        print(f"  skipped {n_unplaceable} rows with unparseable paths")
    return n_sides


def run(args) -> int:
    out_root = Path(args.out_root)
    output_csv = out_root / OUTPUT_RELPATH
    n = write_side_table(manifest_path(out_root), out_root / "features", output_csv)
    print(f"wrote {output_csv}: {n} conversation sides x {len(VARIABLES)} variables")
    return 0
