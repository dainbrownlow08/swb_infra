from __future__ import annotations

import csv
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

MANIFEST_NAME = "manifest.csv"
MANIFEST_HEADER = ("Utterance File Name", "Transcript")

# 200/sw2001A-U0002.wav  →  call_id=2001, side="A", utt_num=2
_REL_PATH_RE = re.compile(r"^\d{3}/sw(\d+)([AB])-U(\d+)\.wav$")


def parse_rel_path(rel: str) -> tuple[int, str, int]:
    m = _REL_PATH_RE.match(rel)
    if not m:
        raise ValueError(f"unrecognized manifest path: {rel!r}")
    return int(m.group(1)), m.group(2), int(m.group(3))


def manifest_path(out_root: Path) -> Path:
    return out_root / MANIFEST_NAME


def already_done_calls(manifest: Path) -> set[tuple[int, str]]:
    if not manifest.exists():
        return set()
    done: set[tuple[int, str]] = set()
    with open(manifest, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return set()
        if tuple(header) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest}: {header!r}"
            )
        for row in reader:
            if not row:
                continue
            try:
                call_id, side, _utt = parse_rel_path(row[0])
            except ValueError:
                continue
            done.add((call_id, side))
    return done


@contextmanager
def open_appender(manifest: Path) -> Iterator[csv.writer]:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    new_file = not manifest.exists() or manifest.stat().st_size == 0
    f = open(manifest, "a", encoding="utf-8", newline="")
    try:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if new_file:
            writer.writerow(MANIFEST_HEADER)
            f.flush()
        yield writer
        f.flush()
    finally:
        f.close()


def write_row(writer: csv.writer, rel_wav: str, text: str) -> None:
    writer.writerow([rel_wav, text])
