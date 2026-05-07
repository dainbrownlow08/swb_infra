from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# sw<CALL><SIDE>-ms98-a-trans.text
_FILENAME_RE = re.compile(r"^sw(\d+)([AB])-ms98-a-trans\.text$")
# sw<CALL><SIDE>-ms98-a-<NNNN>
_ID_RE = re.compile(r"^sw(\d+)([AB])-ms98-a-(\d+)$")


@dataclass(frozen=True)
class Utterance:
    call_id: int
    side: str
    utt_num: int
    start: float
    end: float
    text: str


def _parse_filename(path: Path) -> tuple[int, str]:
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"unrecognized transcript filename: {path.name}")
    return int(m.group(1)), m.group(2)


def parse_transcript(path: Path) -> Iterator[Utterance]:
    expected_call, expected_side = _parse_filename(path)
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=3)
            if len(parts) < 4:
                continue
            id_field, start_s, end_s, text = parts
            m = _ID_RE.match(id_field)
            if not m:
                raise ValueError(f"{path}: unrecognized utterance id: {id_field!r}")
            call_id, side, utt_str = int(m.group(1)), m.group(2), m.group(3)
            if call_id != expected_call or side != expected_side:
                raise ValueError(
                    f"{path}: id {id_field!r} does not match filename "
                    f"(expected call={expected_call} side={expected_side})"
                )
            yield Utterance(
                call_id=call_id,
                side=side,
                utt_num=int(utt_str),
                start=float(start_s),
                end=float(end_s),
                text=text,
            )


def iter_transcript_paths(root: Path) -> Iterator[Path]:
    yield from sorted(root.glob("*/*/sw*-ms98-a-trans.text"))
