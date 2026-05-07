from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DURATION_TOLERANCE_SEC = 0.005


def _sox_bin() -> str:
    p = shutil.which("sox")
    if not p:
        raise RuntimeError("sox not found on PATH")
    return p


def _soxi_bin() -> str:
    p = shutil.which("soxi")
    if not p:
        raise RuntimeError("soxi not found on PATH")
    return p


def slice_utterance(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    *,
    overwrite: bool = False,
) -> None:
    if end <= start:
        raise ValueError(f"end must be > start (got {start=} {end=})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return

    cmd = [_sox_bin(), str(src), str(dst), "trim", f"{start}", f"={end}"]
    subprocess.run(cmd, check=True, capture_output=True)

    actual = _probe_duration(dst)
    expected = end - start
    if abs(actual - expected) > DURATION_TOLERANCE_SEC:
        raise RuntimeError(
            f"slice duration {actual:.6f}s != expected {expected:.6f}s "
            f"(src={src}, dst={dst}, start={start}, end={end})"
        )


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        [_soxi_bin(), "-D", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())
