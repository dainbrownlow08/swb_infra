from __future__ import annotations

import json
import re
from pathlib import Path

_WAV_RE = re.compile(r"^sw(\d{5})\.([AB])\.wav$")

AudioIndex = dict[tuple[int, str], Path]


def build_index(audio_root: Path) -> AudioIndex:
    idx: AudioIndex = {}
    for p in audio_root.rglob("sw*.wav"):
        m = _WAV_RE.match(p.name)
        if not m:
            continue
        key = (int(m.group(1)), m.group(2))
        if key in idx and idx[key] != p:
            raise RuntimeError(
                f"duplicate audio for {key}: {idx[key]} and {p}"
            )
        idx[key] = p
    return idx


def save_index(idx: AudioIndex, cache_path: Path) -> None:
    payload = [
        {"call_id": k[0], "side": k[1], "path": str(v)}
        for k, v in sorted(idx.items())
    ]
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_index(cache_path: Path) -> AudioIndex:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return {(e["call_id"], e["side"]): Path(e["path"]) for e in payload}


def resolve(idx: AudioIndex, call_id: int, side: str) -> Path:
    try:
        return idx[(call_id, side)]
    except KeyError:
        raise KeyError(f"no audio for call_id={call_id} side={side}") from None
