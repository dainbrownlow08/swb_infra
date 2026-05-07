"""Pitch features per utterance, computed from sliced WAV files.

Tool: librosa.pyin (probabilistic YIN). Same algorithm as the legacy paper,
with three corrections vs. legacy:
  1. sr=None preserves the native 8kHz of Switchboard audio (legacy default
     of sr=22050 silently upsampled, wasting CPU without improving F0).
  2. fmin=50, fmax=400. Tightens the search range to physically plausible
     adult-speech F0 (legacy used C2..C7 = 65..2093 Hz, allowing octave
     errors above 500 Hz). The lower bound of 50 Hz captures vocal fry /
     creaky voice common in conversational speech; the 400 Hz ceiling sits
     above any real adult F0.
  3. frame_length scaled to sample rate. librosa's default 2048 samples is
     calibrated for 22050 Hz (~93 ms window). At Switchboard's 8 kHz native
     rate, that 256 ms window was too coarse for pyin's HMM to lock onto
     F0 in roughly 10% of utterances. We use 512 samples (~64 ms) for
     sr <= 11025, restoring the time-scale and recovering most of those
     missing rows. As a side benefit, the tighter time-resolution also
     fixes octave errors where pyin previously locked onto 2*F0.

Output: utterances_v2/features/pitch.csv
Header: Utterance File Name,pitch mean,pitch std,pitch range
Empty cell ("") is written for utterances with no voiced frames; pandas
reads these as NaN, distinguishing 'no measurement' from 'measured 0'.
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path

FEATURE_NAME = "pitch"
HEADER = ("Utterance File Name", "pitch mean", "pitch std", "pitch range")

PITCH_FMIN = 50   # Hz; lower edge of vocal fry / creaky voice
PITCH_FMAX = 400  # Hz; hard ceiling above which any F0 reading is an octave error


def _frame_length_for_sr(sr: int) -> int:
    """Match librosa's default 22050-Hz time-scale (~93 ms) at any sample rate.

    8 kHz telephone audio (Switchboard) → 512 samples (~64 ms).
    22050 Hz default → 2048 samples (~93 ms).
    """
    return 512 if sr <= 11025 else 2048


def extract_pitch(
    wav_path: Path,
) -> tuple[float | None, float | None, float | None]:
    """Compute (pitch_mean, pitch_std, pitch_range) in Hz over voiced frames.

    Returns (None, None, None) if no voiced frames are detected.
    Pure function so it picklesable for ProcessPoolExecutor.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav_path), sr=None)
    f0, _voiced_flag, _voiced_prob = librosa.pyin(
        y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr,
        frame_length=_frame_length_for_sr(sr),
    )
    voiced = f0[~np.isnan(f0)]
    if voiced.size == 0:
        return None, None, None
    return (
        float(np.mean(voiced)),
        float(np.std(voiced)),
        float(np.max(voiced) - np.min(voiced)),
    )


def _worker(arg: tuple[str, str]) -> tuple[str, tuple[float | None, float | None, float | None]]:
    rel, abs_path = arg
    return rel, extract_pitch(Path(abs_path))


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def _read_existing(output_csv: Path) -> dict[str, list[str]]:
    """Map Utterance File Name → already-written row (for resume)."""
    if not output_csv.exists():
        return {}
    with open(output_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != HEADER:
            return {}
        return {row[0]: row for row in reader if row}


def write_pitch(
    manifest_csv: Path,
    output_csv: Path,
    out_root: Path,
    workers: int = 4,
    limit: int = 0,
    overwrite: bool = False,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read manifest in order
    with open(manifest_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest_csv}: {header!r}"
            )
        rels = [row[0] for row in reader if row]
    if limit:
        rels = rels[:limit]

    # 2. Resume: skip already-extracted rels
    cache: dict[str, list[str]] = {} if overwrite else _read_existing(output_csv)
    needs_work = [r for r in rels if r not in cache]
    print(
        f"pitch: {len(rels):,} total, {len(cache):,} cached, "
        f"{len(needs_work):,} to extract (workers={workers})"
    )

    # 3. Extract missing in parallel
    fresh: dict[str, tuple[float | None, float | None, float | None]] = {}
    if needs_work:
        work = [(r, str(out_root / r)) for r in needs_work]
        if workers <= 1:
            for arg in work:
                rel, result = _worker(arg)
                fresh[rel] = result
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                # chunksize: tradeoff between dispatch overhead and progress visibility
                done = 0
                last_log = 0
                for rel, result in ex.map(_worker, work, chunksize=8):
                    fresh[rel] = result
                    done += 1
                    if done - last_log >= 1000 or done == len(work):
                        print(f"  {done:,}/{len(work):,}", flush=True)
                        last_log = done

    # 4. Write output in manifest order
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for rel in rels:
            if rel in cache:
                writer.writerow(cache[rel])
            else:
                m, s, r = fresh[rel]
                writer.writerow([rel, _fmt(m), _fmt(s), _fmt(r)])
    return len(rels)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_pitch(
        manifest_path(out_root),
        out_root / "features" / "pitch.csv",
        out_root=out_root,
        workers=args.workers,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"wrote {n} pitch rows")
    return 0
