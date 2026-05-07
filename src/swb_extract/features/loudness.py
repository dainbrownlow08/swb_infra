"""Loudness features per utterance, computed from sliced WAV files.

Same algorithm and parameters as the legacy paper (FELoud.py):
  y, sr = librosa.load(path)              # sr=22050 default; upsamples 8kHz audio
  S = |librosa.stft(y)|                   # magnitude spectrogram (default n_fft=2048)
  rms = librosa.feature.rms(S=S)          # per-frame RMS energy via Parseval
  → mean(rms), std(rms), max(rms)-min(rms)

Output: utterances_v2/features/loudness.csv
Header: Utterance File Name,loudness mean,loudness std,loudness range
Linear RMS units. 0.0 is a valid measurement (silent audio); empty cell is
reserved for processing failures (missing/corrupted file).
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path

FEATURE_NAME = "loudness"
HEADER = ("Utterance File Name", "loudness mean", "loudness std", "loudness range")


def extract_loudness(
    wav_path: Path,
) -> tuple[float | None, float | None, float | None]:
    """Compute (loudness_mean, loudness_std, loudness_range) in linear RMS.

    Returns (None, None, None) only on file/format errors. RMS=0.0 is a valid
    measurement for silent audio (matches legacy semantics).
    Pure function, picklable for ProcessPoolExecutor.
    """
    import librosa
    import numpy as np

    try:
        y, sr = librosa.load(str(wav_path))  # sr=22050 default — matches legacy
    except Exception:
        return None, None, None
    if y.size == 0:
        return None, None, None

    S, _ = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    return (
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.max(rms) - np.min(rms)),
    )


def _worker(arg: tuple[str, str]) -> tuple[str, tuple[float | None, float | None, float | None]]:
    rel, abs_path = arg
    return rel, extract_loudness(Path(abs_path))


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def _read_existing(output_csv: Path) -> dict[str, list[str]]:
    if not output_csv.exists():
        return {}
    with open(output_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != HEADER:
            return {}
        return {row[0]: row for row in reader if row}


def write_loudness(
    manifest_csv: Path,
    output_csv: Path,
    out_root: Path,
    workers: int = 4,
    limit: int = 0,
    overwrite: bool = False,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

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

    cache: dict[str, list[str]] = {} if overwrite else _read_existing(output_csv)
    needs_work = [r for r in rels if r not in cache]
    print(
        f"loudness: {len(rels):,} total, {len(cache):,} cached, "
        f"{len(needs_work):,} to extract (workers={workers})"
    )

    fresh: dict[str, tuple[float | None, float | None, float | None]] = {}
    if needs_work:
        work = [(r, str(out_root / r)) for r in needs_work]
        if workers <= 1:
            for arg in work:
                rel, result = _worker(arg)
                fresh[rel] = result
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                done = 0
                last_log = 0
                for rel, result in ex.map(_worker, work, chunksize=8):
                    fresh[rel] = result
                    done += 1
                    if done - last_log >= 1000 or done == len(work):
                        print(f"  {done:,}/{len(work):,}", flush=True)
                        last_log = done

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
    n = write_loudness(
        manifest_path(out_root),
        out_root / "features" / "loudness.csv",
        out_root=out_root,
        workers=args.workers,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"wrote {n} loudness rows")
    return 0
