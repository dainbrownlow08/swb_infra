"""Loudness features per utterance — word-tight RMS over sliced WAV files.

**2026-08-19 redesign (AUDIT.md §3 risk 6).** The legacy algorithm (and this
module's first version) averaged frame RMS over the *entire* trans-span slice,
silence included, so the shipped value was speech loudness × speech fraction —
quantified in `analysis/validate_loudness_dilution.py`:
log(whole) ≈ 0.95·log(word-tight) + 0.67·log(speech_frac), R² .96; median
utterance shipped at 70% of its true speech RMS. Speaker ranking survived
(side-level r ≈ .98), but the value itself conflated amplitude with speech
density. This version computes the same frame RMS and then keeps only frames
whose centers fall inside word-aligned intervals (``*-word.text``), so all four
statistics describe *speech*, not the slice:

  y, sr = librosa.load(path)              # sr=22050 default; upsamples 8kHz audio
  S = |librosa.stft(y)|                   # magnitude spectrogram (default n_fft=2048)
  rms = librosa.feature.rms(S=S)          # per-frame RMS via Parseval
  speech = rms[frame center ∈ some word interval, slice-relative via trans start]
  → mean(speech), std(speech), max(speech)-min(speech), speech_time/slice_dur

Column semantics under the redesign:
  loudness mean        — mean RMS over speech frames (silence-dilution-free)
  loudness std         — RMS spread *within speech* (no longer inflated by
                         speech/silence alternation)
  loudness range       — max−min over speech frames (no longer peak-vs-silence-
                         floor: min is now the quietest *spoken* frame)
  loudness speech frac — clipped word time / slice duration; the dilution
                         diagnostic, kept so any corpus this pipeline is pointed
                         at can audit its own padding conventions instantly

Empty cells = unmeasurable: missing/corrupted audio, no word-alignment rows for
the utterance, or no frame center inside any word interval (sub-frame words;
counted and reported). RMS 0.0 remains a valid measurement (silent-but-aligned
audio). The resume cache keys on (row, header); this redesign CHANGED the header
(added ``loudness speech frac``), so pre-redesign rows can never be silently
reused — re-running without --overwrite still recomputes everything.

Output: utterances_v2/features/loudness.csv
Header: Utterance File Name,loudness mean,loudness std,loudness range,loudness speech frac
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ...manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from ...transcripts import iter_transcript_paths, parse_transcript
from .word_align import build_word_index

FEATURE_NAME = "loudness"
HEADER = (
    "Utterance File Name",
    "loudness mean",
    "loudness std",
    "loudness range",
    "loudness speech frac",
)

# (rel_path, abs_wav_path, slice-relative word intervals, speech fraction)
WorkItem = tuple[str, str, list[tuple[float, float]], float]
Result = tuple[float | None, float | None, float | None, float | None]

_HOP = 512  # librosa.feature.rms default hop; frame center at i*hop/sr


def extract_loudness(
    wav_path: Path, intervals: list[tuple[float, float]]
) -> tuple[float | None, float | None, float | None]:
    """(mean, std, range) of frame RMS restricted to word intervals.

    ``intervals`` are slice-relative (start, end) seconds. Returns
    (None, None, None) on file/format errors or when no frame center falls
    inside any interval. RMS 0.0 is a valid measurement for silent audio.
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
    rms = librosa.feature.rms(S=S)[0]
    t = np.arange(len(rms)) * _HOP / sr
    mask = np.zeros(len(rms), dtype=bool)
    for ws, we in intervals:
        mask |= (t >= ws) & (t < we)
    if not mask.any():
        return None, None, None
    speech = rms[mask]
    return (
        float(np.mean(speech)),
        float(np.std(speech)),
        float(np.max(speech) - np.min(speech)),
    )


def _worker(item: WorkItem) -> tuple[str, Result]:
    rel, abs_path, intervals, frac = item
    m, s, r = extract_loudness(Path(abs_path), intervals)
    if m is None:
        return rel, (None, None, None, None)
    return rel, (m, s, r, frac)


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


def _build_bounds_index(transcript_root: Path) -> dict[tuple[int, str, int], tuple[float, float]]:
    """(call, side, utt) → (trans start, trans end) — the slice boundaries."""
    idx: dict[tuple[int, str, int], tuple[float, float]] = {}
    for tpath in iter_transcript_paths(transcript_root):
        for u in parse_transcript(tpath):
            idx[(u.call_id, u.side, u.utt_num)] = (u.start, u.end)
    return idx


def _make_work_item(
    rel: str,
    abs_path: str,
    bounds: tuple[float, float] | None,
    words: list[tuple[float, float, str]],
) -> WorkItem | None:
    """Slice-relative word intervals + speech fraction, or None if unplaceable."""
    if bounds is None or not words:
        return None
    t0, t1 = bounds
    dur = t1 - t0
    if dur <= 0:
        return None
    intervals = []
    speech = 0.0
    for ws, we, _tok in words:
        s = max(ws, t0)
        e = min(we, t1)
        if e > s:
            intervals.append((s - t0, e - t0))
            speech += e - s
    if not intervals:
        return None
    return rel, abs_path, intervals, speech / dur


def write_loudness(
    manifest_csv: Path,
    output_csv: Path,
    out_root: Path,
    transcript_root: Path,
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

    fresh: dict[str, Result] = {}
    n_unplaceable = 0
    if needs_work:
        bounds_idx = _build_bounds_index(transcript_root)
        word_idx = build_word_index(transcript_root)
        work: list[WorkItem] = []
        for r in needs_work:
            try:
                key = parse_rel_path(r)
            except ValueError:
                key = None
            item = (
                _make_work_item(
                    r,
                    str(out_root / r),
                    bounds_idx.get(key) if key else None,
                    word_idx.get(key, []) if key else [],
                )
                if key
                else None
            )
            if item is None:
                fresh[r] = (None, None, None, None)
                n_unplaceable += 1
            else:
                work.append(item)
        if n_unplaceable:
            print(f"  unplaceable (no bounds/word rows): {n_unplaceable}")
        if workers <= 1:
            for item in work:
                rel, result = _worker(item)
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
                m, s, r, fr = fresh[rel]
                writer.writerow([rel, _fmt(m), _fmt(s), _fmt(r), _fmt(fr)])
    return len(rels)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_loudness(
        manifest_path(out_root),
        out_root / "features" / "loudness.csv",
        out_root=out_root,
        transcript_root=Path(args.transcript_root),
        workers=args.workers,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"wrote {n} loudness rows")
    return 0
