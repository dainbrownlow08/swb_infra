"""One acoustic pass per utterance, shared by pv, lv, boplen and wpp.

The paper's phonology variables all hang off one signal — OpenSMILE's F0 track, whose
presence defines "a speech signal" — so the four modules that need audio share one
pass and one cache instead of pitch-tracking the corpus four times (pyin runs at
~0.05x real time: a full corpus pass is hours, not minutes).

Per utterance slice (trans-span WAV), on the in-house pitch settings (pitch.py:
librosa.pyin, 50–400 Hz, native 8 kHz, 512-sample ≈ 64 ms window, hop = 128 = 16 ms):

  voiced frames  — pyin reports an F0 (the paper's "speech signal")
  f0 mean / var  — over voiced frames, Hz / Hz²                           (→ pv)
  rms mean / var — librosa frame RMS on the SAME frame grid, voiced frames (→ lv)
  pauses         — maximal runs of unvoiced frames bounded on both sides by voiced
                   frames of the same slice = the paper's between-own pauses; count and
                   total seconds (run length × hop / sr)                   (→ boplen, wpp)

Leading and trailing unvoiced stretches are slice padding / turn silence, not pauses
between the speaker's own words, so they are never counted. Taken literally, as the
paper did, every unvoiced run counts, voiceless-consonant stretches included — hence
pause lengths "on the order of 0.1 s". Variances are population variances (ddof=0), so
per-utterance (n, mean, var) pool exactly to the side-level variance (``_io``).

Cache: ``utterances_v2/derived/thomas2018_prosody.csv`` (a derived artifact, outside
``features/`` so the canonical-table builder never ingests it). Rows are keyed on the
utterance path and resumed on the header, exactly as pitch.py does; pass --overwrite
to recompute. Unreadable audio → all cells blank; readable audio with no voiced frames
→ voiced n = 0, pause n = 0, pause sec = 0.0.
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from ..inhouse.pitch import PITCH_FMAX, PITCH_FMIN, _frame_length_for_sr
from ._io import fmt, iter_manifest, num

CACHE_RELPATH = Path("derived") / "thomas2018_prosody.csv"
HEADER = (
    "Utterance File Name",
    "voiced n",
    "f0 mean",
    "f0 var",
    "rms mean",
    "rms var",
    "pause n",
    "pause sec",
)

# (voiced_n, f0_mean, f0_var, rms_mean, rms_var, pause_n, pause_sec)
Prosody = tuple[int, float | None, float | None, float | None, float | None, int, float]
NO_VOICING: Prosody = (0, None, None, None, None, 0, 0.0)


def extract_prosody(wav_path: Path) -> Prosody | None:
    """One utterance's voiced-frame F0 / RMS moments and between-own pauses.

    None = unreadable or too short to frame (unmeasurable). Pure function, picklable.
    """
    import librosa
    import numpy as np

    try:
        y, sr = librosa.load(str(wav_path), sr=None)
    except Exception:
        return None
    frame_length = _frame_length_for_sr(sr)
    if y.size < frame_length:
        return None
    hop = frame_length // 4  # librosa.pyin's default hop
    f0, _flag, _prob = librosa.pyin(
        y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr, frame_length=frame_length
    )
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    n = min(f0.size, rms.size)
    voiced = ~np.isnan(f0[:n])
    if not voiced.any():
        return NO_VOICING
    fv = f0[:n][voiced]
    rv = rms[:n][voiced]
    gaps = np.diff(np.flatnonzero(voiced)) - 1  # unvoiced frames between voiced ones
    gaps = gaps[gaps > 0]
    return (
        int(voiced.sum()),
        float(fv.mean()),
        float(fv.var()),
        float(rv.mean()),
        float(rv.var()),
        int(gaps.size),
        float(gaps.sum() * hop / sr),
    )


def _worker(arg: tuple[str, str]) -> tuple[str, Prosody | None]:
    rel, abs_path = arg
    return rel, extract_prosody(Path(abs_path))


def _fmt_row(rel: str, p: Prosody | None) -> list[str]:
    if p is None:
        return [rel, "", "", "", "", "", "", ""]
    nv, f0m, f0v, rm, rv, pn, ps = p
    return [rel, str(nv), fmt(f0m), fmt(f0v), fmt(rm), fmt(rv), str(pn), fmt(ps)]


def _parse_row(row: list[str]) -> Prosody | None:
    if len(row) < len(HEADER) or row[1] == "":
        return None
    return (
        int(row[1]), num(row[2]), num(row[3]), num(row[4]), num(row[5]),
        int(row[6]), float(row[7]),
    )


def _read_cache(cache_csv: Path) -> dict[str, Prosody | None]:
    if not cache_csv.exists():
        return {}
    with open(cache_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if tuple(next(reader, None) or ()) != HEADER:
            return {}
        return {row[0]: _parse_row(row) for row in reader if row}


def ensure_prosody(
    manifest_csv: Path,
    out_root: Path,
    workers: int = 4,
    limit: int = 0,
    overwrite: bool = False,
) -> dict[str, Prosody | None]:
    """Prosody for the (first ``limit``) manifest rows, computing + caching what is missing.

    Returns a dict in manifest order so projections can write straight from it.
    """
    cache_csv = out_root / CACHE_RELPATH
    all_rels = [rel for rel, _ in iter_manifest(manifest_csv)]
    rels = all_rels[:limit] if limit else all_rels
    cache = {} if overwrite else _read_cache(cache_csv)
    todo = [r for r in rels if r not in cache]
    print(
        f"thomas2018 prosody: {len(rels):,} requested, {len(rels) - len(todo):,} cached, "
        f"{len(todo):,} to extract (workers={workers})"
    )
    if todo:
        work = [(r, str(out_root / r)) for r in todo]
        if workers <= 1:
            for arg in work:
                rel, result = _worker(arg)
                cache[rel] = result
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                done = 0
                last_log = 0
                for rel, result in ex.map(_worker, work, chunksize=8):
                    cache[rel] = result
                    done += 1
                    if done - last_log >= 1000 or done == len(work):
                        print(f"  {done:,}/{len(work):,}", flush=True)
                        last_log = done
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_csv, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(HEADER)
            for rel in all_rels:  # every row we have, manifest order
                if rel in cache:
                    writer.writerow(_fmt_row(rel, cache[rel]))
    return {r: cache[r] for r in rels}


def write_projection(
    manifest_csv: Path,
    output_csv: Path,
    out_root: Path,
    header: tuple[str, ...],
    project: Callable[[Prosody | None], list[str]],
    workers: int = 4,
    limit: int = 0,
    overwrite: bool = False,
) -> int:
    """Write one module's columns from the shared prosody pass, manifest order."""
    prosody = ensure_prosody(manifest_csv, out_root, workers, limit, overwrite)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for rel, p in prosody.items():
            writer.writerow([rel, *project(p)])
    return len(prosody)
