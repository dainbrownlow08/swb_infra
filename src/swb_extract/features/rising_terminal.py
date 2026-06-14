"""Rising terminal — declarative-form questions via a final pitch rise.

Tannen ref: Ch.4 (PDF p.113) "You stayed at the Plaza?" — questions marked by
rising intonation rather than interrogative syntax. Catches what the syntactic
Question Flag misses; an engagement / High-Involvement marker.

Redesigned per AUDIT.md §3 fix 3. The original anchored the analysis window at
the END OF THE WAV FILE, but utterance slices follow trans-line bounds, which
carry trailing silence — so the "final 0.3 s" was often silence and 68.5% of
rows came back unjudgeable, missing not-at-random (short/quiet tails). The
window is now anchored at the END OF THE LAST WORD (sibling ``*-word.text``
timing, converted to file-relative time via the utterance's trans start),
falling back to the file end when word timing is unavailable. Only the tail
window is pitch-tracked, which also makes extraction much faster than running
pyin over the whole utterance.

For the TAIL_SEC window ending at the anchor, fit a least-squares slope to the
voiced F0 frames (librosa.pyin, same config as pitch.py):

  Rising Terminal Flag =
    1   if slope >= RISE_MIN_HZ_PER_SEC over >= MIN_VOICED voiced tail frames
    0   otherwise
    ""  too few voiced frames in the tail to judge

  Terminal F0 Slope =
    the fitted slope in Hz/s whenever the flag is defined, else "". Reported
    so analyses can use the continuous contour (and model missingness)
    instead of only the thresholded flag.

Empty cells mean "no measurement" — downstream code must NOT fill them with 0
(that conflates "unmeasurable" with "not rising").

Output: utterances_v2/features/rising_terminal.csv
Header: Utterance File Name,Rising Terminal Flag,Terminal F0 Slope
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from .pitch import PITCH_FMAX, PITCH_FMIN, _frame_length_for_sr
from .turn_gap import build_turn_gap_index
from .word_align import build_word_index

FEATURE_NAME = "rising_terminal"
HEADER = ("Utterance File Name", "Rising Terminal Flag", "Terminal F0 Slope")

TAIL_SEC = 0.3              # window before the speech anchor to fit
RISE_MIN_HZ_PER_SEC = 30.0  # slope threshold for "rising"
MIN_VOICED = 3              # minimum voiced frames in the tail to judge

# (flag, slope_hz_per_sec); (None, None) = unjudgeable
Result = tuple[int | None, float | None]


def extract_rising_terminal(
    wav_path: Path, tail_anchor_sec: float | None = None
) -> Result:
    """Fit the terminal F0 slope in the TAIL_SEC window ending at the anchor.

    ``tail_anchor_sec`` is the file-relative end of speech (last word offset);
    None anchors at the end of the file.
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav_path), sr=None)
    if y.size == 0:
        return None, None
    duration = y.size / sr
    end = duration
    if tail_anchor_sec is not None and 0.0 < tail_anchor_sec < duration:
        end = tail_anchor_sec
    start = max(0.0, end - TAIL_SEC)
    y_win = y[int(start * sr): int(end * sr)]

    frame_length = _frame_length_for_sr(sr)
    if y_win.size < frame_length:
        return None, None
    f0, _vf, _vp = librosa.pyin(
        y_win, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr, frame_length=frame_length
    )
    hop_length = frame_length // 4  # librosa.pyin default
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    voiced = ~np.isnan(f0)
    fv = f0[voiced]
    tv = times[voiced]
    if fv.size < MIN_VOICED:
        return None, None
    slope = float(np.polyfit(tv, fv, 1)[0])
    return (1 if slope >= RISE_MIN_HZ_PER_SEC else 0), slope


def build_anchor_index(transcript_root: Path) -> dict[tuple[int, str, int], float]:
    """File-relative end-of-speech per utterance: last word end − trans start."""
    words = build_word_index(transcript_root)
    anchors: dict[tuple[int, str, int], float] = {}
    starts: dict[tuple[int, str, int], float] = {}
    for call_id, merged in build_turn_gap_index(transcript_root).items():
        for side, utt_num, start, _end in merged:
            starts[(call_id, side, utt_num)] = start
    for key, rows in words.items():
        start = starts.get(key)
        if start is None:
            continue
        last_word_end = max(end for _s, end, _t in rows)
        anchor = last_word_end - start
        if anchor > 0.0:
            anchors[key] = anchor
    return anchors


def _worker(arg: tuple[str, str, float | None]) -> tuple[str, Result]:
    rel, abs_path, anchor = arg
    return rel, extract_rising_terminal(Path(abs_path), anchor)


def _fmt_row(rel: str, result: Result) -> list[str]:
    flag, slope = result
    return [
        rel,
        "" if flag is None else str(flag),
        "" if slope is None else repr(slope),
    ]


def _read_existing(output_csv: Path) -> dict[str, list[str]]:
    if not output_csv.exists():
        return {}
    with open(output_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != HEADER:
            return {}
        return {row[0]: row for row in reader if row}


def write_rising_terminal(
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
        f"rising_terminal: {len(rels):,} total, {len(cache):,} cached, "
        f"{len(needs_work):,} to extract (workers={workers})"
    )

    fresh: dict[str, Result] = {}
    if needs_work:
        anchors = build_anchor_index(transcript_root)
        n_anchored = 0
        work = []
        for r in needs_work:
            anchor = anchors.get(parse_rel_path(r))
            n_anchored += anchor is not None
            work.append((r, str(out_root / r), anchor))
        print(f"  word-anchored tails: {n_anchored:,}/{len(work):,}")
        if workers <= 1:
            for arg in work:
                rel, result = _worker(arg)
                fresh[rel] = result
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                done = 0
                last_log = 0
                for rel, result in ex.map(_worker, work, chunksize=16):
                    fresh[rel] = result
                    done += 1
                    if done - last_log >= 5000 or done == len(work):
                        print(f"  {done:,}/{len(work):,}", flush=True)
                        last_log = done

    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for rel in rels:
            if rel in cache:
                writer.writerow(cache[rel])
            else:
                writer.writerow(_fmt_row(rel, fresh[rel]))
    return len(rels)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_rising_terminal(
        manifest_path(out_root),
        out_root / "features" / "rising_terminal.csv",
        out_root=out_root,
        transcript_root=Path(args.transcript_root),
        workers=args.workers,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"wrote {n} rising-terminal rows")
    return 0
