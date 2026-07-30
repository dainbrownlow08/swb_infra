"""Tests for the prosodic rising-terminal extractor.

Synthesises pitch glides so we exercise the real librosa.pyin + slope-fit path.
pyin can occasionally return too few voiced frames on synthetic tones; those
cases are skipped rather than asserted, so the test validates DIRECTION
(rising → 1, flat/falling → 0) without being flaky.

The 2026-07-29 semitone redesign thresholds the semitone-scale slope
(register-invariant) instead of the Hz/s slope; the register-invariance test
below is the runnable check that the same proportional gesture classifies
identically at low and high registers.
"""
import csv

import numpy as np
import pytest

pytest.importorskip("librosa")
from scipy.io import wavfile  # noqa: E402

from swb_extract.features.rising_terminal import (  # noqa: E402
    RISE_MIN_ST_PER_SEC,
    extract_rising_terminal,
)

SR = 8000


def _glide_samples(f_start, f_end, dur):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    # instantaneous-frequency linear glide → integrated phase
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * dur))
    return 0.6 * np.sin(phase)


def _write(path, y):
    wavfile.write(str(path), SR, (y * 32767).astype(np.int16))


def _glide(path, f_start, f_end, dur=0.6, trailing_silence=0.0):
    y = _glide_samples(f_start, f_end, dur)
    if trailing_silence:
        y = np.concatenate([y, np.zeros(int(SR * trailing_silence))])
    _write(path, y)


def test_rising_glide_flagged(tmp_path):
    p = tmp_path / "rise.wav"
    _glide(p, 130, 250)
    flag, _hz, st = extract_rising_terminal(p)
    if flag is None:
        pytest.skip("pyin found too few voiced frames in tail")
    assert flag == 1
    assert st >= RISE_MIN_ST_PER_SEC


def test_flat_tone_not_flagged(tmp_path):
    p = tmp_path / "flat.wav"
    _glide(p, 180, 180)
    flag, _hz, st = extract_rising_terminal(p)
    if flag is None:
        pytest.skip("pyin found too few voiced frames in tail")
    assert flag == 0
    assert abs(st) < RISE_MIN_ST_PER_SEC


def test_falling_glide_not_flagged(tmp_path):
    p = tmp_path / "fall.wav"
    _glide(p, 250, 130)
    flag, _hz, _st = extract_rising_terminal(p)
    if flag is None:
        pytest.skip("pyin found too few voiced frames in tail")
    assert flag == 0


def test_register_invariance_same_proportional_rise(tmp_path):
    """The semitone redesign's contract: identical ~3 st gestures at a low and
    a high register classify identically, even though their Hz/s slopes
    differ ~2x (the asymmetry the old 30 Hz/s rule imposed by register)."""
    lo, hi = tmp_path / "lo.wav", tmp_path / "hi.wav"
    _glide(lo, 120, 143)   # 12*log2(143/120) ≈ 3.02 st over 0.6 s
    _glide(hi, 240, 286)   # same semitone gesture, doubled register
    flag_lo, hz_lo, st_lo = extract_rising_terminal(lo)
    flag_hi, hz_hi, st_hi = extract_rising_terminal(hi)
    if flag_lo is None or flag_hi is None:
        pytest.skip("pyin found too few voiced frames in tail")
    assert flag_lo == flag_hi == 1
    assert st_lo >= RISE_MIN_ST_PER_SEC and st_hi >= RISE_MIN_ST_PER_SEC
    # same gesture on the semitone scale ...
    assert abs(st_lo - st_hi) < 0.35 * max(st_lo, st_hi)
    # ... but ~2x apart on the Hz scale the old rule thresholded
    assert hz_hi > 1.5 * hz_lo


def test_trailing_silence_defeats_unanchored_tail(tmp_path):
    """The MNAR regression case: speech ends, then silence pads the slice.

    Unanchored, the tail window is pure silence → unjudgeable. Anchored at the
    end of speech, the same file is judged and the rise is found.
    """
    p = tmp_path / "rise_padded.wav"
    _glide(p, 130, 250, dur=0.6, trailing_silence=0.5)

    flag_unanchored, _, _ = extract_rising_terminal(p)
    assert flag_unanchored is None  # tail = silence → no voiced frames

    flag_anchored, _hz, st = extract_rising_terminal(p, tail_anchor_sec=0.6)
    if flag_anchored is None:
        pytest.skip("pyin found too few voiced frames in anchored tail")
    assert flag_anchored == 1
    assert st >= RISE_MIN_ST_PER_SEC


def test_anchor_beyond_duration_falls_back_to_file_end(tmp_path):
    p = tmp_path / "rise2.wav"
    _glide(p, 130, 250)
    flag_anchored, _, _ = extract_rising_terminal(p, tail_anchor_sec=99.0)
    flag_plain, _, _ = extract_rising_terminal(p)
    assert flag_anchored == flag_plain


def test_empty_audio_none(tmp_path):
    p = tmp_path / "empty.wav"
    wavfile.write(str(p), SR, np.zeros(0, dtype=np.int16))
    assert extract_rising_terminal(p) == (None, None, None)


def test_too_short_window_none(tmp_path):
    p = tmp_path / "tiny.wav"
    _write(p, _glide_samples(180, 180, 0.03))  # shorter than one pyin frame
    assert extract_rising_terminal(p) == (None, None, None)


def test_cache_recomputes_empty_flag_rows(tmp_path, monkeypatch):
    """A cached unjudgeable ('' flag) row is recomputed; a settled row is reused.

    Freezing the original not-at-random missingness is the §3 fix-3 failure mode.
    """
    from swb_extract.features import rising_terminal as rt
    from swb_extract.manifest import manifest_path, open_appender, write_row

    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    rel_empty = "200/sw2001A-U0002.wav"
    rel_done = "200/sw2001B-U0004.wav"
    with open_appender(mp) as w:
        write_row(w, rel_empty, "")
        write_row(w, rel_done, "")

    out_csv = out / "features" / "rising_terminal.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(rt.HEADER)
        wr.writerow([rel_empty, "", "", ""])          # prior run: unjudgeable
        wr.writerow([rel_done, "0", "12.0", "1.1"])   # prior run: settled

    calls = []

    def fake_extract(path, anchor=None):
        calls.append(str(path))
        return (1, 42.0, 3.5)

    monkeypatch.setattr(rt, "extract_rising_terminal", fake_extract)
    troot = tmp_path / "swb_ms98_transcriptions_cleaned"
    troot.mkdir()
    rt.write_rising_terminal(
        mp, out_csv, out_root=out, transcript_root=troot, workers=1
    )

    by_rel = {r[0]: r[1:] for r in csv.reader(out_csv.open())}
    assert by_rel[rel_empty] == ["1", repr(42.0), repr(3.5)]  # recomputed
    assert by_rel[rel_done] == ["0", "12.0", "1.1"]           # reused from cache
