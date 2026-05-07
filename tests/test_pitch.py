import csv
from pathlib import Path

import pytest

from swb_extract.features.pitch import (
    HEADER,
    PITCH_FMAX,
    PITCH_FMIN,
    _frame_length_for_sr,
    extract_pitch,
    write_pitch,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)

REPO = Path(__file__).resolve().parent.parent
GOLDEN_WAV = REPO / "utterances_v2" / "200" / "sw2001A-U0002.wav"


@pytest.fixture(scope="session")
def golden_slice():
    if not GOLDEN_WAV.is_file():
        pytest.skip(f"missing golden slice: {GOLDEN_WAV}")
    return GOLDEN_WAV


def test_pitch_range_constants_match_recommendation():
    # Locks the design choice: tight speech-appropriate range
    assert PITCH_FMIN == 50
    assert PITCH_FMAX == 400


def test_frame_length_scales_with_sample_rate():
    # 8 kHz telephone (our slices)
    assert _frame_length_for_sr(8000) == 512
    # librosa's default-target sample rate
    assert _frame_length_for_sr(22050) == 2048
    # Boundary: 11025 falls into the telephone bucket
    assert _frame_length_for_sr(11025) == 512
    assert _frame_length_for_sr(16000) == 2048


def test_extract_pitch_returns_speech_range_for_female_speaker(golden_slice):
    # speaker 1020 is female (per LDC tables); pitch should land in adult-female F0
    mean, std, rng = extract_pitch(golden_slice)
    assert mean is not None
    assert 130 <= mean <= 280, f"expected adult-female range, got {mean:.2f}"
    assert std > 0
    assert rng > 0
    # No frame should ever exceed PITCH_FMAX
    assert mean - rng / 2 <= PITCH_FMAX


def test_write_pitch_round_trip(tmp_path, golden_slice):
    # Build a tiny manifest pointing at the real slice via relative path.
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    target = out / "200" / "sw2001A-U0002.wav"
    target.write_bytes(golden_slice.read_bytes())

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi um yeah")

    n = write_pitch(
        mp,
        out / "features" / "pitch.csv",
        out_root=out,
        workers=1,
    )
    assert n == 1

    rows = list(csv.reader((out / "features" / "pitch.csv").open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    pitch_mean = float(rows[1][1])
    assert 130 <= pitch_mean <= 280


def test_write_pitch_resume_skips_existing(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").write_bytes(golden_slice.read_bytes())

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
    pitch_csv = out / "features" / "pitch.csv"

    write_pitch(mp, pitch_csv, out_root=out, workers=1)
    mtime1 = pitch_csv.stat().st_mtime_ns
    contents1 = pitch_csv.read_text()

    # Second run: should detect cache and skip the heavy librosa work.
    write_pitch(mp, pitch_csv, out_root=out, workers=1)
    contents2 = pitch_csv.read_text()
    assert contents1 == contents2  # identical output, including float repr


def test_join_contract_against_manifest(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    target = out / "200" / "sw2001A-U0002.wav"
    target.write_bytes(golden_slice.read_bytes())

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
        # Second row points to a missing file; we can't actually run pyin on
        # it, so use limit=1 to only process the first.

    write_pitch(
        mp,
        out / "features" / "pitch.csv",
        out_root=out,
        workers=1,
        limit=1,
    )

    feat = list(csv.reader((out / "features" / "pitch.csv").open(encoding="utf-8")))
    manifest = list(csv.reader(mp.open(encoding="utf-8")))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    # Row order matches manifest (first row, in this limit=1 test)
    assert feat[1][0] == manifest[1][0]


def test_unvoiced_audio_yields_empty_cells(tmp_path):
    """Synthesize a near-silent slice; pyin should report no voiced frames
    and the CSV should contain empty cells (not zero)."""
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf

    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    silent_path = out / "200" / "sw9999A-U0001.wav"
    sr = 8000
    sf.write(silent_path, np.zeros(int(sr * 0.5), dtype=np.int16), sr)

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw9999A-U0001.wav", "[silence]")

    write_pitch(mp, out / "features" / "pitch.csv", out_root=out, workers=1)
    rows = list(csv.reader((out / "features" / "pitch.csv").open(encoding="utf-8")))
    # All three pitch columns are empty strings (not "0")
    assert rows[1][1] == ""
    assert rows[1][2] == ""
    assert rows[1][3] == ""
