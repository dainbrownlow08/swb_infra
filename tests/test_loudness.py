import csv
from pathlib import Path

import pytest

from swb_extract.cli import main
from swb_extract.features.loudness import (
    HEADER,
    extract_loudness,
    write_loudness,
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


def test_extract_loudness_returns_plausible_values_for_real_speech(golden_slice):
    mean, std, rng = extract_loudness(golden_slice)
    assert mean is not None
    # Sanity bounds for normalized audio: well above silence, well below clipping.
    assert 0.0 < mean < 1.0
    assert std > 0.0
    assert rng > 0.0


def test_write_loudness_round_trip(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    target = out / "200" / "sw2001A-U0002.wav"
    target.write_bytes(golden_slice.read_bytes())

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi um yeah")

    n = write_loudness(
        mp, out / "features" / "loudness.csv", out_root=out, workers=1
    )
    assert n == 1

    rows = list(csv.reader((out / "features" / "loudness.csv").open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    # All three values parse as positive floats
    assert float(rows[1][1]) > 0
    assert float(rows[1][2]) > 0
    assert float(rows[1][3]) > 0


def test_write_loudness_resume_skips_existing(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").write_bytes(golden_slice.read_bytes())

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
    out_csv = out / "features" / "loudness.csv"

    write_loudness(mp, out_csv, out_root=out, workers=1)
    contents1 = out_csv.read_text()
    write_loudness(mp, out_csv, out_root=out, workers=1)
    contents2 = out_csv.read_text()
    assert contents1 == contents2  # byte-identical on resume


def test_join_contract_against_manifest(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").write_bytes(golden_slice.read_bytes())

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")

    write_loudness(
        mp, out / "features" / "loudness.csv",
        out_root=out, workers=1, limit=1,
    )

    feat = list(csv.reader((out / "features" / "loudness.csv").open(encoding="utf-8")))
    manifest = list(csv.reader(mp.open(encoding="utf-8")))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    assert feat[1][0] == manifest[1][0]


def test_silent_audio_yields_zero_not_empty(tmp_path):
    """Silent audio: RMS=0 is a valid measurement, not a failure."""
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

    write_loudness(
        mp, out / "features" / "loudness.csv", out_root=out, workers=1
    )
    rows = list(csv.reader((out / "features" / "loudness.csv").open(encoding="utf-8")))
    # Distinct from None/empty: zero floats parse, empty strings don't
    assert rows[1][1] != ""
    assert float(rows[1][1]) == 0.0
    assert float(rows[1][2]) == 0.0
    assert float(rows[1][3]) == 0.0


def test_features_dispatch_routes_loudness(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").write_bytes(golden_slice.read_bytes())

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
    rc = main(["features", "loudness", "--out-root", str(out)])
    assert rc == 0
    assert (out / "features" / "loudness.csv").exists()
