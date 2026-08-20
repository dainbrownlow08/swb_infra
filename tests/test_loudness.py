import csv
from pathlib import Path

import pytest

from swb_extract.cli import main
from swb_extract.features.inhouse.loudness import (
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
REAL_TRANS_ROOT = REPO / "swb_ms98_transcriptions_cleaned"

# Real alignment for the golden slice (trans span 0.977625–11.561375 s).
GOLDEN_T0 = 0.977625
GOLDEN_WORDS_ABS = [(1.215250, 1.724625), (2.273625, 2.927625), (3.221500, 3.661750)]
GOLDEN_INTERVALS = [(s - GOLDEN_T0, e - GOLDEN_T0) for s, e in GOLDEN_WORDS_ABS]


@pytest.fixture(scope="session")
def golden_slice():
    if not GOLDEN_WAV.is_file():
        pytest.skip(f"missing golden slice: {GOLDEN_WAV}")
    return GOLDEN_WAV


def _write_alignment(
    root: Path,
    call: int,
    side: str,
    utt: int,
    t0: float,
    t1: float,
    words: list[tuple[float, float, str]],
) -> None:
    """Minimal trans.text + word.text pair for one utterance (absolute times)."""
    d = root / f"{call // 100}" / f"{call}"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"sw{call}{side}-ms98-a-{utt:04d}"
    text = " ".join(w for _s, _e, w in words) or "uh"
    (d / f"sw{call}{side}-ms98-a-trans.text").write_text(
        f"{tag} {t0} {t1} {text}\n", encoding="utf-8"
    )
    (d / f"sw{call}{side}-ms98-a-word.text").write_text(
        "".join(f"{tag} {s} {e} {w}\n" for s, e, w in words), encoding="utf-8"
    )


def _tone_and_silence_wav(path: Path):
    """2 s wav: 1 s silence, then 1 s full-scale tone. Returns (sr, seconds)."""
    import numpy as np
    import soundfile as sf

    sr = 8000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = np.concatenate([np.zeros(sr, dtype=np.float32), tone])
    sf.write(path, y, sr)
    return sr, 2.0


def test_word_tight_ignores_silence(tmp_path):
    """The redesign's contract: silence outside word intervals must not dilute."""
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    wav = tmp_path / "tone.wav"
    _tone_and_silence_wav(wav)

    tight_mean, tight_std, tight_rng = extract_loudness(wav, [(1.0, 2.0)])
    diluted_mean, _, _ = extract_loudness(wav, [(0.0, 2.0)])
    assert tight_mean is not None and diluted_mean is not None
    # Whole-slice averaging halves the mean (1 s silence + 1 s tone);
    # the word-tight value must sit well above it.
    assert tight_mean > 1.5 * diluted_mean
    # Within the steady tone, spread and range are small relative to the mean.
    assert tight_std < 0.5 * tight_mean
    assert tight_rng < tight_mean


def test_no_frame_in_interval_returns_none(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    wav = tmp_path / "tone.wav"
    _tone_and_silence_wav(wav)
    # 1 ms interval: shorter than a frame hop, no frame center can land inside.
    m, s, r = extract_loudness(wav, [(1.5, 1.501)])
    assert (m, s, r) == (None, None, None)


def test_extract_loudness_returns_plausible_values_for_real_speech(golden_slice):
    mean, std, rng = extract_loudness(golden_slice, GOLDEN_INTERVALS)
    assert mean is not None
    # Sanity bounds for normalized audio: well above silence, well below clipping.
    assert 0.0 < mean < 1.0
    assert std > 0.0
    assert rng > 0.0
    # Word-tight mean must exceed the whole-slice mean (silence removed).
    dur = 11.561375 - GOLDEN_T0
    whole_mean, _, _ = extract_loudness(golden_slice, [(0.0, dur)])
    assert mean > whole_mean


def test_write_loudness_round_trip(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").write_bytes(golden_slice.read_bytes())
    trans = tmp_path / "trans"
    _write_alignment(
        trans, 2001, "A", 2, GOLDEN_T0, 11.561375,
        [(s, e, "w") for s, e in GOLDEN_WORDS_ABS],
    )

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi um yeah")

    n = write_loudness(
        mp, out / "features" / "loudness.csv",
        out_root=out, transcript_root=trans, workers=1,
    )
    assert n == 1

    rows = list(csv.reader((out / "features" / "loudness.csv").open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    # All three stats positive; speech frac strictly inside (0, 1).
    assert float(rows[1][1]) > 0
    assert float(rows[1][2]) > 0
    assert float(rows[1][3]) > 0
    assert 0.0 < float(rows[1][4]) < 1.0


def test_write_loudness_resume_skips_existing(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").write_bytes(golden_slice.read_bytes())
    trans = tmp_path / "trans"
    _write_alignment(
        trans, 2001, "A", 2, GOLDEN_T0, 11.561375,
        [(s, e, "w") for s, e in GOLDEN_WORDS_ABS],
    )

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
    out_csv = out / "features" / "loudness.csv"

    write_loudness(mp, out_csv, out_root=out, transcript_root=trans, workers=1)
    contents1 = out_csv.read_text()
    write_loudness(mp, out_csv, out_root=out, transcript_root=trans, workers=1)
    contents2 = out_csv.read_text()
    assert contents1 == contents2  # byte-identical on resume


def test_stale_pre_redesign_cache_is_not_reused(tmp_path, golden_slice):
    """The old 4-column header must not satisfy the resume cache (vintage guard)."""
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").write_bytes(golden_slice.read_bytes())
    trans = tmp_path / "trans"
    _write_alignment(
        trans, 2001, "A", 2, GOLDEN_T0, 11.561375,
        [(s, e, "w") for s, e in GOLDEN_WORDS_ABS],
    )
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")

    out_csv = out / "features" / "loudness.csv"
    out_csv.parent.mkdir(parents=True)
    out_csv.write_text(
        "Utterance File Name,loudness mean,loudness std,loudness range\n"
        "200/sw2001A-U0002.wav,9.0,9.0,9.0\n",
        encoding="utf-8",
    )
    write_loudness(mp, out_csv, out_root=out, transcript_root=trans, workers=1)
    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert float(rows[1][1]) < 1.0  # recomputed, not the planted stale 9.0


def test_missing_word_rows_yield_blank_cells(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").write_bytes(golden_slice.read_bytes())
    trans = tmp_path / "trans"
    trans.mkdir()  # no alignment files at all

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")

    write_loudness(
        mp, out / "features" / "loudness.csv",
        out_root=out, transcript_root=trans, workers=1,
    )
    rows = list(csv.reader((out / "features" / "loudness.csv").open(encoding="utf-8")))
    assert rows[1][1:] == ["", "", "", ""]


def test_silent_audio_yields_zero_not_empty(tmp_path):
    """Silent-but-aligned audio: RMS=0 is a valid measurement, not a failure."""
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf

    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    sr = 8000
    sf.write(out / "200" / "sw9999A-U0001.wav", np.zeros(sr, dtype=np.int16), sr)
    trans = tmp_path / "trans"
    _write_alignment(trans, 9999, "A", 1, 10.0, 11.0, [(10.1, 10.9, "uh")])

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw9999A-U0001.wav", "uh")

    write_loudness(
        mp, out / "features" / "loudness.csv",
        out_root=out, transcript_root=trans, workers=1,
    )
    rows = list(csv.reader((out / "features" / "loudness.csv").open(encoding="utf-8")))
    # Distinct from None/empty: zero floats parse, empty strings don't
    assert rows[1][1] != ""
    assert float(rows[1][1]) == 0.0
    assert float(rows[1][2]) == 0.0
    assert float(rows[1][3]) == 0.0
    assert float(rows[1][4]) == pytest.approx(0.8)


def test_features_dispatch_routes_loudness(tmp_path, golden_slice):
    out = tmp_path / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").write_bytes(golden_slice.read_bytes())
    trans = tmp_path / "trans"
    _write_alignment(
        trans, 2001, "A", 2, GOLDEN_T0, 11.561375,
        [(s, e, "w") for s, e in GOLDEN_WORDS_ABS],
    )

    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
    rc = main([
        "features", "loudness",
        "--out-root", str(out),
        "--transcript-root", str(trans),
    ])
    assert rc == 0
    assert (out / "features" / "loudness.csv").exists()
