from swb_extract.slicer import _probe_duration, slice_utterance


def test_slice_utt2_of_sw2001A(golden_audio, tmp_path):
    dst = tmp_path / "sw2001A-U0002.wav"
    slice_utterance(golden_audio, dst, 0.977625, 11.561375)
    assert dst.exists()
    duration = _probe_duration(dst)
    expected = 11.561375 - 0.977625
    assert abs(duration - expected) < 0.005


def test_slice_skips_when_dst_exists(golden_audio, tmp_path):
    dst = tmp_path / "out.wav"
    slice_utterance(golden_audio, dst, 0.977625, 1.5)
    mtime1 = dst.stat().st_mtime_ns
    slice_utterance(golden_audio, dst, 0.977625, 1.5)
    assert dst.stat().st_mtime_ns == mtime1


def test_slice_overwrite_redoes_work(golden_audio, tmp_path):
    dst = tmp_path / "out.wav"
    slice_utterance(golden_audio, dst, 0.977625, 1.5)
    mtime1 = dst.stat().st_mtime_ns
    slice_utterance(golden_audio, dst, 0.977625, 1.5, overwrite=True)
    assert dst.stat().st_mtime_ns >= mtime1


def test_slice_rejects_zero_duration(golden_audio, tmp_path):
    import pytest

    with pytest.raises(ValueError):
        slice_utterance(golden_audio, tmp_path / "x.wav", 1.0, 1.0)
