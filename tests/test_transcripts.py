from swb_extract.transcripts import parse_transcript


def test_first_eight_utterances_match_sparse_numbering(golden_transcript):
    utterances = list(parse_transcript(golden_transcript))
    assert len(utterances) >= 8
    first8 = utterances[:8]
    # Sparse: cleaned transcripts skip pure-silence lines, so utt_num is NOT 1..8.
    assert [u.utt_num for u in first8] == [2, 4, 6, 8, 9, 11, 13, 15]


def test_first_utterance_has_exact_float_precision(golden_transcript):
    first = next(iter(parse_transcript(golden_transcript)))
    assert first.start == 0.977625
    assert first.end == 11.561375
    assert first.call_id == 2001
    assert first.side == "A"
    assert first.utt_num == 2


def test_inline_brackets_preserved(golden_transcript):
    by_num = {u.utt_num: u for u in parse_transcript(golden_transcript)}
    # Line 6 in the cleaned file: utt_num 11 contains "[noise]"
    assert "[noise]" in by_num[11].text


def test_id_field_must_match_filename(tmp_path):
    bad = tmp_path / "sw2001A-ms98-a-trans.text"
    bad.write_text("sw9999A-ms98-a-0002 0.0 1.0 mismatch\n")
    import pytest

    with pytest.raises(ValueError):
        list(parse_transcript(bad))
