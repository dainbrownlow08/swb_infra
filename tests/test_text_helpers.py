"""Pins the shared text helpers in features/_text.py.

Ported from switchboard's test_repetition_rate.py / test_filler_word_rate.py /
test_pronoun_rate.py, keeping the tests for the shared helper functions the
trusted per-second extractors import; the deprecated per-token rate columns'
own tests (compute_rate, CSV writers, CLI dispatch) were dropped with them.
"""
from swb_extract.features._text import (
    DEFAULT_FILLERS,
    count_filler_hits,
    count_repetitions,
    strip_bracket_tokens,
    tokenize,
)


def test_tokenize_strips_whole_bracket():
    assert tokenize("yes [laughter] um") == ["yes", "um"]
    assert tokenize("[noise] right") == ["right"]


def test_tokenize_keeps_inline_brackets():
    # i[t]- is a partial-word marker, not a whole-bracket token
    assert tokenize("i[t]- yeah") == ["i[t]-", "yeah"]


def test_tokenize_lowercases():
    assert tokenize("YEAH the THE") == ["yeah", "the", "the"]


def test_count_repetitions_basic():
    assert count_repetitions(["the", "the"]) == 1
    assert count_repetitions(["the", "dog", "the", "dog"]) == 2


def test_count_repetitions_binary_per_word():
    # Legacy semantics: count increments only on the SECOND occurrence,
    # so 'the the the' is 1 repetition (not 2).
    assert count_repetitions(["the", "the", "the"]) == 1
    assert count_repetitions(["the", "the", "the", "the", "the"]) == 1


def test_count_repetitions_no_repeats():
    assert count_repetitions(["the", "dog", "barks"]) == 0
    assert count_repetitions([]) == 0


def test_count_single_word_fillers():
    assert count_filler_hits(["um", "yeah", "um"], DEFAULT_FILLERS) == 2


def test_count_phrase_filler():
    assert count_filler_hits(["yeah", "you", "know", "that"], DEFAULT_FILLERS) == 1


def test_count_overlapping_phrases_greedy():
    # 'i mean' (2) + 'i guess' (2) + 'so' (1) = 3 hits
    assert count_filler_hits(
        ["i", "mean", "i", "guess", "so"], DEFAULT_FILLERS
    ) == 3


def test_count_does_not_double_count():
    assert count_filler_hits(
        ["you", "know", "you", "know"], DEFAULT_FILLERS
    ) == 2


def test_count_empty_list():
    assert count_filler_hits([], DEFAULT_FILLERS) == 0


def test_strip_bracket_tokens_removes_whole_brackets():
    assert strip_bracket_tokens("yes [laughter] um") == "yes um"
    assert strip_bracket_tokens("[noise] right") == "right"
    assert strip_bracket_tokens("i [laughter-yeah] you") == "i you"


def test_strip_bracket_tokens_keeps_inline_brackets():
    # i[t]- is a partial-word marker, not a whole-bracket token
    assert strip_bracket_tokens("i[t]- yeah") == "i[t]- yeah"
