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


def test_strip_bracket_tokens_keeps_inline_brackets():
    # i[t]- is a partial-word marker, not a whole-bracket token
    assert strip_bracket_tokens("i[t]- yeah") == "i[t]- yeah"


def test_laughed_words_are_unwrapped_not_dropped():
    # [laughter-yeah] = the speaker SAID "yeah" while laughing (2026-08-19 fix)
    assert tokenize("um [laughter-yeah] ok") == ["um", "yeah", "ok"]
    assert strip_bracket_tokens("i [laughter-yeah] you") == "i yeah you"
    # pure non-speech events still dropped
    assert tokenize("[laughter] [laughter-yeah]") == ["yeah"]


def test_laughed_partial_word_inner_form_kept_verbatim():
    # inner form may itself be a partial word; kept per the partial-word policy
    assert tokenize("[laughter-no[t]-] way") == ["no[t]-", "way"]


def test_angle_markup_tokens_are_dropped():
    # ms98 aside-span delimiters are markup, not words (2026-08-19 fix)
    assert tokenize("<b_aside> hang on <e_aside> sorry") == ["hang", "on", "sorry"]
    assert strip_bracket_tokens("<b_aside> hi <e_aside>") == "hi"


def test_laughed_words_feed_repetitions_and_fillers():
    # a laughed "yeah" pairs with a spoken "yeah"; a laughed "um" is a filler
    assert count_repetitions(tokenize("yeah [laughter-yeah]")) == 1
    assert count_filler_hits(tokenize("[laughter-um] right"), DEFAULT_FILLERS) == 1


def test_filler_partition_is_exact():
    # The 2026-08-20 split (audit §4E-c) must partition the legacy allowlist:
    # any drift breaks the split-columns-sum-to-combined identity.
    from swb_extract.features._text import DISCOURSE_MARKERS, FILLED_PAUSES
    assert FILLED_PAUSES | DISCOURSE_MARKERS == DEFAULT_FILLERS
    assert not (FILLED_PAUSES & DISCOURSE_MARKERS)
