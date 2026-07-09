"""Tests for the NXT gold-annotation layer (truncated-real-file fixtures).

Fixture files under tests/fixtures/nxt/ are verbatim-shaped excerpts of
corpus/nxt_switchboard_ann/xml/ (sw2005.A), small enough to reason about:
5 timed terminals + 1 punc + 1 sil; 5 dialog acts covering plain / decorated /
multi-act / missing-terminal tags; 1 disfluency group whose repair terminal is
absent from the terminals fixture (span-None path).
"""
from pathlib import Path

import pytest

from swb_extract.nxt import (
    NXT_XML,
    align_to_utterances,
    list_conversations,
    load_dialacts,
    load_disfluencies,
    load_terminal_times,
    parse_tag,
)

FIXTURES = Path(__file__).parent / "fixtures" / "nxt"


# --- parse_tag: never exact-match a swbdType --------------------------------


def test_parse_tag_plain_and_decorated():
    assert parse_tag("sd") == (("sd",), frozenset())
    assert parse_tag("qy^d^t") == (("qy",), frozenset({"^d", "^t"}))
    assert parse_tag("sd(^q)") == (("sd",), frozenset({"(^q)"}))
    assert parse_tag("b^m@") == (("b",), frozenset({"^m", "@"}))
    assert parse_tag("qy^g^t") == (("qy",), frozenset({"^g", "^t"}))
    assert parse_tag("sd(^q)*") == (("sd",), frozenset({"(^q)", "*"}))


def test_parse_tag_multi_act_and_bare():
    assert parse_tag("ba,fe") == (("ba", "fe"), frozenset())
    assert parse_tag("+") == (("+",), frozenset())
    assert parse_tag("+@") == (("+",), frozenset({"@"}))
    assert parse_tag('"') == (('"',), frozenset())
    assert parse_tag("%-") == (("%-",), frozenset())


def test_parse_tag_decoration_only_act_keeps_innermost_as_base():
    assert parse_tag("^2") == (("^2",), frozenset())
    assert parse_tag("^q") == (("^q",), frozenset())
    assert parse_tag("^2^t") == (("^2",), frozenset({"^t"}))
    assert parse_tag("^g") == (("^g",), frozenset())


# --- terminals ----------------------------------------------------------------


def test_load_terminal_times_skips_untimed_elements():
    times = load_terminal_times(2005, "A", xml_root=FIXTURES)
    assert times["s1_1"] == (0.8, 1.28)
    assert times["s2_3"] == (1.5, 1.985)
    assert "s1_2" not in times  # punc: no times
    assert "s2_2" not in times  # sil: no times
    assert len(times) == 5


# --- dialog acts ----------------------------------------------------------------


def test_load_dialacts_resolves_spans_through_terminals():
    acts = {a.da_id: a for a in load_dialacts(2005, "A", xml_root=FIXTURES)}
    assert len(acts) == 5
    assert acts["da1"].bases == ("o",)
    assert acts["da1"].start == 0.8 and acts["da1"].end == 1.28
    # da2 references s2_5, which the terminals fixture lacks — span from the rest
    assert acts["da2"].n_terminals == 3
    assert acts["da2"].start == 1.28 and acts["da2"].end == 1.985
    assert acts["da3"].bases == ("sd",) and acts["da3"].decorations == {"(^q)", "^t"}
    assert acts["da4"].bases == ("b",) and acts["da4"].decorations == {"^m", "@"}
    # da5's only terminal is missing entirely → no span
    assert acts["da5"].bases == ("ba", "fe")
    assert acts["da5"].start is None and acts["da5"].end is None


# --- disfluency -----------------------------------------------------------------


def test_load_disfluencies_spans_and_missing_repair():
    (d,) = load_disfluencies(2005, "A", xml_root=FIXTURES)
    assert d["reparandum_ids"] == ["s2_21", "s2_22"]
    assert d["reparandum_span"] == (10.0, 10.6)
    assert d["repair_ids"] == ["s2_24"]
    assert d["repair_span"] is None  # terminal absent from the fixture


# --- alignment ---------------------------------------------------------------------


def test_align_best_overlap_with_shorter_span_rule():
    utts = [(0.0, 1.5), (1.6, 2.5), (3.0, 4.0)]
    gold = [(0.0, 1.0), (1.4, 2.0), (2.6, 2.9), (4.5, 5.0), (3.9, 4.05)]
    #        → utt 0     → utt 1     → none      → none      → utt 2 (0.1 ≥ 0.5·0.15)
    assert align_to_utterances(gold, utts) == [0, 1, None, None, 2]


def test_align_is_input_order_independent():
    utts = [(3.0, 4.0), (0.0, 1.5), (1.6, 2.5)]  # shuffled
    assert align_to_utterances([(1.4, 2.0)], utts) == [2]


def test_align_threshold_rejects_thin_overlap():
    # overlap 0.2, shorter span 1.0 → 0.2 < 0.5 rejected
    assert align_to_utterances([(0.0, 1.0)], [(0.8, 2.0)]) == [None]
    # same spans, permissive threshold → accepted
    assert align_to_utterances([(0.0, 1.0)], [(0.8, 2.0)], min_frac=0.1) == [0]


# --- real-corpus smoke (skipped when the corpus is not on disk) ---------------------


@pytest.mark.skipif(not NXT_XML.is_dir(), reason="NXT corpus not on disk")
def test_real_corpus_smoke():
    convs = list_conversations()
    assert len(convs) == 642 and 2005 in convs
    acts = load_dialacts(2005, "A")
    assert acts[0].swbd_type == "o"
    assert acts[1].bases == ("qo",) and acts[1].start is not None
