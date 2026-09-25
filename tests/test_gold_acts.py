"""gold_acts — the human-labelled pragmatics columns: one runnable check per mechanism."""
import csv
from pathlib import Path

import pytest

from swb_extract import swda
from swb_extract.features.inhouse import gold_acts as G
from swb_extract.manifest import manifest_path, open_appender, write_row

REPO = Path(__file__).resolve().parent.parent
TR = REPO / "swb_ms98_transcriptions_cleaned"


def test_swda_markup_is_stripped_to_ms98_like_tokens():
    assert swda.normalize_swda_text("{D So, } [ I, + I ] guess, /") == ["so", "i", "i", "guess"]
    assert swda.normalize_swda_text("<Laughter>. <<off-line>> Um-hum. --") == ["um-hum"]
    assert swda.normalize_ms98_token("an[y]-") == "an" and swda.normalize_ms98_token("because_1") == "because"
    assert swda.normalize_ms98_token("[t]-") is None


def test_alignment_survives_insertions_and_deletions():
    # SwDA side: two acts; ms98 side: an extra token, a missing token, three utterances
    swda_tokens = [(t, 0) for t in "i think it is great".split()] + [(t, 1) for t in "do you".split()]
    ms98_tokens = [(t, 2) for t in "i think it's great".split()] + [("uh", 4)] + [(t, 4) for t in "do you".split()]
    by_utt, ratio = swda.align_side(swda_tokens, ms98_tokens)
    assert by_utt[2] == [0] and by_utt[4] == [1]
    assert 0.6 < ratio < 1.0  # "it's" and "uh" unmatched


def test_flags_from_tags():
    assert G.flags_for("qy") == (1, 1, 0)
    assert G.flags_for("qy^d") == (1, 1, 0)
    assert G.flags_for("qh") == (1, 0, 0)          # rhetorical: a question, not information-seeking
    assert G.flags_for("sd^g") == (1, 0, 0)        # tag question
    assert G.flags_for("bh") == (0, 0, 1)          # "oh really?"
    assert G.flags_for("qy^m") == (1, 1, 1)        # mirrored question = echo
    assert G.flags_for("sd") == (0, 0, 0) and G.flags_for("b^m") == (0, 0, 0)


@pytest.mark.skipif(not (TR.is_dir() and (REPO / "corpus/swda/swda").is_dir() and (REPO / "corpus/nxt_switchboard_ann/xml/dialAct").is_dir()),
                    reason="needs transcripts + SwDA + NXT")
def test_text_join_reproduces_time_join_on_one_nxt_conversation():
    conv = 2005
    a = {}
    for side in "AB":
        a.update(G.nxt_terminal_acts(conv, side, TR))
    b, ratios, swapped = G.swda_conversation_acts(conv, TR)
    assert not swapped and min(ratios.values()) >= swda.MIN_MATCH_RATIO
    common = set(a) & set(b)
    q_agree = sum(G.flags_for(a[k])[0] == G.flags_for(b[k])[0] for k in common)
    assert len(common) > 50 and q_agree / len(common) >= 0.95
    # sw2006: SwDA's caller labels sit on the other ms98 channel — the swap must be detected
    _acts, ratios2, swapped2 = G.swda_conversation_acts(2006, TR)
    assert swapped2 and min(ratios2.values()) >= 0.9


@pytest.mark.skipif(not (TR.is_dir() and (REPO / "corpus/swda/swda").is_dir()), reason="needs transcripts + SwDA")
def test_cli_writes_rows_in_manifest_order_with_blanks_for_unlabelled(tmp_path):
    from swb_extract.cli import main

    from swb_extract import nxt
    from swb_extract.transcripts import parse_transcript

    # an NXT conversation, a SwDA-only conversation (with transcripts), and one with no gold
    swda_only = next(c for c in swda.list_conversations() if c not in set(nxt.list_conversations())
                     and G._trans_path(TR, c, "A").is_file())
    first_utt = lambda c: next(parse_transcript(G._trans_path(TR, c, "A"))).utt_num
    rels = [f"{c//10:03d}/sw{c:04d}A-U{first_utt(c):04d}.wav" for c in (2005, swda_only, 2001)]
    out = tmp_path / "utterances_v2"
    with open_appender(manifest_path(out)) as w:
        for rel in rels:
            write_row(w, rel, "")
    assert main(["features", "gold_acts", "--out-root", str(out), "--transcript-root", str(TR)]) == 0
    rows = list(csv.reader((out / "features" / "gold_acts.csv").open()))
    assert tuple(rows[0]) == G.HEADER and [r[0] for r in rows[1:]] == rels
    assert rows[1][2] == "nxt" and rows[2][2] == "swda" and rows[3] == [rels[2], "", "", "", "", ""]
    for r in rows[1:3]:
        assert r[1] and r[3] in ("0", "1")
