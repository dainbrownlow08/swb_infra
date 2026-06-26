import csv

import pytest

from swb_extract.features.personal_focus_score import (
    ALL_TANNEN_CATEGORIES,
    HEADER,
    IMPERSONAL_CATEGORIES,
    MIN_HITS_FOR_SCORE,
    PERSONAL_CATEGORIES,
    _get_lexicon,
    compute_counts,
    run,
    strip_bracket_tokens,
    write_personal_focus_scores,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)


@pytest.fixture(scope="module", autouse=True)
def _check_empath():
    try:
        import empath  # noqa: F401
    except ImportError:
        pytest.skip("empath not installed")


PERSONAL_TEXT = (
    "mom always made sure dinner was ready and the family ate together "
    "at home before dad got back"
)
IMPERSONAL_TEXT = (
    "the government passed the new tax law and the economy reacted to the "
    "policy while business leaders lobbied congress"
)


def test_categories_partition():
    assert PERSONAL_CATEGORIES & IMPERSONAL_CATEGORIES == frozenset()
    assert ALL_TANNEN_CATEGORIES == PERSONAL_CATEGORIES | IMPERSONAL_CATEGORIES


def test_all_categories_exist_in_empath():
    lex = _get_lexicon()
    missing = ALL_TANNEN_CATEGORIES - set(lex.cats.keys())
    assert not missing, f"Empath missing categories: {sorted(missing)}"


def test_strip_bracket_tokens():
    assert strip_bracket_tokens("[laughter] hello there") == "hello there"
    assert strip_bracket_tokens("[noise] [vocalized-noise]") == ""
    assert strip_bracket_tokens("hello [laughter] there") == "hello there"
    assert strip_bracket_tokens("i[t]- said") == "i[t]- said"
    assert strip_bracket_tokens("") == ""


def test_personal_topic_counts_and_score():
    p, i, tokens, score = compute_counts(PERSONAL_TEXT)
    assert p > i
    assert tokens == len(PERSONAL_TEXT.split())
    assert score is not None and score > 0.7


def test_impersonal_topic_counts_and_score():
    p, i, tokens, score = compute_counts(IMPERSONAL_TEXT)
    assert i > p
    assert score is not None and score < 0.3


def test_score_gated_by_min_hits():
    """The score must be None exactly when total hits < MIN_HITS_FOR_SCORE."""
    for text in (
        "uh-huh",
        "yeah",
        "my mom",
        "the law",
        PERSONAL_TEXT,
        IMPERSONAL_TEXT,
    ):
        p, i, _tokens, score = compute_counts(text)
        if p + i >= MIN_HITS_FOR_SCORE:
            assert score == pytest.approx(p / (p + i))
        else:
            assert score is None


def test_counts_always_defined():
    assert compute_counts("uh-huh")[:3] == (0, 0, 1)
    assert compute_counts("")[:3] == (0, 0, 0)
    assert compute_counts("[laughter] [noise]")[:3] == (0, 0, 0)


def test_brackets_stripped_before_counting():
    p_clean, i_clean, tok_clean, _ = compute_counts(PERSONAL_TEXT)
    p_brkt, i_brkt, tok_brkt, _ = compute_counts("[laughter] " + PERSONAL_TEXT)
    assert (p_brkt, i_brkt, tok_brkt) == (p_clean, i_clean, tok_clean)


def test_canonical_pronoun_failure_case():
    """High pronoun count + impersonal topic must count impersonal, not personal.

    This is the failure case for the old pronoun-rate proxy that
    personal_focus_score must correctly handle.
    """
    p, i, _tokens, _score = compute_counts(
        "i think the government should invest more money in public infrastructure"
    )
    assert i > p


def test_write_round_trip(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0001.wav", PERSONAL_TEXT)
        write_row(w, "200/sw2001A-U0002.wav", IMPERSONAL_TEXT)
        write_row(w, "200/sw2001A-U0003.wav", "uh-huh")
        write_row(w, "200/sw2001A-U0004.wav", "[laughter]")

    out_csv = out / "features" / "personal_focus_score.csv"
    n = write_personal_focus_scores(mp, out_csv, workers=1)
    assert n == 4

    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert len(rows) == 5

    # Personal row: counts parse as ints, high score
    assert rows[1][0] == "200/sw2001A-U0001.wav"
    assert int(rows[1][1]) > int(rows[1][2])
    assert float(rows[1][4]) > 0.7

    # Impersonal row: low score
    assert int(rows[2][2]) > int(rows[2][1])
    assert float(rows[2][4]) < 0.3

    # Backchannel: zero counts, one token, empty score
    assert rows[3][1:] == ["0", "0", "1", ""]

    # Brackets only: zero counts, zero tokens, empty score
    assert rows[4][1:] == ["0", "0", "0", ""]


def test_join_contract_against_manifest(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
        write_row(w, "200/sw2001B-U0003.wav", "y")
        write_row(w, "211/sw2113A-U0007.wav", "z")

    feat_csv = out / "features" / "personal_focus_score.csv"
    write_personal_focus_scores(mp, feat_csv, workers=1)

    manifest_rows = list(csv.reader(mp.open(encoding="utf-8")))
    feat_rows = list(csv.reader(feat_csv.open(encoding="utf-8")))

    assert manifest_rows[0] == list(MANIFEST_HEADER)
    assert feat_rows[0] == list(HEADER)
    assert len(manifest_rows) == len(feat_rows)
    for mrow, frow in zip(manifest_rows[1:], feat_rows[1:]):
        assert mrow[0] == frow[0]


def test_resume_uses_cache(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0001.wav", PERSONAL_TEXT)
        write_row(w, "200/sw2001A-U0002.wav", IMPERSONAL_TEXT)

    out_csv = out / "features" / "personal_focus_score.csv"
    write_personal_focus_scores(mp, out_csv, workers=1)
    first_rows = list(csv.reader(out_csv.open(encoding="utf-8")))

    with open(mp, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["200/sw2001A-U0003.wav", "uh-huh"])
    write_personal_focus_scores(mp, out_csv, workers=1)
    second_rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert len(second_rows) == 4
    assert second_rows[1] == first_rows[1]
    assert second_rows[2] == first_rows[2]


def test_run_via_cli_args(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0001.wav", PERSONAL_TEXT)

    class A:
        pass
    a = A()
    a.out_root = str(out)
    a.workers = 1
    a.limit = 0
    a.overwrite = False
    rc = run(a)
    assert rc == 0
    assert (out / "features" / "personal_focus_score.csv").exists()
