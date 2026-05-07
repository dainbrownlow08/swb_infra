import csv

import pytest

from swb_extract.cli import main
from swb_extract.features.pronoun_rate import (
    HEADER,
    compute_rate,
    strip_bracket_tokens,
    write_pronoun_rates,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)

# spaCy model is heavy; load once for all tests.
spacy = pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")


def test_compute_rate_basic_pronoun(nlp):
    # spaCy tags both 'i' (PRON) and 'that' (PRON in context) → 2/3
    assert compute_rate("i like that", nlp) == pytest.approx(2 / 3)


def test_compute_rate_contraction(nlp):
    # 'i' is PRON; 'm and 'sorry' are not. spaCy tokenizes "i'm sorry" → 3 tokens
    rate = compute_rate("i'm sorry", nlp)
    assert rate == pytest.approx(1 / 3)


def test_compute_rate_empty_returns_zero(nlp):
    assert compute_rate("", nlp) == 0.0


def test_compute_rate_no_pronouns(nlp):
    rate = compute_rate("yeah", nlp)
    assert rate == 0.0


def test_strip_bracket_tokens_removes_whole_brackets():
    assert strip_bracket_tokens("yes [laughter] um") == "yes um"
    assert strip_bracket_tokens("[noise] right") == "right"
    assert strip_bracket_tokens("i [laughter-yeah] you") == "i you"


def test_strip_bracket_tokens_keeps_inline_brackets():
    # i[t]- is a partial-word marker, not a whole-bracket token
    assert strip_bracket_tokens("i[t]- yeah") == "i[t]- yeah"


def test_compute_rate_strips_brackets_before_spacy(nlp):
    # 'i [laughter] [noise]' → after strip: 'i' → 1 token (PRON) → rate 1.0
    assert compute_rate("i [laughter] [noise]", nlp) == 1.0


def test_compute_rate_all_brackets_yields_zero(nlp):
    # After stripping the only tokens, nothing remains → 0.0
    assert compute_rate("[laughter] [noise]", nlp) == 0.0


def test_write_pronoun_rates_round_trip(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "i like that")
        write_row(w, "200/sw2001A-U0004.wav", "yeah")
        write_row(w, "200/sw2001A-U0006.wav", "")

    n = write_pronoun_rates(
        mp, out / "features" / "pronoun_rate.csv", workers=1
    )
    assert n == 3

    rows = list(csv.reader((out / "features" / "pronoun_rate.csv").open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    assert float(rows[1][1]) == pytest.approx(2 / 3)
    assert float(rows[2][1]) == 0.0
    assert float(rows[3][1]) == 0.0  # empty transcript


def test_write_pronoun_rates_resume_skips_existing(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "i like that")
    out_csv = out / "features" / "pronoun_rate.csv"

    write_pronoun_rates(mp, out_csv, workers=1)
    contents1 = out_csv.read_text()
    write_pronoun_rates(mp, out_csv, workers=1)
    contents2 = out_csv.read_text()
    assert contents1 == contents2


def test_join_contract_against_manifest(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "i")
        write_row(w, "200/sw2001B-U0003.wav", "you")
        write_row(w, "211/sw2113A-U0007.wav", "ok")

    write_pronoun_rates(
        mp, out / "features" / "pronoun_rate.csv", workers=1
    )

    feat = list(csv.reader((out / "features" / "pronoun_rate.csv").open(encoding="utf-8")))
    manifest = list(csv.reader(mp.open(encoding="utf-8")))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    assert len(feat) == len(manifest)
    for fr, mr in zip(feat[1:], manifest[1:]):
        assert fr[0] == mr[0]


def test_features_dispatch_routes_pronoun_rate(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "i like that")
    rc = main(["features", "pronoun_rate", "--out-root", str(out)])
    assert rc == 0
    assert (out / "features" / "pronoun_rate.csv").exists()
