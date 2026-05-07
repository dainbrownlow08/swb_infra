"""Combined tests for the three per-second sibling features."""
import csv
from pathlib import Path

import pytest

from swb_extract.features.filler_word_per_second import (
    HEADER as FILLER_PS_HEADER,
    compute_rate_per_second as filler_ps,
    write_filler_words_per_second,
)
from swb_extract.features.repetition_per_second import (
    HEADER as REP_PS_HEADER,
    compute_rate_per_second as rep_ps,
    write_repetitions_per_second,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)

REPO = Path(__file__).resolve().parent.parent
TRANSCRIPT_ROOT = REPO / "swb_ms98_transcriptions_cleaned"


@pytest.fixture(scope="session")
def transcript_root():
    if not TRANSCRIPT_ROOT.is_dir():
        pytest.skip(f"missing transcript root: {TRANSCRIPT_ROOT}")
    return TRANSCRIPT_ROOT


# ---- pure-function unit tests (no IO) ----

def test_filler_ps_compute():
    # 'um yeah um' has 2 filler hits in 3 tokens; over 1.5s → 2/1.5 = 1.333...
    assert filler_ps("um yeah um", 1.5) == pytest.approx(2 / 1.5)


def test_filler_ps_zero_duration_returns_none():
    assert filler_ps("um", 0.0) is None
    assert filler_ps("um", -1.0) is None
    assert filler_ps("um", None) is None


def test_filler_ps_empty_text_returns_zero():
    assert filler_ps("", 5.0) == 0.0
    assert filler_ps("[laughter] [noise]", 5.0) == 0.0  # all stripped


def test_repetition_ps_compute():
    # 'the the dog' has 1 repeat (the), duration 2s → 1/2 = 0.5
    assert rep_ps("the the dog", 2.0) == pytest.approx(0.5)


def test_repetition_ps_strips_brackets():
    # '[laughter] [laughter] right' → after strip 'right' → 0 repeats
    assert rep_ps("[laughter] [laughter] right", 5.0) == 0.0


def test_repetition_ps_empty_returns_zero():
    assert rep_ps("", 3.0) == 0.0


# ---- integration: against real golden transcript ----

def test_filler_ps_round_trip_on_golden_call(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        # sw2001A-U0002: ms98 says start=0.977625, end=11.561375 → dur ≈ 10.5838s
        # transcript has 3 single-word fillers (um, like, um); no multi-word match
        # → rate ≈ 3 / 10.5838 ≈ 0.2834
        write_row(
            w,
            "200/sw2001A-U0002.wav",
            "hi um yeah i'd like to talk about how you dress for work and "
            "and um what do you normally what type of outfit do you "
            "normally have to wear",
        )

    n = write_filler_words_per_second(
        mp,
        out / "features" / "filler_word_per_second.csv",
        transcript_root=transcript_root,
    )
    assert n == 1
    rows = list(csv.reader((out / "features" / "filler_word_per_second.csv").open()))
    assert tuple(rows[0]) == FILLER_PS_HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    rate = float(rows[1][1])
    assert 0.27 < rate < 0.30, f"expected ~0.283, got {rate}"


def test_repetition_ps_round_trip_on_golden_call(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(
            w,
            "200/sw2001A-U0002.wav",
            "hi um yeah i'd like to talk about how you dress for work and "
            "and um what do you normally what type of outfit do you "
            "normally have to wear",
        )

    n = write_repetitions_per_second(
        mp,
        out / "features" / "repetition_per_second.csv",
        transcript_root=transcript_root,
    )
    assert n == 1
    rows = list(csv.reader((out / "features" / "repetition_per_second.csv").open()))
    assert tuple(rows[0]) == REP_PS_HEADER
    rate = float(rows[1][1])
    # 7 repeats / ~10.58s ≈ 0.66
    assert 0.5 < rate < 0.8, f"expected ~0.66, got {rate}"


def test_join_contract_against_manifest(tmp_path, transcript_root):
    """All three sibling CSVs must align to manifest 1:1 in row order."""
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(
            w,
            "200/sw2001A-U0002.wav",
            "hi um yeah i'd like to talk about",
        )

    write_filler_words_per_second(
        mp, out / "features" / "filler_word_per_second.csv",
        transcript_root=transcript_root,
    )
    write_repetitions_per_second(
        mp, out / "features" / "repetition_per_second.csv",
        transcript_root=transcript_root,
    )

    manifest = list(csv.reader(mp.open()))
    filler = list(csv.reader((out / "features" / "filler_word_per_second.csv").open()))
    rep = list(csv.reader((out / "features" / "repetition_per_second.csv").open()))

    assert manifest[0] == list(MANIFEST_HEADER)
    assert filler[0] == list(FILLER_PS_HEADER)
    assert rep[0] == list(REP_PS_HEADER)
    assert len(manifest) == len(filler) == len(rep)
    for m_row, f_row, r_row in zip(manifest[1:], filler[1:], rep[1:]):
        assert m_row[0] == f_row[0] == r_row[0]


def test_features_dispatch_routes_per_second_features(tmp_path, transcript_root):
    from swb_extract.cli import main

    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "um yeah um")

    rc1 = main([
        "features", "filler_word_per_second",
        "--out-root", str(out),
        "--transcript-root", str(transcript_root),
    ])
    assert rc1 == 0
    assert (out / "features" / "filler_word_per_second.csv").exists()

    rc2 = main([
        "features", "repetition_per_second",
        "--out-root", str(out),
        "--transcript-root", str(transcript_root),
    ])
    assert rc2 == 0
    assert (out / "features" / "repetition_per_second.csv").exists()


# ---- pronoun_per_second has spaCy dep; gate on availability ----

def test_pronoun_ps_compute():
    spacy = pytest.importorskip("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")

    from swb_extract.features.pronoun_per_second import compute_rate_per_second
    # 'i like that' has 2 pronouns (i, that); 1.5s duration → 2/1.5 = 1.333
    assert compute_rate_per_second("i like that", 1.5, nlp=nlp) == pytest.approx(2 / 1.5)
    assert compute_rate_per_second("i", 0, nlp=nlp) is None
    assert compute_rate_per_second("[laughter]", 1.0, nlp=nlp) == 0.0
