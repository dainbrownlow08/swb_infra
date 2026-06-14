import csv
from pathlib import Path

from swb_extract.features.question_flags import (
    HEADER,
    is_echo_question,
    question_flag,
    write_question_flags,
)
from swb_extract.manifest import manifest_path, open_appender, write_row


# --- question_flag unit tests ---


def test_wh_onset():
    assert question_flag("what time is it") == 1
    assert question_flag("where did you go") == 1


def test_aux_onset():
    assert question_flag("do you like it") == 1
    assert question_flag("are you sure about that") == 1


def test_declarative_zero():
    assert question_flag("i was working all day") == 0
    assert question_flag("you stayed at the plaza") == 0  # rising-terminal, missed by syntax


def test_empty_none():
    assert question_flag("[laughter]") is None
    assert question_flag("") is None


def test_bracket_stripped_before_onset():
    # leading [noise] is dropped, real onset is "what"
    assert question_flag("[noise] what do you think") == 1


# --- echo question unit tests ---


def test_echo_positive():
    # short, reuses prev content words
    assert is_echo_question("what plaza", "we stayed at the plaza downtown") == 1


def test_echo_too_long():
    long_q = "what exactly did you mean by that whole long statement just now"
    assert is_echo_question(long_q, "by that whole long statement") == 0


def test_echo_no_overlap():
    assert is_echo_question("what time", "the dog ran fast") == 0


# --- integration ---


def _make_trans_root(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "swb_ms98_transcriptions_cleaned"
    for fname, body in files.items():
        call = int(fname[2:6])
        nn = call // 100
        d = root / f"{nn:02d}" / f"{call:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(body, encoding="utf-8")
    return root


def test_write_round_trip(tmp_path):
    body_a = "sw2001A-ms98-a-0002 0.0 5.0 we drove to the lake on saturday\n"
    body_b = "sw2001B-ms98-a-0001 5.5 7.0 what lake\n"  # echo question of "lake"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "we drove to the lake on saturday")
        write_row(w, "200/sw2001B-U0001.wav", "what lake")
    write_question_flags(
        mp, out / "features" / "question_flags.csv", transcript_root=root
    )
    rows = list(csv.reader((out / "features" / "question_flags.csv").open()))
    assert tuple(rows[0]) == HEADER
    by_rel = {r[0]: r[1:] for r in rows[1:]}
    assert by_rel["200/sw2001A-U0002.wav"] == ["0", "0"]   # declarative
    assert by_rel["200/sw2001B-U0001.wav"] == ["1", "1"]   # question + echo of "lake"
