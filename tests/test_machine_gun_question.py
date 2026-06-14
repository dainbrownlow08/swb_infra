import csv
from pathlib import Path

import pytest

import swb_extract.features.machine_gun_question as mg
from swb_extract.manifest import manifest_path, open_appender, write_row

A1, A2, A7 = "200/sw2001A-U0001.wav", "200/sw2001A-U0002.wav", "200/sw2001A-U0007.wav"
B3, B4, B5, B6 = (
    "200/sw2001B-U0003.wav",
    "200/sw2001B-U0004.wav",
    "200/sw2001B-U0005.wav",
    "200/sw2001B-U0006.wav",
)
ORDER = [A1, A2, B3, B4, B5, B6, A7]


def _write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _setup(tmp_path: Path) -> Path:
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        for rel in ORDER:
            write_row(w, rel, "text")
    fdir = out / "features"
    # (QF, RT, tok, onset, bc, pitch) per row:
    # A1: full MG — QF1 tok3 onset0.2 pitch200 (A p75=175) → score 4, flag 1
    # A2: QF0, RT unmeasured → empty
    # B3: backchannel → empty
    # B4: QF0 RT0 → score 0
    # B5: same absolute pitch as A1 (200) but B p75=220 → no pitch point → 3
    # B6: filler, QF0 RT0 → 0 (gives B a third pitch)
    # A7: QF1 but long, onset unmeasured, pitch unmeasured → score 1
    _write_csv(
        fdir / "question_flags.csv",
        ["Utterance File Name", "Question Flag", "Echo Question Flag"],
        [[A1, "1", "0"], [A2, "0", "0"], [B3, "0", "0"], [B4, "0", "0"],
         [B5, "1", "0"], [B6, "0", "0"], [A7, "1", "0"]],
    )
    _write_csv(
        fdir / "rising_terminal.csv",
        ["Utterance File Name", "Rising Terminal Flag", "Terminal F0 Slope"],
        [[A1, "0", "-10.0"], [A2, "", ""], [B3, "", ""], [B4, "0", "-5.0"],
         [B5, "0", "2.0"], [B6, "0", "0.0"], [A7, "0", "-1.0"]],
    )
    _write_csv(
        fdir / "token_count.csv",
        ["Utterance File Name", "token_count"],
        [[A1, "3"], [A2, "5"], [B3, "1"], [B4, "6"], [B5, "3"], [B6, "8"], [A7, "12"]],
    )
    _write_csv(
        fdir / "fto.csv",
        ["Utterance File Name", "FTO Sec", "Onset Gap Sec", "Turn Initial Flag",
         "Backchannel Flag", "Interjection Flag"],
        [[A1, "0.2", "0.2", "1", "0", "0"], [A2, "", "0.9", "0", "0", "0"],
         [B3, "", "", "0", "1", "0"], [B4, "0.3", "0.3", "1", "0", "0"],
         [B5, "0.1", "0.1", "1", "0", "0"], [B6, "0.4", "0.4", "1", "0", "0"],
         [A7, "", "", "1", "0", "0"]],
    )
    _write_csv(
        fdir / "pitch.csv",
        ["Utterance File Name", "pitch mean", "pitch std", "pitch range"],
        [[A1, "200", "10", "50"], [A2, "100", "10", "50"], [B3, "", "", ""],
         [B4, "150", "10", "50"], [B5, "200", "10", "50"], [B6, "240", "10", "50"],
         [A7, "", "", ""]],
    )
    return out


def test_compose_unit():
    # intent gate indeterminate: no syntax, prosody unmeasured
    assert mg.compose("0", "", "3", "0.1", "0", "200", 150.0) == (None, None)
    # backchannel always empty, even if other cells look question-like
    assert mg.compose("1", "1", "1", "0.0", "1", "300", 150.0) == (None, None)
    # measured non-question: honest zero
    assert mg.compose("0", "0", "3", "0.1", "0", "200", 150.0) == (0, 0)
    # full house
    assert mg.compose("1", "0", "3", "0.1", "0", "200", 150.0) == (4, 1)
    # rising terminal alone supplies intent
    assert mg.compose("0", "1", "3", "0.1", "0", "200", 150.0) == (4, 1)
    # missing pitch/baseline: conservative no-point
    assert mg.compose("1", "0", "3", "0.1", "0", "", 150.0) == (3, 0)
    assert mg.compose("1", "0", "3", "0.1", "0", "200", None) == (3, 0)


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "MIN_SIDE_PITCH_N", 2)
    out = _setup(tmp_path)
    n = mg.write_machine_gun(
        manifest_path(out), out / "features",
        out / "features" / "machine_gun_question.csv",
    )
    assert n == 7
    rows = list(csv.reader((out / "features" / "machine_gun_question.csv").open()))
    assert tuple(rows[0]) == mg.HEADER
    got = {r[0]: (r[1], r[2]) for r in rows[1:]}
    assert got[A1] == ("4", "1")   # full machine-gun question
    assert got[A2] == ("", "")     # intent indeterminate
    assert got[B3] == ("", "")     # backchannel
    assert got[B4] == ("0", "0")   # measured non-question
    assert got[B6] == ("0", "0")
    assert got[A7] == ("1", "0")   # question, but long/slow/unpitched


def test_pitch_is_speaker_relative(tmp_path, monkeypatch):
    """Same absolute pitch (200 Hz): high for side A (p75=175), not for side B
    (p75=220). The old population-median version scored these identically —
    this is the gender-confound regression test."""
    monkeypatch.setattr(mg, "MIN_SIDE_PITCH_N", 2)
    out = _setup(tmp_path)
    mg.write_machine_gun(
        manifest_path(out), out / "features",
        out / "features" / "machine_gun_question.csv",
    )
    rows = list(csv.reader((out / "features" / "machine_gun_question.csv").open()))
    got = {r[0]: (r[1], r[2]) for r in rows[1:]}
    assert got[A1] == ("4", "1")   # pitch 200 ≥ A's own p75 (175) → point
    assert got[B5] == ("3", "0")   # pitch 200 < B's own p75 (220) → no point


def test_out_of_sync_input_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "MIN_SIDE_PITCH_N", 2)
    out = _setup(tmp_path)
    # Corrupt one input's order
    tok = out / "features" / "token_count.csv"
    rows = list(csv.reader(tok.open()))
    rows[1], rows[2] = rows[2], rows[1]
    _write_csv(tok, rows[0], rows[1:])
    with pytest.raises(mg.CompositeInputError, match="token_count.csv out of sync"):
        mg.write_machine_gun(
            manifest_path(out), out / "features",
            out / "features" / "machine_gun_question.csv",
        )


def test_missing_input_raises(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, A1, "text")
    with pytest.raises(mg.CompositeInputError, match="missing input"):
        mg.write_machine_gun(mp, out / "features", out / "features" / "mg.csv")
