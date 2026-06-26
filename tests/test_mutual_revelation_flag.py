import csv
from pathlib import Path

import pytest

from swb_extract.features.mutual_revelation_flag import (
    HEADER,
    MIN_TOKENS,
    is_personal_anecdote,
    lookup_mutual_revelation,
    write_mutual_revelation_flags,
)
from swb_extract.features.turn_gap import build_text_index, build_turn_gap_index
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)


def _make_trans_root(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "swb_ms98_transcriptions_cleaned"
    for fname, body in files.items():
        call = int(fname[2:6])
        nn = call // 100
        d = root / f"{nn:02d}" / f"{call:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(body, encoding="utf-8")
    return root


# --- is_personal_anecdote unit tests ---


def test_anecdote_positive():
    assert is_personal_anecdote(
        "i was working at a consulting firm and we had a really casual dress code"
    )


def test_anecdote_no_first_person():
    # "they were working" — no first-person, fails
    assert not is_personal_anecdote(
        "they were working at a firm and had casual dress every day"
    )


def test_anecdote_no_past_tense():
    # "i like to work" — present tense, fails
    assert not is_personal_anecdote("i like to work from home every day")


def test_anecdote_too_short():
    # < MIN_TOKENS
    assert not is_personal_anecdote("i was")
    # exactly MIN_TOKENS-1 tokens with first-person + past
    s = " ".join(["i", "was"] + ["x"] * (MIN_TOKENS - 3))
    assert not is_personal_anecdote(s)


def test_anecdote_strips_brackets():
    assert is_personal_anecdote("[laughter] i was in school for a long time")
    # Bracket-only utterance fails
    assert not is_personal_anecdote("[laughter] [noise]")


def test_anecdote_present_perfect():
    # "i've been working" — "been" is in past-tense pattern
    assert is_personal_anecdote("i've been working at this company for ten years now")


def test_anecdote_my_var():
    # "my" counts as first-person
    assert is_personal_anecdote("my mom always made dinner before dad got home")


def test_anecdote_verbs_outside_regex_list():
    """spaCy must catch past-tense verbs that the old handcrafted regex
    list missed. These were the v1 false negatives."""
    assert is_personal_anecdote("i traveled to paris last summer with my family")
    assert is_personal_anecdote("i played tennis and watched movies all afternoon")
    assert is_personal_anecdote("we ate pizza at that little place on the corner")
    assert is_personal_anecdote("my brother called me last night about the news")
    assert is_personal_anecdote("i learned how to drive when i was sixteen years old")


# --- cross-utterance lookup tests ---


def test_first_of_conversation_returns_none(tmp_path):
    body_a = "sw2001A-ms98-a-0002 0.0 5.0 i was working at a consulting firm and we had it good\n"
    body_b = "sw2001B-ms98-a-0001 5.5 10.0 i had the same experience at my first job\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    # A.U002 starts at 0.0 → first-of-conv → None
    assert lookup_mutual_revelation(merged, text, "200/sw2001A-U0002.wav") is None
    # B.U001 prev=A.U002 — both anecdotes, different speakers → 1
    assert lookup_mutual_revelation(merged, text, "200/sw2001B-U0001.wav") == 1


def test_only_current_anecdote(tmp_path):
    # A: backchannel; B: anecdote → previous is A but A is not anecdote → 0
    body_a = "sw2001A-ms98-a-0002 0.0 1.0 yeah\n"
    body_b = "sw2001B-ms98-a-0001 1.5 6.0 i was working at a consulting firm last year\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    assert lookup_mutual_revelation(merged, text, "200/sw2001B-U0001.wav") == 0


def test_only_previous_anecdote(tmp_path):
    # A: anecdote; B: backchannel — for B, current is not anecdote → 0
    body_a = "sw2001A-ms98-a-0002 0.0 5.0 i was working at a firm and we had a great time\n"
    body_b = "sw2001B-ms98-a-0001 5.5 6.0 yeah really\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    assert lookup_mutual_revelation(merged, text, "200/sw2001B-U0001.wav") == 0


def test_same_speaker_consecutive_returns_zero(tmp_path):
    # A.U002 then A.U003 — same speaker, NOT mutual revelation even if both anecdotes
    body_a = (
        "sw2001A-ms98-a-0002 0.0 5.0 i was working at a consulting firm last year\n"
        "sw2001A-ms98-a-0003 5.5 10.0 i had to commute every day for work\n"
    )
    body_b = "sw2001B-ms98-a-0001 100.0 105.0 yeah okay\n"  # far away, not relevant
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    # A.U003 prev=A.U002 — same speaker → 0 (not mutual)
    assert lookup_mutual_revelation(merged, text, "200/sw2001A-U0003.wav") == 0


def test_both_anecdotes_different_speakers(tmp_path):
    body_a = "sw2001A-ms98-a-0002 0.0 5.0 i was a teacher for many years before retiring\n"
    body_b = "sw2001B-ms98-a-0001 5.5 10.0 my husband was also a teacher and we loved it\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    assert lookup_mutual_revelation(merged, text, "200/sw2001B-U0001.wav") == 1


def test_unknown_rel_path_returns_none(tmp_path):
    body_a = "sw2001A-ms98-a-0002 0.0 1.0 yeah\n"
    body_b = "sw2001B-ms98-a-0001 0.5 1.5 yes\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    assert lookup_mutual_revelation(merged, text, "200/sw2001A-U9999.wav") is None


# --- end-to-end / contract tests ---


def test_write_round_trip(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "")
        write_row(w, "200/sw2001A-U0004.wav", "")

    n = write_mutual_revelation_flags(
        mp,
        out / "features" / "mutual_revelation_flag.csv",
        transcript_root=transcript_root,
    )
    assert n == 2

    rows = list(csv.reader((out / "features" / "mutual_revelation_flag.csv").open()))
    assert tuple(rows[0]) == HEADER
    # Each non-first-of-conversation row should be a valid 0 or 1
    for row in rows[1:]:
        if row[1]:
            assert row[1] in {"0", "1"}


def test_join_contract_against_manifest(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "")
        write_row(w, "200/sw2001A-U0004.wav", "")
        write_row(w, "200/sw2001B-U0003.wav", "")

    write_mutual_revelation_flags(
        mp, out / "features" / "mutual_revelation_flag.csv",
        transcript_root=transcript_root,
    )
    feat = list(csv.reader((out / "features" / "mutual_revelation_flag.csv").open()))
    manifest = list(csv.reader(mp.open()))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    assert len(feat) == len(manifest)
    for fr, mr in zip(feat[1:], manifest[1:]):
        assert fr[0] == mr[0]


def test_features_dispatch_routes_mutual_revelation(tmp_path, transcript_root):
    from swb_extract.cli import main

    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "")

    rc = main([
        "features", "mutual_revelation_flag",
        "--out-root", str(out),
        "--transcript-root", str(transcript_root),
    ])
    assert rc == 0
    assert (out / "features" / "mutual_revelation_flag.csv").exists()
