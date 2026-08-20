import csv

import pytest

from swb_extract.features.inhouse.pronoun_per_second import (
    HEADER,
    count_personal_pronouns,
    compute_rate_per_second,
    write_pronouns_per_second,
)
from swb_extract.manifest import manifest_path, open_appender, write_row


def test_counts_first_and_second_person_only():
    assert count_personal_pronouns("i think you know") == 2
    assert count_personal_pronouns("we told them it was theirs") == 1  # we only
    # third person and thing-reference excluded (the v2 point)
    assert count_personal_pronouns("it that they he she them what there") == 0


def test_contractions_match_by_base():
    assert count_personal_pronouns("i'm sure you're right we'll see") == 3
    assert count_personal_pronouns("it's fine") == 0  # "it" not personal


def test_yall_and_possessives():
    assert count_personal_pronouns("y'all keep your mine and ours") == 4


def test_partial_word_attempt_does_not_match():
    assert count_personal_pronouns("i[t]- i mean") == 1  # only the bare "i"


def test_laughed_personal_pronoun_counts():
    assert count_personal_pronouns("[laughter-i] know") == 1


def test_markup_and_brackets_ignored():
    assert count_personal_pronouns("[laughter] <b_aside> you <e_aside>") == 1


def test_rate_semantics():
    assert compute_rate_per_second("i you", 2.0) == 1.0
    assert compute_rate_per_second("i you", None) is None
    assert compute_rate_per_second("i you", 0.0) is None


def test_write_round_trip(tmp_path):
    out = tmp_path / "utterances_v2"
    out.mkdir()
    trans = tmp_path / "trans"
    d = trans / "20" / "2001"
    d.mkdir(parents=True)
    (d / "sw2001A-ms98-a-trans.text").write_text(
        "sw2001A-ms98-a-0002 0.0 2.0 i think you know\n", encoding="utf-8"
    )
    (d / "sw2001A-ms98-a-word.text").write_text(
        "sw2001A-ms98-a-0002 0.1 0.4 i\n"
        "sw2001A-ms98-a-0002 0.4 0.9 think\n"
        "sw2001A-ms98-a-0002 0.9 1.3 you\n"
        "sw2001A-ms98-a-0002 1.3 1.9 know\n",
        encoding="utf-8",
    )
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "i think you know")
    out_csv = out / "features" / "pronoun_per_second.csv"
    n = write_pronouns_per_second(mp, out_csv, transcript_root=trans)
    assert n == 1
    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert float(rows[1][1]) == pytest.approx(1.0)  # 2 pronouns / 2 s
