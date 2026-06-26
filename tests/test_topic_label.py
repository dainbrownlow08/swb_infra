import csv
from pathlib import Path

import pytest

from swb_extract.features.topic_label import (
    HEADER,
    build_topic_index,
    load_conv_topics,
    load_topic_tab,
    run,
    write_topic_labels,
)
from swb_extract.manifest import MANIFEST_HEADER, manifest_path, open_appender, write_row

REPO = Path(__file__).resolve().parent.parent
LDC_CONV = REPO / "tables" / "conv_tab.csv"
LDC_TOPIC = REPO / "tables" / "topic_tab.csv"


@pytest.fixture(scope="session")
def ldc_tables():
    if not LDC_CONV.is_file() or not LDC_TOPIC.is_file():
        pytest.skip("LDC tables missing")
    return LDC_CONV, LDC_TOPIC


def test_load_topic_tab(ldc_tables):
    _, topic = ldc_tables
    topics = load_topic_tab(topic)
    # Switchboard ships ~70 topics
    assert 60 <= len(topics) <= 100
    # Known mapping: ivi_no 303 → CLOTHING AND DRESS
    assert topics[303] == "CLOTHING AND DRESS"
    # Known mapping: ivi_no 317 → AFFIRMATIVE ACTION
    assert topics[317] == "AFFIRMATIVE ACTION"


def test_load_conv_topics(ldc_tables):
    conv, _ = ldc_tables
    convs = load_conv_topics(conv)
    # Call 2001 → ivi_no 303
    assert convs[2001] == 303


def test_build_topic_index(ldc_tables):
    conv, topic = ldc_tables
    idx = build_topic_index(conv, topic)
    # Call 2001 → CLOTHING AND DRESS
    assert idx[2001] == "CLOTHING AND DRESS"


def test_unk_ivi_resolves_to_unk_topic(ldc_tables):
    """Calls with ivi_no=UNK in conv_tab (e.g. 3178) should map to "UNK"."""
    conv, topic = ldc_tables
    idx = build_topic_index(conv, topic)
    # Call 3178 has ivi_no=UNK — must resolve to UNK rather than be missing
    assert idx.get(3178) == "UNK"


def test_write_topic_labels_round_trip(tmp_path, ldc_tables):
    conv, topic = ldc_tables
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi")
        write_row(w, "200/sw2001B-U0001.wav", "yeah")

    idx = build_topic_index(conv, topic)
    out_csv = out / "features" / "topic_label.csv"
    n = write_topic_labels(mp, out_csv, idx)
    assert n == 2

    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert len(rows) == 3  # header + 2 data rows
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    assert rows[1][1] == "CLOTHING AND DRESS"
    assert rows[2][1] == "CLOTHING AND DRESS"  # both sides share topic


def test_join_contract_against_manifest(tmp_path, ldc_tables):
    """Topic label CSV must join cleanly to manifest by Utterance File Name."""
    conv, topic = ldc_tables
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
        write_row(w, "200/sw2001B-U0003.wav", "y")
        write_row(w, "211/sw2113A-U0007.wav", "z")

    idx = build_topic_index(conv, topic)
    feat_csv = out / "features" / "topic_label.csv"
    write_topic_labels(mp, feat_csv, idx)

    manifest_rows = list(csv.reader(mp.open(encoding="utf-8")))
    feat_rows = list(csv.reader(feat_csv.open(encoding="utf-8")))

    assert manifest_rows[0] == list(MANIFEST_HEADER)
    assert feat_rows[0] == list(HEADER)
    assert len(manifest_rows) == len(feat_rows)
    for mrow, frow in zip(manifest_rows[1:], feat_rows[1:]):
        assert mrow[0] == frow[0]


def test_run_via_cli_args(tmp_path, ldc_tables):
    conv, topic = ldc_tables
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")

    class A:
        pass
    a = A()
    a.conv_tab = str(conv)
    a.topic_tab = str(topic)
    a.out_root = str(out)
    rc = run(a)
    assert rc == 0
    out_csv = out / "features" / "topic_label.csv"
    assert out_csv.exists()
    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert rows[1][1] == "CLOTHING AND DRESS"
