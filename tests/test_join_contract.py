import csv

from swb_extract.cli import main
from swb_extract.manifest import manifest_path


def test_feature_csv_joins_cleanly_on_utterance_file_name(
    tmp_path, audio_root, transcript_root
):
    out = tmp_path / "utterances_v2"
    cache = tmp_path / "idx.json"
    rc = main([
        "extract", "--call", "2001", "--side", "A",
        "--audio-root", str(audio_root),
        "--transcript-root", str(transcript_root),
        "--out-root", str(out),
        "--cache", str(cache),
    ])
    assert rc == 0

    mp = manifest_path(out)
    with open(mp, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, *data = rows
    assert header == ["Utterance File Name", "Transcript"]
    keys_in_order = [r[0] for r in data]

    # Simulate a feature extractor: emit a sibling CSV under features/
    # with the same key column + same row order + one extra column.
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    feat_path = feat_dir / "dummy.csv"
    with open(feat_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Utterance File Name", "dummy_score"])
        for k in keys_in_order:
            w.writerow([k, "1.0"])

    # Stdlib join: dict by key, no missing rows, no extras.
    with open(feat_path, encoding="utf-8", newline="") as f:
        feat = list(csv.reader(f))[1:]
    feat_map = {r[0]: r[1] for r in feat}

    assert set(feat_map) == set(keys_in_order)
    for k in keys_in_order:
        assert feat_map[k] == "1.0"


def test_features_subcommand_is_stub(tmp_path):
    import pytest

    with pytest.raises(NotImplementedError):
        main(["features", "myfeat", "--out-root", str(tmp_path)])
