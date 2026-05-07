from swb_extract.audio_index import build_index, load_index, resolve, save_index


def test_resolve_2001A_on_disc1(audio_root):
    idx = build_index(audio_root)
    p = resolve(idx, 2001, "A")
    assert p.name == "sw02001.A.wav"
    assert "disc1" in str(p)


def test_index_round_trip(audio_root, tmp_path):
    idx = build_index(audio_root)
    cache = tmp_path / "idx.json"
    save_index(idx, cache)
    loaded = load_index(cache)
    assert loaded == idx


def test_index_covers_both_sides_of_2001(audio_root):
    idx = build_index(audio_root)
    assert (2001, "A") in idx
    assert (2001, "B") in idx
