"""Thomas et al. (2018) replication — one runnable check per mechanism."""
import csv
import wave
from pathlib import Path

import numpy as np
import pytest

from swb_extract.features.thomas2018 import (
    EXTRACTORS,
    PAPER_LOADINGS,
    _bounds,
    _prosody,
    _terms,
    boplen,
    lv,
    olap,
    poplen,
    ppron,
    pv,
    rept,
    repu,
    wpp,
    wps,
    wpu,
)
from swb_extract.manifest import manifest_path, open_appender, write_row

REPO = Path(__file__).resolve().parent.parent
TRANSCRIPT_ROOT = REPO / "swb_ms98_transcriptions_cleaned"
GOLDEN_WAV = REPO / "utterances_v2" / "200" / "sw2001A-U0002.wav"
GOLDEN_TEXT = (
    "hi um yeah i'd like to talk about how you dress for work and and um what do you "
    "normally what type of outfit do you normally have to wear"
)


def test_eleven_extractors_named_as_in_the_paper():
    names = [m.FEATURE_NAME for m in EXTRACTORS]
    assert names == ["ppron", "wps", "wpu", "wpp", "boplen", "poplen", "pv", "lv", "olap", "rept", "repu"]
    assert set(PAPER_LOADINGS) == set(names)
    for m in EXTRACTORS:
        assert m.HEADER[0] == "Utterance File Name" and all(c.startswith(m.FEATURE_NAME) for c in m.HEADER[1:])


def test_wpu_and_ppron_count_words_and_first_second_person_pronouns():
    text = "i'd like to [laughter] talk about how you dress [laughter-yeah] and he said"
    assert wpu.count_words(text) == 12  # [laughter] dropped, [laughter-yeah] -> yeah
    assert ppron.count_pronouns(text) == 2  # i'd, you — not he


def test_terms_drop_stopwords_and_fillers_then_stem():
    assert _terms.terms("um the treatments were uh-huh for migraines because_1") == {"treatment", "migrain"}
    assert _terms.terms("uh-huh um-hum") == {"um-hum"}  # the paper's list is exactly um/uh/uh-huh


def test_repeat_index_uses_the_same_side_previous_utterance():
    texts = {
        (1, "A", 1): "beta blockers for migraines",
        (1, "B", 2): "blockers",
        (1, "A", 3): "the beta blockers then",
        (1, "A", 5): "uh-huh",
    }
    reps = _terms.build_repeat_index(texts)
    assert (1, "A", 1) not in reps and (1, "B", 2) not in reps  # each side's first utterance
    assert reps[(1, "A", 3)] == 2  # beta, blocker — shared with A's previous, not B's
    assert reps[(1, "A", 5)] == 0


def test_onset_timing_rules():
    # A 0–2 | B 1–1.5 overlaps | B 2.5–3 post-other gap .5 | B 3.2–4 own-own | A 4–5 tie -> 0
    utts = sorted([(0.0, 2.0, "A", 1), (1.0, 1.5, "B", 2), (2.5, 3.0, "B", 4), (3.2, 4.0, "B", 6), (4.0, 5.0, "A", 7)])
    t = _bounds.onset_timing(9, utts)
    assert t[(9, "A", 1)] == (0, None)
    assert t[(9, "B", 2)] == (1, None)
    assert t[(9, "B", 4)] == (0, pytest.approx(0.5))
    assert t[(9, "B", 6)] == (0, None)
    assert t[(9, "A", 7)] == (0, pytest.approx(0.0))


def test_aggregates_are_pooled_not_averaged():
    rows = [
        {"wpu": "10", "wps sec": "4.0", "ppron": "2", "wpp pauses": "3", "boplen n": "3",
         "boplen sec": "0.6", "poplen": "0.5", "olap": "1", "rept": "2", "repu": "1"},
        {"wpu": "2", "wps sec": "", "ppron": "0", "wpp pauses": "", "boplen n": "0",
         "boplen sec": "0.0", "poplen": "", "olap": "0", "rept": "", "repu": ""},
    ]
    assert wpu.aggregate(rows) == 6.0
    assert wps.aggregate(rows) == 2.5  # unmeasured duration drops that row from BOTH sums
    assert ppron.aggregate(rows) == pytest.approx(2 / 12)
    assert wpp.aggregate(rows) == pytest.approx(10 / 3)
    assert boplen.aggregate(rows) == pytest.approx(0.2)
    assert poplen.aggregate(rows) == 0.5 and olap.aggregate(rows) == 0.5
    assert rept.aggregate(rows) == 2.0 and repu.aggregate(rows) == 1.0
    a, b = np.array([100.0, 110.0, 120.0]), np.array([200.0, 202.0])

    def prow(x, p):
        return {f"{p} n": str(x.size), f"{p} mean": repr(float(x.mean())), f"{p} var": repr(float(x.var()))}

    assert pv.aggregate([prow(a, "pv"), prow(b, "pv")]) == pytest.approx(np.concatenate([a, b]).var())
    assert lv.aggregate([prow(a, "lv"), {"lv n": "0", "lv mean": "", "lv var": ""}]) == pytest.approx(a.var())
    assert pv.aggregate([{"pv n": "", "pv mean": "", "pv var": ""}]) is None


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())


def test_prosody_counts_the_unvoiced_gap_between_two_tones(tmp_path):
    sr = 8000
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 150 * t)
    _write_wav(tmp_path / "two_tones.wav", np.concatenate([tone, np.zeros(int(0.3 * sr)), tone]), sr)
    nv, f0m, f0v, rm, rv, pn, ps = _prosody.extract_prosody(tmp_path / "two_tones.wav")
    assert nv > 50 and abs(f0m - 150) < 5 and f0v < 25  # a steady tone: tiny Hz² variance
    assert rm > 0 and rv < rm * rm  # RMS of a constant-amplitude tone barely varies
    assert pn == 1 and abs(ps - 0.3) < 0.1  # ONE between-own pause of ~0.3 s
    _write_wav(tmp_path / "silence.wav", np.zeros(sr), sr)
    assert _prosody.extract_prosody(tmp_path / "silence.wav") == _prosody.NO_VOICING
    assert _prosody.extract_prosody(tmp_path / "missing.wav") is None


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    if not TRANSCRIPT_ROOT.is_dir() or not GOLDEN_WAV.is_file():
        pytest.skip("needs the cleaned transcripts + the golden slice")
    out = tmp_path_factory.mktemp("thomas2018") / "utterances_v2"
    (out / "200").mkdir(parents=True)
    (out / "200" / "sw2001A-U0002.wav").symlink_to(GOLDEN_WAV)
    with open_appender(manifest_path(out)) as w:
        write_row(w, "200/sw2001A-U0002.wav", GOLDEN_TEXT)
        write_row(w, "200/sw2001A-U0004.wav", "um-hum")  # no wav -> acoustic cells blank
    return out


def test_cli_writes_all_eleven_in_manifest_order_and_the_side_table(scratch):
    from swb_extract.cli import main

    rels = ["200/sw2001A-U0002.wav", "200/sw2001A-U0004.wav"]
    for m in EXTRACTORS:
        rc = main(["features", m.FEATURE_NAME, "--out-root", str(scratch),
                   "--transcript-root", str(TRANSCRIPT_ROOT), "--workers", "1"])
        assert rc == 0
        rows = list(csv.reader((scratch / "features" / f"{m.FEATURE_NAME}.csv").open()))
        assert tuple(rows[0]) == m.HEADER and [r[0] for r in rows[1:]] == rels
    by = {m.FEATURE_NAME: list(csv.reader((scratch / "features" / f"{m.FEATURE_NAME}.csv").open()))
          for m in EXTRACTORS}
    assert by["wpu"][1][1] == "30" and by["wpu"][2][1] == "1"
    assert by["ppron"][1][1] == "4"  # i'd, you, you, you
    # B's "okay hi" (0.82-2.22 s) is still running at A's first word (1.22 s): overlap, so no pause
    assert by["olap"][1][1] == "1" and by["poplen"][1][1] == ""
    assert by["rept"][1][1] == "" and by["rept"][2][1] == "0"  # side-first blank; "um-hum" shares nothing
    assert by["pv"][1][1] != "" and by["pv"][2] == [rels[1], "", "", ""]  # 19 pauses on the golden slice
    assert by["boplen"][1][1] == "19" and by["wpp"][1][1] == "19"
    assert (scratch / "derived" / "thomas2018_prosody.csv").is_file()  # one shared acoustic pass

    assert main(["thomas2018-side", "--out-root", str(scratch)]) == 0
    side = list(csv.DictReader((scratch / "derived" / "thomas2018_side.csv").open()))
    assert len(side) == 1 and side[0]["side"] == "A" and side[0]["n_utts"] == "2"
    assert 2.0 < float(side[0]["wps"]) < 4.0 and float(side[0]["wpu"]) == 15.5
    assert float(side[0]["ppron"]) == pytest.approx(4 / 31)
    assert 0.1 < float(side[0]["boplen"]) < 0.3 and 100 < float(side[0]["pv"]) < 5000


def test_prosody_cache_checkpoints_and_resumes(tmp_path, monkeypatch):
    """Second run must reuse the cache (no re-extraction); --overwrite must recompute."""
    sr = 8000
    t = np.arange(sr) / sr
    out = tmp_path / "utterances_v2"; (out / "200").mkdir(parents=True)
    for i in (2, 4):
        _write_wav(out / "200" / f"sw2001A-U{i:04d}.wav", 0.5 * np.sin(2 * np.pi * 150 * t), sr)
    with open_appender(manifest_path(out)) as w:
        for i in (2, 4):
            write_row(w, f"200/sw2001A-U{i:04d}.wav", "")
    calls = []
    real = _prosody.extract_prosody
    monkeypatch.setattr(_prosody, "extract_prosody", lambda p: (calls.append(str(p)), real(p))[1])
    first = _prosody.ensure_prosody(manifest_path(out), out, workers=1)
    assert len(calls) == 2 and (out / "derived" / "thomas2018_prosody.csv").is_file()
    second = _prosody.ensure_prosody(manifest_path(out), out, workers=1)
    assert len(calls) == 2 and second == first          # fully served from the checkpoint
    _prosody.ensure_prosody(manifest_path(out), out, workers=1, overwrite=True)
    assert len(calls) == 4                               # --overwrite recomputes
