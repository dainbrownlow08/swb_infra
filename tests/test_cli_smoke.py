import csv

from swb_extract.cli import main
from swb_extract.manifest import MANIFEST_HEADER, manifest_path
from swb_extract.transcripts import parse_transcript


def test_extract_call_2001_side_A(tmp_path, repo, audio_root, transcript_root, golden_transcript):
    out = tmp_path / "utterances_v2"
    cache = tmp_path / "idx.json"

    rc = main([
        "extract",
        "--call", "2001",
        "--side", "A",
        "--audio-root", str(audio_root),
        "--transcript-root", str(transcript_root),
        "--out-root", str(out),
        "--cache", str(cache),
    ])
    assert rc == 0

    expected = list(parse_transcript(golden_transcript))
    n = len(expected)

    mp = manifest_path(out)
    rows = list(csv.reader(mp.open(encoding="utf-8")))
    assert tuple(rows[0]) == MANIFEST_HEADER
    assert len(rows) == n + 1

    # Every wav filename's NNNN must match a transcript utt_num — no row-shift.
    seen_utt_nums = []
    for rel, _text in rows[1:]:
        utt = int(rel.rsplit("-U", 1)[1].split(".")[0])
        seen_utt_nums.append(utt)
        assert (out / rel).exists()
    assert seen_utt_nums == [u.utt_num for u in expected]


def test_resume_skips_already_done(tmp_path, audio_root, transcript_root):
    out = tmp_path / "utterances_v2"
    cache = tmp_path / "idx.json"
    args = [
        "extract", "--call", "2001", "--side", "A",
        "--audio-root", str(audio_root),
        "--transcript-root", str(transcript_root),
        "--out-root", str(out),
        "--cache", str(cache),
    ]
    assert main(args) == 0
    mp = manifest_path(out)
    size1 = mp.stat().st_size
    assert main(args) == 0
    assert mp.stat().st_size == size1  # nothing appended on resume
