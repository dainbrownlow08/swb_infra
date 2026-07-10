from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable

from . import audio_index as audio_index_mod
from .manifest import (
    MANIFEST_HEADER,
    already_done_calls,
    manifest_path,
    open_appender,
    parse_rel_path,
    write_row,
)
from .slicer import _probe_duration, slice_utterance
from .transcripts import iter_transcript_paths, parse_transcript

DEFAULT_INDEX_CACHE = ".swb_extract_index.json"
DEFAULT_OUT_DIR = "utterances_v2"
DEFAULT_AUDIO_DIR = "audio"
DEFAULT_TRANSCRIPT_DIR = "swb_ms98_transcriptions_cleaned"


def _bucket(call_id: int) -> str:
    return f"{call_id // 10:03d}"


def _wav_relpath(call_id: int, side: str, utt_num: int) -> str:
    return f"{_bucket(call_id)}/sw{call_id:04d}{side}-U{utt_num:04d}.wav"


def _transcript_path(transcript_root: Path, call_id: int, side: str) -> Path:
    nn = call_id // 100
    return (
        transcript_root
        / f"{nn:02d}"
        / f"{call_id:04d}"
        / f"sw{call_id:04d}{side}-ms98-a-trans.text"
    )


def _discover_all_units(transcript_root: Path) -> list[tuple[int, str]]:
    units: list[tuple[int, str]] = []
    for p in iter_transcript_paths(transcript_root):
        name = p.name
        call_id = int(name[2:6])
        side = name[6]
        units.append((call_id, side))
    return sorted(set(units))


def _resolve_units(
    args: argparse.Namespace, transcript_root: Path
) -> list[tuple[int, str]]:
    if args.all:
        return _discover_all_units(transcript_root)
    if args.calls_file:
        ids = [
            int(line.strip())
            for line in Path(args.calls_file).read_text().splitlines()
            if line.strip()
        ]
        sides = (args.side,) if args.side else ("A", "B")
        return sorted({(c, s) for c in ids for s in sides})
    if args.call is not None:
        sides = (args.side,) if args.side else ("A", "B")
        return [(args.call, s) for s in sides]
    raise SystemExit("extract: must specify --all, --call, or --calls-file")


def cmd_index(args: argparse.Namespace) -> int:
    audio_root = Path(args.audio_root)
    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        idx = audio_index_mod.load_index(cache)
        print(f"loaded index from {cache} ({len(idx)} entries)")
        return 0
    idx = audio_index_mod.build_index(audio_root)
    audio_index_mod.save_index(idx, cache)
    print(f"built index at {cache} ({len(idx)} entries)")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    audio_root = Path(args.audio_root)
    transcript_root = Path(args.transcript_root)
    out_root = Path(args.out_root)
    cache = Path(args.cache)

    if cache.exists():
        idx = audio_index_mod.load_index(cache)
    else:
        print(f"building audio index (no cache at {cache})", file=sys.stderr)
        idx = audio_index_mod.build_index(audio_root)
        audio_index_mod.save_index(idx, cache)

    units = _resolve_units(args, transcript_root)
    if args.limit:
        units = units[: args.limit]

    manifest = manifest_path(out_root)
    skip = set() if args.no_resume else already_done_calls(manifest)
    if skip:
        print(f"resuming: {len(skip)} (call,side) units already in manifest")

    work = [u for u in units if u not in skip]
    if not work:
        print("nothing to do")
        return 0

    print(f"processing {len(work)} (call,side) units → {manifest}")
    total_slices = 0
    with open_appender(manifest) as writer:
        for call_id, side in work:
            tpath = _transcript_path(transcript_root, call_id, side)
            if not tpath.is_file():
                print(f"  skip {call_id}{side}: transcript missing ({tpath})", file=sys.stderr)
                continue
            try:
                src_wav = audio_index_mod.resolve(idx, call_id, side)
            except KeyError as e:
                print(f"  skip {call_id}{side}: {e}", file=sys.stderr)
                continue

            n = 0
            for u in parse_transcript(tpath):
                rel = _wav_relpath(u.call_id, u.side, u.utt_num)
                dst = out_root / rel
                slice_utterance(src_wav, dst, u.start, u.end, overwrite=args.overwrite)
                write_row(writer, rel, u.text)
                n += 1
            total_slices += n
            print(f"  {call_id}{side}: {n} utterances")
    print(f"done: {total_slices} slices written")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root)
    transcript_root = Path(args.transcript_root)
    manifest = manifest_path(out_root)
    if not manifest.exists():
        print(f"no manifest at {manifest}", file=sys.stderr)
        return 2

    import csv as _csv

    with open(manifest, encoding="utf-8", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            print(f"unexpected header: {header}", file=sys.stderr)
            return 2
        rows = [r for r in reader if r]

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.sample, len(rows)))

    print(f"verifying {len(sample)} of {len(rows)} rows")
    failures = 0
    for rel, _text in sample:
        try:
            call_id, side, utt_num = parse_rel_path(rel)
        except ValueError:
            print(f"  BAD path format: {rel}", file=sys.stderr)
            failures += 1
            continue

        wav = out_root / rel
        if not wav.exists():
            print(f"  MISSING wav: {wav}", file=sys.stderr)
            failures += 1
            continue

        tpath = _transcript_path(transcript_root, call_id, side)
        match = next(
            (u for u in parse_transcript(tpath) if u.utt_num == utt_num), None
        )
        if match is None:
            print(f"  no transcript line for {rel}", file=sys.stderr)
            failures += 1
            continue
        expected = match.end - match.start
        actual = _probe_duration(wav)
        if abs(actual - expected) > 0.005:
            print(
                f"  DUR mismatch {rel}: expected {expected:.4f}s actual {actual:.4f}s",
                file=sys.stderr,
            )
            failures += 1

    if failures:
        print(f"FAIL: {failures} mismatches", file=sys.stderr)
        return 1
    print("OK")
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    from .features_table import run as table_run

    return table_run(args)


def cmd_features(args: argparse.Namespace) -> int:
    from .features import (
        demographics,
        filler_word_per_second,
        filler_word_rate,
        fto,
        latching_flag,
        laughter,
        loudness,
        machine_gun_question,
        mutual_revelation_flag,
        overlap,
        overlap_split,
        personal_focus_score,
        pitch,
        pronoun_per_second,
        pronoun_rate,
        question_flags,
        repetition_per_second,
        repetition_rate,
        repetitions_in_current,
        repetitions_in_previous,
        rising_terminal,
        syllable_rate,
        token_count,
        topic_label,
        turn_gap,
        within_utterance_pauses,
        word_rate,
    )

    if args.name == demographics.FEATURE_NAME:
        return demographics.run(args)
    if args.name == filler_word_rate.FEATURE_NAME:
        return filler_word_rate.run(args)
    if args.name == filler_word_per_second.FEATURE_NAME:
        return filler_word_per_second.run(args)
    if args.name == pitch.FEATURE_NAME:
        return pitch.run(args)
    if args.name == loudness.FEATURE_NAME:
        return loudness.run(args)
    if args.name == pronoun_rate.FEATURE_NAME:
        return pronoun_rate.run(args)
    if args.name == pronoun_per_second.FEATURE_NAME:
        return pronoun_per_second.run(args)
    if args.name == repetition_rate.FEATURE_NAME:
        return repetition_rate.run(args)
    if args.name == repetition_per_second.FEATURE_NAME:
        return repetition_per_second.run(args)
    if args.name == syllable_rate.FEATURE_NAME:
        return syllable_rate.run(args)
    if args.name == turn_gap.FEATURE_NAME:
        return turn_gap.run(args)
    if args.name == fto.FEATURE_NAME:
        return fto.run(args)
    if args.name == repetitions_in_current.FEATURE_NAME:
        return repetitions_in_current.run(args)
    if args.name == repetitions_in_previous.FEATURE_NAME:
        return repetitions_in_previous.run(args)
    if args.name == word_rate.FEATURE_NAME:
        return word_rate.run(args)
    if args.name == token_count.FEATURE_NAME:
        return token_count.run(args)
    if args.name == topic_label.FEATURE_NAME:
        return topic_label.run(args)
    if args.name == personal_focus_score.FEATURE_NAME:
        return personal_focus_score.run(args)
    if args.name == mutual_revelation_flag.FEATURE_NAME:
        return mutual_revelation_flag.run(args)
    if args.name == latching_flag.FEATURE_NAME:
        return latching_flag.run(args)
    if args.name == laughter.FEATURE_NAME:
        return laughter.run(args)
    if args.name == within_utterance_pauses.FEATURE_NAME:
        return within_utterance_pauses.run(args)
    if args.name == overlap.FEATURE_NAME:
        return overlap.run(args)
    if args.name == overlap_split.FEATURE_NAME:
        return overlap_split.run(args)
    if args.name == question_flags.FEATURE_NAME:
        return question_flags.run(args)
    if args.name == rising_terminal.FEATURE_NAME:
        return rising_terminal.run(args)
    if args.name == machine_gun_question.FEATURE_NAME:
        return machine_gun_question.run(args)
    raise NotImplementedError(
        f"feature extractor {args.name!r} not implemented yet. "
        f"Write to {Path(args.out_root) / 'features' / (args.name + '.csv')} "
        f"with header 'Utterance File Name,<col1>[,<col2>...]' "
        f"keyed on the same paths as manifest.csv."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swb-extract")
    p.add_argument("--repo-root", default=str(Path.cwd()),
                   help="repo root (used to resolve default paths)")
    sub = p.add_subparsers(dest="cmd", required=True)

    repo = Path.cwd()
    default_audio = str(repo / DEFAULT_AUDIO_DIR)
    default_trans = str(repo / DEFAULT_TRANSCRIPT_DIR)
    default_out = str(repo / DEFAULT_OUT_DIR)
    default_cache = str(repo / DEFAULT_INDEX_CACHE)

    s = sub.add_parser("index", help="build/cache the audio index")
    s.add_argument("--audio-root", default=default_audio)
    s.add_argument("--cache", default=default_cache)
    s.add_argument("--rebuild", action="store_true")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("extract", help="slice utterances and append to manifest")
    s.add_argument("--all", action="store_true")
    s.add_argument("--call", type=int)
    s.add_argument("--side", choices=["A", "B"])
    s.add_argument("--calls-file")
    s.add_argument("--audio-root", default=default_audio)
    s.add_argument("--transcript-root", default=default_trans)
    s.add_argument("--out-root", default=default_out)
    s.add_argument("--cache", default=default_cache)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser(
        "table", help="rebuild features_table.csv from manifest + features/*.csv"
    )
    s.add_argument("--out-root", default=default_out)
    s.set_defaults(func=cmd_table)

    s = sub.add_parser("verify", help="spot-check durations of N random manifest rows")
    s.add_argument("--out-root", default=default_out)
    s.add_argument("--transcript-root", default=default_trans)
    s.add_argument("--sample", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_verify)

    default_caller = str(repo / "tables" / "caller_tab.csv")
    default_conv = str(repo / "tables" / "conv_tab.csv")
    default_topic = str(repo / "tables" / "topic_tab.csv")
    s = sub.add_parser("features", help="run a feature extractor that writes a sibling CSV")
    s.add_argument("name", help="extractor name (e.g. 'demographics')")
    s.add_argument("--out-root", default=default_out)
    s.add_argument("--caller-tab", default=default_caller,
                   help="LDC caller_tab.csv (used by demographics)")
    s.add_argument("--conv-tab", default=default_conv,
                   help="LDC conv_tab.csv (used by demographics, topic_label)")
    s.add_argument("--topic-tab", default=default_topic,
                   help="LDC topic_tab.csv (used by topic_label)")
    s.add_argument("--transcript-root", default=default_trans,
                   help="cleaned ms98 transcript root (used by per-second features)")
    s.add_argument("--workers", type=int, default=4,
                   help="parallel workers (used by pitch)")
    s.add_argument("--limit", type=int, default=0,
                   help="process at most N rows (used by pitch; 0=all)")
    s.add_argument("--overwrite", action="store_true",
                   help="ignore existing feature CSV (used by pitch)")
    s.set_defaults(func=cmd_features)

    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
