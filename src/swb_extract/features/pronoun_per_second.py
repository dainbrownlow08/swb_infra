"""Pronouns per Second per utterance.

Same algorithm as pronoun_rate (spaCy POS tagging with whole-bracket tokens
stripped before tokenization), but divides by utterance duration in seconds
instead of token count.

Output: utterances_v2/features/pronoun_per_second.csv
Header: Utterance File Name,Pronouns per Second
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path
from ._duration_lookup import build_duration_index, lookup_duration
from .pronoun_rate import _get_nlp, strip_bracket_tokens

FEATURE_NAME = "pronoun_per_second"
HEADER = ("Utterance File Name", "Pronouns per Second")


def count_pronouns(text: str, nlp=None) -> int:
    if nlp is None:
        nlp = _get_nlp()
    cleaned = strip_bracket_tokens(text)
    if not cleaned:
        return 0
    doc = nlp(cleaned)
    return sum(1 for tok in doc if tok.pos_ == "PRON")


def compute_rate_per_second(
    text: str, duration: float | None, nlp=None
) -> float | None:
    if duration is None or duration <= 0:
        return None
    return count_pronouns(text, nlp) / duration


def _worker(arg: tuple[str, str, float | None]) -> tuple[str, float | None]:
    rel, text, duration = arg
    return rel, compute_rate_per_second(text, duration)


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def write_pronouns_per_second(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
    workers: int = 4,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    durations = build_duration_index(transcript_root)

    with open(manifest_csv, encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest_csv}: {header!r}"
            )
        rows = [(row[0], row[1]) for row in reader if row]

    work = [(rel, text, lookup_duration(durations, rel)) for rel, text in rows]

    results: dict[str, float | None] = {}
    if workers <= 1:
        for arg in work:
            rel, rate = _worker(arg)
            results[rel] = rate
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = 0
            last_log = 0
            for rel, rate in ex.map(_worker, work, chunksize=200):
                results[rel] = rate
                done += 1
                if done - last_log >= 5000 or done == len(work):
                    print(f"  {done:,}/{len(work):,}", flush=True)
                    last_log = done

    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for rel, _text in rows:
            writer.writerow([rel, _fmt(results[rel])])
    return len(rows)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_pronouns_per_second(
        manifest_path(out_root),
        out_root / "features" / "pronoun_per_second.csv",
        transcript_root=Path(args.transcript_root),
        workers=args.workers,
    )
    print(f"wrote {n} pronouns-per-second rows")
    return 0
