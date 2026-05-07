"""Pronoun Rate per utterance, computed via spaCy POS tagging.

Same algorithm as the legacy paper (FEPronounRate.py), with one correction:
whole-bracket tokens (`[noise]`, `[laughter]`, `[vocalized-noise]`, etc.) are
stripped before spaCy sees them. They are not real spoken words and would
otherwise inflate the denominator of the rate. This is consistent with our
filler_word_rate and repetition_rate features — all three text-based rate
features treat bracket tokens uniformly as non-words.

  text = drop tokens where text.startswith('[') and text.endswith(']')
  doc = nlp(text)
  rate = (# tokens with pos_ == 'PRON') / (total spaCy token count)

User decisions for this rewrite:
- Strip whole-bracket tokens before feeding text to spaCy.
- Use en_core_web_sm (matches legacy).
- Empty / single-non-pronoun transcript → 0.0 (avoids divide-by-zero crash).

Output: utterances_v2/features/pronoun_rate.csv
Header: Utterance File Name,Pronoun Rate
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path

FEATURE_NAME = "pronoun_rate"
HEADER = ("Utterance File Name", "Pronoun Rate")
SPACY_MODEL = "en_core_web_sm"

# Lazy module-level cache so each ProcessPoolExecutor worker loads spaCy once.
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(SPACY_MODEL)
    return _NLP


def strip_bracket_tokens(text: str) -> str:
    """Remove whole-bracket tokens like [noise], [laughter] before tokenization.

    A whole-bracket token starts with '[' and ends with ']'. Inline markers
    like 'i[t]-' (partial words) do NOT start with '[' so they are kept.
    """
    return " ".join(
        t for t in text.split()
        if not (t.startswith("[") and t.endswith("]"))
    )


def compute_rate(text: str, nlp=None) -> float:
    """Pronoun count / total token count, per spaCy POS tagging.

    Whole-bracket tokens are stripped before spaCy tokenization. Returns 0.0
    for empty input to avoid divide-by-zero (legacy would crash).
    """
    if nlp is None:
        nlp = _get_nlp()
    cleaned = strip_bracket_tokens(text)
    doc = nlp(cleaned)
    if len(doc) == 0:
        return 0.0
    pronouns = sum(1 for tok in doc if tok.pos_ == "PRON")
    return pronouns / len(doc)


def _worker(arg: tuple[str, str]) -> tuple[str, float]:
    rel, text = arg
    return rel, compute_rate(text)


def _read_existing(output_csv: Path) -> dict[str, list[str]]:
    if not output_csv.exists():
        return {}
    with open(output_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != HEADER:
            return {}
        return {row[0]: row for row in reader if row}


def write_pronoun_rates(
    manifest_csv: Path,
    output_csv: Path,
    workers: int = 4,
    limit: int = 0,
    overwrite: bool = False,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest_csv}: {header!r}"
            )
        rows = [(row[0], row[1]) for row in reader if row]
    if limit:
        rows = rows[:limit]

    cache: dict[str, list[str]] = {} if overwrite else _read_existing(output_csv)
    needs_work = [(rel, text) for rel, text in rows if rel not in cache]
    print(
        f"pronoun_rate: {len(rows):,} total, {len(cache):,} cached, "
        f"{len(needs_work):,} to extract (workers={workers})"
    )

    fresh: dict[str, float] = {}
    if needs_work:
        if workers <= 1:
            for arg in needs_work:
                rel, rate = _worker(arg)
                fresh[rel] = rate
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                done = 0
                last_log = 0
                for rel, rate in ex.map(_worker, needs_work, chunksize=200):
                    fresh[rel] = rate
                    done += 1
                    if done - last_log >= 5000 or done == len(needs_work):
                        print(f"  {done:,}/{len(needs_work):,}", flush=True)
                        last_log = done

    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for rel, _ in rows:
            if rel in cache:
                writer.writerow(cache[rel])
            else:
                writer.writerow([rel, fresh[rel]])
    return len(rows)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_pronoun_rates(
        manifest_path(out_root),
        out_root / "features" / "pronoun_rate.csv",
        workers=args.workers,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"wrote {n} pronoun rate rows")
    return 0
