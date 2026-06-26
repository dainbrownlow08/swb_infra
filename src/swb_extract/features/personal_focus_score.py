"""Personal focus per utterance, via Empath lexical categories — counts first.

Tannen reference: Ch.7 dim 1 "Relative personal focus of topic" (PDF p. 202);
Ch.4 "Personal vs. Impersonal Topics" (PDF pp. 90-103).

Why not pronoun ratio: counting personal pronouns measures HOW the speaker
frames talk (self-referential vs. not), not WHAT the talk is about.
  "I think the government should invest more"  → personal pronoun, impersonal topic
  "Mom always made dinner before dad got home" → no pronouns, personal topic
Tannen's Dim 1 is about content/topic, not surface lexical framing.

Redesigned per AUDIT.md §3 fix 3. The original emitted only the per-utterance
ratio personal/(personal+impersonal), which on short utterances was undefined
(54.7% empty) or saturated at exactly 0.0/1.0 (64.6% of defined values) — one
or two category hits decide the whole ratio. The fix is to emit the raw
ingredients so ratios can be pooled at whatever level the analysis needs
(ratio of sums per speaker/conversation/topic — NOT mean of per-utterance
ratios), plus a hit-gated utterance-level score:

  Personal Hits        — raw Empath hit count over PERSONAL categories
  Impersonal Hits      — raw hit count over IMPERSONAL categories
  Analyzed Tokens      — whitespace tokens after bracket stripping (density basis)
  Personal Focus Score — personal/(personal+impersonal), only when
                         personal+impersonal >= MIN_HITS_FOR_SCORE; else empty.

Counts are always defined (0 is an honest measurement); only the score cell
can be empty, and emptiness now means "too little topical content to judge",
gated by a documented, sweepable constant.

Categorization derived from Tannen's framework over the 194 Empath built-in
categories: 66 PERSONAL (family, body, feelings, domestic), 41 IMPERSONAL
(institutions, abstract, economics, tech, politics), 87 UNDECIDED (sensory,
functional, context-dependent — deliberately excluded so they don't bias
either direction).

Output: utterances_v2/features/personal_focus_score.csv
Header: Utterance File Name,Personal Hits,Impersonal Hits,Analyzed Tokens,Personal Focus Score
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path

FEATURE_NAME = "personal_focus_score"
HEADER = (
    "Utterance File Name",
    "Personal Hits",
    "Impersonal Hits",
    "Analyzed Tokens",
    "Personal Focus Score",
)

# Minimum combined hits for the per-utterance ratio to be reported.
MIN_HITS_FOR_SCORE = 3

# 66 categories indexing personal life, body, family, feelings, domestic activity
PERSONAL_CATEGORIES = frozenset({
    "achievement", "affection", "alcohol", "anger", "appearance", "attractive",
    "beauty", "body", "celebration", "cheerfulness", "childish", "children",
    "cleaning", "clothing", "contentment", "cooking", "dance", "death",
    "disappointment", "domestic_work", "eating", "emotional", "envy",
    "exasperation", "exercise", "family", "fashion", "fear", "friends", "fun",
    "giving", "hate", "healing", "health", "home", "horror", "hygiene",
    "injury", "irritability", "joy", "leisure", "love", "lust",
    "medical_emergency", "negative_emotion", "neglect", "nervousness", "pain",
    "party", "pet", "positive_emotion", "pride", "rage", "sadness", "sexual",
    "shame", "sleep", "suffering", "surprise", "sympathy", "timidity",
    "torment", "vacation", "wedding", "youth", "zest",
})

# 41 categories indexing institutions, abstract concepts, economics, tech, politics
IMPERSONAL_CATEGORIES = frozenset({
    "ancient", "banking", "blue_collar_job", "business", "computer", "crime",
    "dominant_heirarchical", "economics", "government", "internet",
    "journalism", "law", "legend", "medieval", "military", "money",
    "occupation", "office", "payment", "philosophy", "politics", "poor",
    "power", "prison", "programming", "real_estate", "royalty", "science",
    "social_media", "stealing", "technology", "terrorism", "urban",
    "valuable", "war", "wealthy", "weapon", "white_collar_job", "work",
    "worship", "writing",
})

# 87 Empath categories deliberately omitted — sensory, functional, or
# context-dependent (e.g. weather, music, sports, religion, school).
# They contribute zero to numerator and zero to denominator.

ALL_TANNEN_CATEGORIES = PERSONAL_CATEGORIES | IMPERSONAL_CATEGORIES  # 107 categories

_LEXICON = None
_CATEGORIES_VERIFIED = False

# (personal_hits, impersonal_hits, analyzed_tokens, score-or-None)
Counts = tuple[int, int, int, float | None]


def _get_lexicon():
    """Lazy-load Empath lexicon. Each ProcessPoolExecutor worker loads it once.

    On first load, asserts that all categories in ALL_TANNEN_CATEGORIES exist
    in the installed Empath version. Fails loudly if any category drifts.
    """
    global _LEXICON, _CATEGORIES_VERIFIED
    if _LEXICON is None:
        from empath import Empath
        _LEXICON = Empath()
    if not _CATEGORIES_VERIFIED:
        missing = ALL_TANNEN_CATEGORIES - set(_LEXICON.cats.keys())
        if missing:
            raise RuntimeError(
                f"Empath categories not present in installed version: "
                f"{sorted(missing)}"
            )
        _CATEGORIES_VERIFIED = True
    return _LEXICON


def strip_bracket_tokens(text: str) -> str:
    """Remove whole-bracket tokens like [noise], [laughter] before analysis.

    A whole-bracket token starts with '[' and ends with ']'. Inline markers
    like 'i[t]-' (partial words) do NOT start with '[' so they are kept.
    """
    return " ".join(
        t for t in text.split()
        if not (t.startswith("[") and t.endswith("]"))
    )


def compute_counts(text: str, lexicon=None) -> Counts:
    """Raw Empath hit counts plus the gated ratio.

    Counts are always defined (0 for no hits). The score is None (→ empty
    cell) when personal+impersonal hits < MIN_HITS_FOR_SCORE — backchannels,
    pure fillers, or utterances with too little topical content to judge.
    """
    if lexicon is None:
        lexicon = _get_lexicon()
    cleaned = strip_bracket_tokens(text)
    tokens = len(cleaned.split())
    if not cleaned:
        return 0, 0, 0, None
    result = lexicon.analyze(
        cleaned, categories=list(ALL_TANNEN_CATEGORIES), normalize=False
    ) or {}
    personal = int(round(sum(result.get(c, 0.0) for c in PERSONAL_CATEGORIES)))
    impersonal = int(round(sum(result.get(c, 0.0) for c in IMPERSONAL_CATEGORIES)))
    total = personal + impersonal
    score = personal / total if total >= MIN_HITS_FOR_SCORE else None
    return personal, impersonal, tokens, score


def _worker(arg: tuple[str, str]) -> tuple[str, Counts]:
    rel, text = arg
    return rel, compute_counts(text)


def _fmt_row(rel: str, counts: Counts) -> list[str]:
    personal, impersonal, tokens, score = counts
    return [
        rel,
        str(personal),
        str(impersonal),
        str(tokens),
        "" if score is None else repr(score),
    ]


def _read_existing(output_csv: Path) -> dict[str, list[str]]:
    if not output_csv.exists():
        return {}
    with open(output_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if tuple(header or ()) != HEADER:
            return {}
        return {row[0]: row for row in reader if row}


def write_personal_focus_scores(
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
        f"personal_focus_score: {len(rows):,} total, {len(cache):,} cached, "
        f"{len(needs_work):,} to extract (workers={workers})"
    )

    fresh: dict[str, Counts] = {}
    if needs_work:
        if workers <= 1:
            for arg in needs_work:
                rel, counts = _worker(arg)
                fresh[rel] = counts
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                done = 0
                last_log = 0
                for rel, counts in ex.map(_worker, needs_work, chunksize=200):
                    fresh[rel] = counts
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
                writer.writerow(_fmt_row(rel, fresh[rel]))
    return len(rows)


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_personal_focus_scores(
        manifest_path(out_root),
        out_root / "features" / "personal_focus_score.csv",
        workers=args.workers,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"wrote {n} personal focus score rows")
    return 0
