"""Mutual Revelation flag — both speakers exchange personal anecdotes.

Tannen reference: Ch.4 "Mutual Revelation" (PDF p. 122) — "a personal
statement is intended as a show of rapport. By this strategy, the speaker
expects a statement of personal experience to elicit a similar statement
from the other." Also Ch.7 device #2 (PDF p. 203).

Two-stage detection:

1. is_personal_anecdote(text):
     - At least MIN_TOKENS post-bracket-strip tokens (excludes fragments)
     - Matches FIRST_PERSON_PATTERN (i, my, me, we, our, ...)
     - Has at least one spaCy VBD or VBN tagged token (past-tense verb or
       past participle, covering both simple past and present perfect)

   The cheap regex-based checks short-circuit before spaCy runs, so spaCy
   is only invoked on candidates that already have a first-person pronoun
   and enough tokens.

2. mutual_revelation_flag for current utterance:
     - First of conversation → empty cell (no previous turn)
     - Same speaker as previous turn → 0 (not "mutual")
     - Different speaker AND both turns are personal anecdotes → 1
     - Otherwise → 0

The cross-utterance "previous" comes from the same chronological merge used
by Turn Gap (turn_gap.build_turn_gap_index), so this is consistent with the
rest of the pipeline.

Output: utterances_v2/features/mutual_revelation_flag.csv
Header: Utterance File Name,Mutual Revelation Flag
"""
from __future__ import annotations

import csv
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from .turn_gap import (
    TextIndex,
    TurnGapIndex,
    build_text_index,
    build_turn_gap_index,
)

FEATURE_NAME = "mutual_revelation_flag"
HEADER = ("Utterance File Name", "Mutual Revelation Flag")
SPACY_MODEL = "en_core_web_sm"

FIRST_PERSON_PATTERN = re.compile(
    r"\b(i|my|me|mine|myself|we|our|us|ours|ourselves)\b",
    re.IGNORECASE,
)

# Past-tense verb tags in the Penn Treebank scheme spaCy emits via tok.tag_:
#   VBD = past tense ("went", "made", "was")
#   VBN = past participle ("been", "gone", "made", "worked")
# VBN catches present-perfect narratives like "i've been working".
PAST_TENSE_TAGS = frozenset({"VBD", "VBN"})

MIN_TOKENS = 5

# Lazy module-level cache so each ProcessPoolExecutor worker loads spaCy once.
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(SPACY_MODEL)
    return _NLP


def strip_bracket_tokens(text: str) -> str:
    return " ".join(
        t for t in text.split()
        if not (t.startswith("[") and t.endswith("]"))
    )


def is_personal_anecdote(text: str, nlp=None) -> bool:
    """Detect first-person past-tense personal narrative content via spaCy.

    Cheap checks (token count, first-person regex) short-circuit before
    spaCy is invoked.
    """
    cleaned = strip_bracket_tokens(text)
    if len(cleaned.split()) < MIN_TOKENS:
        return False
    if not FIRST_PERSON_PATTERN.search(cleaned):
        return False
    if nlp is None:
        nlp = _get_nlp()
    doc = nlp(cleaned)
    return any(tok.tag_ in PAST_TENSE_TAGS for tok in doc)


# --- Batch processing helpers ---


def _anecdote_worker(
    item: tuple[tuple[int, str, int], str],
) -> tuple[tuple[int, str, int], bool]:
    key, text = item
    return key, is_personal_anecdote(text)


def build_anecdote_index(
    text_idx: TextIndex,
    workers: int = 4,
) -> dict[tuple[int, str, int], bool]:
    """Pre-compute is_personal_anecdote for every utterance text.

    Each text is parsed by spaCy at most once. With workers > 1, each worker
    process loads spaCy independently via the module-level lazy cache.
    """
    items = list(text_idx.items())
    print(
        f"  computing anecdote flags for {len(items):,} utterance texts "
        f"(workers={workers})"
    )
    out: dict[tuple[int, str, int], bool] = {}
    if workers <= 1:
        for key, text in items:
            out[key] = is_personal_anecdote(text)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = 0
            last_log = 0
            for key, is_anec in ex.map(_anecdote_worker, items, chunksize=200):
                out[key] = is_anec
                done += 1
                if done - last_log >= 10000 or done == len(items):
                    print(f"    {done:,}/{len(items):,}", flush=True)
                    last_log = done
    return out


# --- Cross-utterance lookup ---


def lookup_mutual_revelation(
    merged_idx: TurnGapIndex,
    text_idx: TextIndex,
    rel_path: str,
) -> int | None:
    """1 if both this turn and the previous turn (different speaker) are
    personal anecdotes; 0 otherwise; None for first-of-conversation.

    Computes is_personal_anecdote on demand — used by tests and ad-hoc
    lookups. Batch processing in write_mutual_revelation_flags() uses a
    pre-computed anecdote index instead.
    """
    call_id, side, utt_num = parse_rel_path(rel_path)
    merged = merged_idx.get(call_id)
    if not merged:
        return None
    cur_pos: int | None = None
    for i, e in enumerate(merged):
        if e[0] == side and e[1] == utt_num:
            cur_pos = i
            break
    if cur_pos is None or cur_pos == 0:
        return None
    prev_side, prev_utt, _, _ = merged[cur_pos - 1]
    if prev_side == side:
        return 0
    cur_text = text_idx.get((call_id, side, utt_num))
    prev_text = text_idx.get((call_id, prev_side, prev_utt))
    if cur_text is None or prev_text is None:
        return None
    return int(is_personal_anecdote(cur_text) and is_personal_anecdote(prev_text))


def _lookup_from_index(
    merged_idx: TurnGapIndex,
    anecdote_idx: dict[tuple[int, str, int], bool],
    rel_path: str,
) -> int | None:
    """Same as lookup_mutual_revelation but reads from a pre-computed
    anecdote index — avoids re-parsing texts during the manifest scan."""
    call_id, side, utt_num = parse_rel_path(rel_path)
    merged = merged_idx.get(call_id)
    if not merged:
        return None
    cur_pos: int | None = None
    for i, e in enumerate(merged):
        if e[0] == side and e[1] == utt_num:
            cur_pos = i
            break
    if cur_pos is None or cur_pos == 0:
        return None
    prev_side, prev_utt, _, _ = merged[cur_pos - 1]
    if prev_side == side:
        return 0
    cur_anec = anecdote_idx.get((call_id, side, utt_num))
    prev_anec = anecdote_idx.get((call_id, prev_side, prev_utt))
    if cur_anec is None or prev_anec is None:
        return None
    return int(cur_anec and prev_anec)


def _fmt(v: int | None) -> str:
    return "" if v is None else str(v)


def _is_first_of_conversation(idx: TurnGapIndex, rel_path: str) -> bool:
    try:
        call_id, side, utt_num = parse_rel_path(rel_path)
    except ValueError:
        return False
    merged = idx.get(call_id)
    if not merged:
        return False
    return merged[0][0] == side and merged[0][1] == utt_num


def _collect_needed_keys(
    manifest_csv: Path,
    merged_idx: TurnGapIndex,
) -> tuple[list[str], set[tuple[int, str, int]]]:
    """Read the manifest and return (rel_paths, set of (call_id, side, utt_num)
    keys whose anecdote-flag we need — current row plus its predecessor)."""
    rels: list[str] = []
    needed: set[tuple[int, str, int]] = set()
    with open(manifest_csv, encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest_csv}: {header!r}"
            )
        for row in reader:
            if not row:
                continue
            rel = row[0]
            rels.append(rel)
            try:
                cid, side, utt = parse_rel_path(rel)
            except ValueError:
                continue
            needed.add((cid, side, utt))
            merged = merged_idx.get(cid)
            if not merged:
                continue
            for i, e in enumerate(merged):
                if e[0] == side and e[1] == utt:
                    if i > 0:
                        prev = merged[i - 1]
                        needed.add((cid, prev[0], prev[1]))
                    break
    return rels, needed


def write_mutual_revelation_flags(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
    workers: int = 4,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_idx = build_turn_gap_index(transcript_root)
    text_idx = build_text_index(transcript_root)

    # Only compute anecdote flags for utterances actually referenced by the
    # manifest (current + immediate predecessor). For full-manifest runs this
    # is essentially the entire text_idx; for unit tests with tiny manifests
    # it avoids parsing tens of thousands of irrelevant texts.
    rels, needed_keys = _collect_needed_keys(manifest_csv, merged_idx)
    needed_text_idx: TextIndex = {
        k: text_idx[k] for k in needed_keys if k in text_idx
    }
    anecdote_idx = build_anecdote_index(needed_text_idx, workers=workers)
    n_anec = sum(1 for v in anecdote_idx.values() if v)
    print(f"  anecdotes detected: {n_anec:,}/{len(anecdote_idx):,}")

    n = 0
    n_first = 0
    n_missing = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for rel in rels:
            v = _lookup_from_index(merged_idx, anecdote_idx, rel)
            if v is None:
                if _is_first_of_conversation(merged_idx, rel):
                    n_first += 1
                else:
                    n_missing += 1
            writer.writerow([rel, _fmt(v)])
            n += 1
    if n_first or n_missing:
        print(
            f"  empty cells: first-of-conversation={n_first}, missing={n_missing}"
        )
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_mutual_revelation_flags(
        manifest_path(out_root),
        out_root / "features" / "mutual_revelation_flag.csv",
        transcript_root=Path(args.transcript_root),
        workers=getattr(args, "workers", 4),
    )
    print(f"wrote {n} mutual revelation flag rows")
    return 0
