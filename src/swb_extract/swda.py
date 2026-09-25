"""Switchboard Dialog Act corpus (SwDA) access — gold acts joined to ms98 utterances by text.

Potts' release (``corpus/swda/swda/sw??utt/sw_<n>_<conv>.utt.csv``, 1,155 conversations):
one row per slash-unit act with ``act_tag`` (SWBD-DAMSL, same tag grammar as NXT's
``swbdType`` — always go through ``nxt.parse_tag``), ``caller`` A/B, and ``text`` in
Treebank transcription markup, with NO timestamps. NXT's 642 dialog-act conversations are
a subset of SwDA; the other 513 carry the same labels and are joined to ms98 utterances
here by WORD-SEQUENCE alignment instead of by time:

1. normalise both transcriptions to the same token form (``normalize_swda_text``,
   ``normalize_ms98_token``): SwDA markup dropped ({F um, } {D so, } [ reparandum +
   repair ] / -- <laughter> <<aside>> punctuation case), ms98 partial-word and variant
   markup dropped (an[y]- → an, because_1 → because), laughed words unwrapped;
2. lay each side's SwDA tokens in transcript order, each carrying its act index, and
   its ms98 tokens in utterance order, each carrying its utterance number;
3. ``difflib.SequenceMatcher`` (stdlib; built for two nearly identical sequences, which
   two transcriptions of the same speech are) → matching blocks → every matched ms98
   word inherits an act; an utterance's acts are the acts on its words in order, and its
   TERMINAL act is the act on its last matched word — the same rule the NXT time-based
   join applies (latest-starting act assigned to the utterance).

A side whose match ratio (matched ms98 tokens / all ms98 tokens) is below
``MIN_MATCH_RATIO`` is blanked rather than guessed. The method's own accuracy is measured
on the 642 conversations that have both joins (``analysis/validate_gold_acts.py``).
"""
from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SWDA_ROOT = REPO_ROOT / "corpus" / "swda" / "swda"

MIN_MATCH_RATIO = 0.85

_FILE_RE = re.compile(r"^sw_\d+_(\d{4})\.utt\.csv$")
_SWDA_MARKUP = [
    (re.compile(r"<<[^>]*>>"), " "),          # <<talking off-line>>
    (re.compile(r"<[^>]*>"), " "),            # <laughter>, <noise>
    (re.compile(r"\{[A-Z]\s"), " "),          # {F um, } {D so, } {C and } {E ...} {A ...}
    (re.compile(r"[{}\[\]+/#]"), " "),        # restart brackets, slash-unit ends, overlap marks
    (re.compile(r"\(\(|\)\)"), " "),          # ((unclear)) — keep the guessed words
    (re.compile(r"--"), " "),
]
_TOKEN_RE = re.compile(r"[a-z0-9']+(?:-[a-z0-9']+)*")
_MS98_PARTIAL_RE = re.compile(r"\[[^\]]*\]")   # an[y]- → an-
_MS98_VARIANT_RE = re.compile(r"_\d+$")        # because_1 → because


def list_conversations(root: Path = SWDA_ROOT) -> list[int]:
    out = set()
    for p in root.glob("sw*utt/sw_*.utt.csv"):
        m = _FILE_RE.match(p.name)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def _conv_file(conv: int, root: Path) -> Path | None:
    hits = list(root.glob(f"sw*utt/sw_*_{conv:04d}.utt.csv"))
    return hits[0] if hits else None


def load_acts(conv: int, root: Path = SWDA_ROOT) -> dict[str, list[tuple[str, str]]]:
    """Per side ('A'/'B'): [(act_tag, raw text)] in transcript order; {} if absent."""
    path = _conv_file(conv, root)
    if path is None:
        return {}
    out: dict[str, list[tuple[str, str]]] = {"A": [], "B": []}
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["transcript_index"]))
    for r in rows:
        side = r["caller"].strip().upper()
        if side in out:
            out[side].append((r["act_tag"].strip(), r["text"]))
    return out


def normalize_swda_text(text: str) -> list[str]:
    t = text
    for rx, rep in _SWDA_MARKUP:
        t = rx.sub(rep, t)
    return [tok.strip("'-") for tok in _TOKEN_RE.findall(t.lower()) if tok.strip("'-")]


def normalize_ms98_token(token: str) -> str | None:
    """ms98 token (already lowercased / bracket-stripped by _text.tokenize) → alignment form."""
    t = _MS98_VARIANT_RE.sub("", _MS98_PARTIAL_RE.sub("", token)).strip("'-")
    return t or None


def align_side(
    swda_tokens: list[tuple[str, int]],
    ms98_tokens: list[tuple[str, int]],
) -> tuple[dict[int, list[int]], float]:
    """Align (token, act_idx) against (token, utt_num).

    Returns ({utt_num: [act_idx in token order, consecutive duplicates collapsed]},
    match ratio = matched ms98 tokens / all ms98 tokens).
    """
    if not swda_tokens or not ms98_tokens:
        return {}, 0.0
    a = [t for t, _ in swda_tokens]
    b = [t for t, _ in ms98_tokens]
    sm = SequenceMatcher(None, a, b, autojunk=False)
    acts_by_utt: dict[int, list[int]] = {}
    matched = 0
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            act = swda_tokens[i + k][1]
            utt = ms98_tokens[j + k][1]
            lst = acts_by_utt.setdefault(utt, [])
            if not lst or lst[-1] != act:
                lst.append(act)
        matched += n
    return acts_by_utt, matched / len(b)
