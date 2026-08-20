"""Backchannel allowlist vs laughed-word / bracket tokens — gold adjudication.

Question (2026-08-19 interactional-extractor walkthrough, audit §3 6b family):
`is_backchannel` strips only leading/trailing punctuation, so

  - class A: a laughed backchannel token `[laughter-yeah]` normalizes to
    "laughter-yeah" (not in the allowlist) — the utterance classifies
    SUBSTANTIVE even though every spoken word is a backchannel token
    (83 utterances corpus-wide);
  - class B: a non-speech event token beside backchannel words
    ("yeah [laughter]", "[noise] uh-huh") likewise forces SUBSTANTIVE
    (2,153 utterances corpus-wide, 3.4% of the current bc population).

Class B was a documented deliberate choice ("continuity with the notebooks",
backchannels.py docstring) — but it predates the Step 12 gold validation and
the 6b laughed-word adjudication. Both classes matter beyond the flag itself:
`fto.build_turn_events` treats non-bc other-speaker utterances as substantive,
so misclassified backchannels can BREAK TURNS (spurious transfers → FTO,
latching, overlap_split all inherit), biased against high-laughter =
involvement-style speakers (the 6b error shape).

Adjudication rule (Question Flag / rising-terminal precedent): reproduce the
Step 12 baseline exactly, then adopt a variant only if it does not lose
precision and improves recall/F1 vs gold primary {b}.

Variants:
  v0  — shipped `is_backchannel` (baseline; Step 12 recorded P .842 R .917).
  vA  — unwrap [laughter-X] → X first (the 6b tokenization), then the same
        all-tokens rule.
  vAB — full shared `_text._unwrap_or_drop` policy first: unwrap laughed
        words, DROP [bracket] events and <angle> markup, then the rule
        (an utterance left with zero tokens stays non-bc).

Gold and alignment exactly as NB07 Step 12 (pre-registered): each timed DA →
same-side ms98 utterance with maximal temporal overlap, accepted iff overlap
≥ 50% of the shorter span; gold-bc iff the utterance matched ≥1 DA and ALL
matched DAs have base sets ⊆ {b} (wide variant {b, bh, bk}); P/R over
gold-labelled utterances only.
"""
import string
import sys
from pathlib import Path

sys.path.insert(0, "/Users/dainbrownlow/switchboard/src")
import pandas as pd

from swb_extract import nxt
from swb_extract.features._text import _unwrap_or_drop, _LAUGHED_WORD_RE
from swb_extract.features.backchannels import BACKCHANNEL_TOKENS, is_backchannel
from swb_extract.transcripts import parse_transcript

TRANS_ROOT = Path("/Users/dainbrownlow/switchboard/swb_ms98_transcriptions_cleaned")

BC_PRIMARY = {"b"}
BC_WIDE = {"b", "bh", "bk"}


def _all_bc(tokens: list[str]) -> bool:
    toks = [w.strip(string.punctuation).lower() for w in tokens]
    toks = [w for w in toks if w]
    return len(toks) > 0 and all(w in BACKCHANNEL_TOKENS for w in toks)


def v0(text: str) -> bool:
    return is_backchannel(text)


def vA(text: str) -> bool:
    toks = [
        _LAUGHED_WORD_RE.match(w).group(1) if _LAUGHED_WORD_RE.match(w) else w
        for w in str(text).split()
    ]
    return _all_bc(toks)


def vAB(text: str) -> bool:
    toks = [t for t in (_unwrap_or_drop(w) for w in str(text).split()) if t is not None]
    return _all_bc(toks)


rows = []
for conv in nxt.list_conversations():
    for side in ("A", "B"):
        tp = TRANS_ROOT / str(conv)[:2] / str(conv) / f"sw{conv}{side}-ms98-a-trans.text"
        if not tp.is_file():
            continue
        utts = list(parse_transcript(tp))
        spans = [(u.start, u.end) for u in utts]
        texts = [u.transcript for u in utts]
        das = nxt.load_dialacts(conv, side)
        timed = [d for d in das if d.start is not None]
        # collect matched DA bases per utterance index
        matched: dict[int, list[str]] = {}
        hits = nxt.align_to_utterances([(d.start, d.end) for d in timed], spans)
        for d, mi in zip(timed, hits):
            if mi is not None:
                matched.setdefault(mi, []).extend(d.bases)
        for mi, bases in matched.items():
            t = texts[mi]
            rows.append(
                (
                    conv,
                    side,
                    t,
                    len(bases) > 0 and set(bases) <= BC_PRIMARY,
                    len(bases) > 0 and set(bases) <= BC_WIDE,
                    v0(t),
                    vA(t),
                    vAB(t),
                )
            )

df = pd.DataFrame(
    rows, columns=["conv", "side", "text", "gold_bc", "gold_bc_wide", "v0", "vA", "vAB"]
)
print(f"gold-labelled utterances: {len(df):,} (gold {{b}} rate {100 * df.gold_bc.mean():.2f}%)")


def prf(pred, gold):
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p + r else float("nan")
    return p, r, f1, tp, fp, fn


for gold_name, gcol in (("primary {b}", "gold_bc"), ("wide {b,bh,bk}", "gold_bc_wide")):
    print(f"\nvs gold {gold_name}:")
    for name in ("v0", "vA", "vAB"):
        p, r, f1, tp, fp, fn = prf(df[name], df[gcol])
        print(
            f"  {name:4s} P {p:.3f}  R {r:.3f}  F1 {f1:.3f}   (tp {tp} fp {fp} fn {fn})"
        )

# The flipped rows: what does gold say about them?
for name in ("vA", "vAB"):
    flip = df[df[name] & ~df.v0]
    if not len(flip):
        continue
    print(
        f"\n{name} flips {len(flip)} labelled utts non-bc → bc: "
        f"gold {{b}} {100 * flip.gold_bc.mean():.1f}%, "
        f"wide {100 * flip.gold_bc_wide.mean():.1f}% "
        f"(cf. flip-set precision needed ≥ current P to not dilute)"
    )
    print("  examples:", "; ".join(repr(t) for t in flip.text.head(8)))
