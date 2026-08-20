"""Echo-detector adjudication vs gold ^m mirror — ROUTE CLOSED 2026-08-19.

Question: can a full-corpus allo-repetition (Tannen dim 6) detector be built
from exact lexical echo of the last other-speaker utterance, validated against
the NXT gold ^m (mirror/repeat-other) dialact tag?

Recorded result (all 642 gold convs, 64,376 utterances, 598 gold ^m utts,
base rate 0.93%): NO. The gold construct is not lexically recoverable from
adjacent-utterance matching — 75/598 gold mirrors share ZERO contiguous tokens
with the preceding other-speaker utterance and 277 share only one (recall
ceiling 41% for any shared-bigram rule); annotators marked functional mirroring
(person-transformed repeats, echoes of earlier material). Best swept rule
(lcs>=2 & cover>=0.5): P .238 / R .186 / F1 .208 — far under the project's
pre-registered 0.8 bar (Question Flag precedent: fail -> exclude). Dim 6
therefore stays measured by the gold ^m rate on the 642-conv subset;
`Repetitions In Previous Utterance` keeps its bag-of-words semantics with its
construct caveats recorded in FEATURES.md (side-level r with gold mirror:
.10 raw / .19 normalized other-speaker).
"""
import sys
from pathlib import Path

sys.path.insert(0, "/Users/dainbrownlow/switchboard/src")
import pandas as pd

from swb_extract import nxt
from swb_extract.features._text import tokenize
from swb_extract.features._turn_index import build_text_index, build_turn_gap_index

TROOT = Path("/Users/dainbrownlow/switchboard/swb_ms98_transcriptions_cleaned")
merged = build_turn_gap_index(TROOT)
texts = build_text_index(TROOT)


def lcs_ngram(a: list[str], b: list[str]) -> int:
    """Longest common contiguous token run (classic DP, small inputs)."""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


rows = []
convs = nxt.list_conversations()
for conv in convs:
    entries = merged.get(conv)
    if not entries:
        continue
    # utterance spans per side for alignment
    spans = {"A": [], "B": []}
    keys = {"A": [], "B": []}
    for s, u, st, en in entries:
        spans[s].append((st, en))
        keys[s].append(u)
    gold_pos = set()  # (side, utt_num) containing a ^m dialact
    for side in "AB":
        try:
            das = nxt.load_dialacts(conv, side)
        except Exception:
            continue
        mirrors = [(d.start, d.end) for d in das
                   if ("^m" in d.decorations or "^m" in d.bases)
                   and d.start is not None and d.end is not None]
        if not mirrors:
            continue
        hits = nxt.align_to_utterances(mirrors, spans[side])
        for h in hits:
            if h is not None:
                gold_pos.add((side, keys[side][h]))
    # detector features per utterance (vs last OTHER-speaker utterance)
    last_other_text = {"A": None, "B": None}  # most recent utt by the other side
    for i, (s, u, st, en) in enumerate(entries):
        prev_text = last_other_text[s]
        cur = tokenize(texts.get((conv, s, u), ""))
        if prev_text is not None and cur:
            prev = tokenize(prev_text)
            l = lcs_ngram(cur, prev)
            rows.append((conv, s, u, l, len(cur), (s, u) in gold_pos))
        else:
            rows.append((conv, s, u, 0, len(cur), (s, u) in gold_pos))
        other = "B" if s == "A" else "A"
        last_other_text[other] = texts.get((conv, s, u), "")

df = pd.DataFrame(rows, columns=["conv", "side", "utt", "lcs", "len", "gold"])
df["cover"] = df.lcs / df["len"].clip(lower=1)
print(f"utterances {len(df):,}; gold ^m utterances {df.gold.sum():,} ({df.gold.mean()*100:.2f}%)")


def pr(mask, label):
    tp = (mask & df.gold).sum()
    p = tp / mask.sum() if mask.sum() else 0
    r = tp / df.gold.sum()
    f = 2 * p * r / (p + r) if p + r else 0
    print(f"  {label:38s} flagged {mask.sum():6,}  P {p:.3f}  R {r:.3f}  F1 {f:.3f}")


for k in (2, 3, 4, 5):
    pr(df.lcs >= k, f"lcs >= {k}")
for c in (0.5, 0.8, 1.0):
    pr((df.cover >= c) & (df.lcs >= 1), f"cover >= {c}")
for c in (0.8, 1.0):
    for L in (2, 3, 5):
        pr((df.cover >= c) & (df["len"] <= L) & (df.lcs >= 1), f"cover>={c} & len<={L}")
pr((df.lcs >= 2) & (df.cover >= 0.5), "lcs>=2 & cover>=0.5")
# what do gold mirrors look like on these features?
g = df[df.gold]
print("\ngold ^m utterances: lcs distribution:")
print(g.lcs.value_counts().sort_index().to_string())
print(f"gold cover: med {g.cover.median():.2f}, mean {g.cover.mean():.2f}; len med {g['len'].median():.0f}")
