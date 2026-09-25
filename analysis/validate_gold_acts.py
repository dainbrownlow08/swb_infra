"""Admission evidence for the gold_acts extractor (2026-08-21).

Two joins of the same SWBD-DAMSL annotation to ms98 utterances exist on the 544
conversations that both NXT (time-aligned) and SwDA (text-aligned) cover. The text join
is what labels the 458 conversations NXT lacks, so its agreement with the time join on
the overlap is its accuracy estimate. Bars stated before the run:

  G1a  SwDA side match ratio: median >= .90, and <= 5% of sides below MIN_MATCH_RATIO (.85).
  G1b  Question-flag agreement (both joins label the utterance) >= .95; information-
       and echo-question flags reported alongside. Raw terminal-tag agreement is reported,
       not gated: on multi-act utterances the latest-STARTING act (time rule) and the act
       on the last MATCHED word (text rule) may legitimately differ, and the two sources
       spell a few tags differently (NXT '%-' = SwDA '%').
  G1c  Coverage: the text join labels >= .95 as many utterances as the time join.
  G2   The written CSV (if present): blank rows are exactly the utterances of unlabelled
       conversations; question rate among labelled utterances within 5–10%; both sources
       present.

Run from the repo root:  PYTHONPATH=src python3 analysis/validate_gold_acts.py
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from swb_extract import nxt, swda  # noqa: E402
from swb_extract.features.inhouse import gold_acts as G  # noqa: E402
from swb_extract.features.inhouse._turn_index import build_text_index  # noqa: E402

TR = Path("swb_ms98_transcriptions_cleaned")
t0 = time.time()
man = [r[0] for r in csv.reader(open("utterances_v2/manifest.csv")) if r and r[0] != "Utterance File Name"]
convs_in = {int(r[6:10]) for r in man}
overlap = sorted(set(nxt.list_conversations()) & set(swda.list_conversations()) & convs_in)

def norm(tag: str) -> str:
    return tag.replace("%-", "%").replace("(^q)", "^q")

ratios, low = [], []
agree = Counter(); n_both = 0; n_nxt = n_swda = 0
pairs = Counter(); examples = []
texts = build_text_index(TR)
n_swapped = 0
band = {"[.75,.85)": [0, 0], "[.85,.95)": [0, 0], "[.95,1]": [0, 0]}  # [agree, n] for the question flag
for conv in overlap:
    a = {}
    for side in "AB":
        a.update(G.nxt_terminal_acts(conv, side, TR))
    b, side_ratios, swapped = G.swda_conversation_acts(conv, TR)
    n_swapped += swapped
    for side, ratio in side_ratios.items():
        ratios.append(ratio)
        if ratio < swda.MIN_MATCH_RATIO:
            low.append((conv, side, round(ratio, 3)))
    n_nxt += len(a); n_swda += len(b)
    for k in set(a) & set(b):
        n_both += 1
        fa, fb = G.flags_for(a[k]), G.flags_for(b[k])
        agree["tag"] += norm(a[k]) == norm(b[k])
        agree["question"] += fa[0] == fb[0]
        agree["information"] += fa[1] == fb[1]
        agree["echo"] += fa[2] == fb[2]
        r = side_ratios[k[1]]
        key = "[.75,.85)" if r < .85 else ("[.85,.95)" if r < .95 else "[.95,1]")
        band[key][1] += 1; band[key][0] += fa[0] == fb[0]
        if fa[0] != fb[0]:
            pairs[(norm(a[k]), norm(b[k]))] += 1
            if len(examples) < 8:
                examples.append((k, a[k], b[k], texts.get(k, "")[:70]))
med = sorted(ratios)[len(ratios) // 2]
print(f"[G1] overlap conversations {len(overlap)} ({len(ratios)} sides); SwDA caller labels A/B-swapped in {n_swapped} [{time.time()-t0:.0f}s]")
print(f"  G1a match ratio: median {med:.3f}, q05 {sorted(ratios)[len(ratios)//20]:.3f}; sides below {swda.MIN_MATCH_RATIO}: {len(low)} ({100*len(low)/len(ratios):.1f}%) -> {'PASS' if med >= .90 and len(low) <= .05*len(ratios) else 'FAIL'}")
print(f"  G1b agreement on {n_both:,} utterances labelled by both: question flag {agree['question']/n_both:.4f}  information {agree['information']/n_both:.4f}  echo {agree['echo']/n_both:.4f}  (raw terminal tag {agree['tag']/n_both:.3f}) -> {'PASS' if agree['question']/n_both >= .95 else 'FAIL'}")
print(f"  G1c coverage: time join {n_nxt:,} labelled, text join {n_swda:,} ({n_swda/n_nxt:.3f}) -> {'PASS' if n_swda/n_nxt >= .95 else 'FAIL'}")
print("  question-flag agreement by side match-ratio band:", {k: (round(v[0]/v[1], 4) if v[1] else None, v[1]) for k, v in band.items()})
print("  question-flag disagreements by (nxt, swda) tag:", pairs.most_common(8))
for k, ta, tb, tx in examples:
    print(f"    sw{k[0]}{k[1]}-{k[2]:04d}  nxt={ta:8s} swda={tb:8s} | {tx}")
ok = med >= .90 and len(low) <= .05 * len(ratios) and agree["question"] / n_both >= .95 and n_swda / n_nxt >= .95

csv_path = Path("utterances_v2/features/gold_acts.csv")
if csv_path.is_file():
    rows = list(csv.DictReader(open(csv_path)))
    lab = [r for r in rows if r["Gold Act"]]
    blank_convs = {int(r["Utterance File Name"][6:10]) for r in rows if not r["Gold Act"]}
    lab_convs = {int(r["Utterance File Name"][6:10]) for r in lab}
    src = Counter(r["Gold DA Source"] for r in lab)
    qrate = sum(r["Gold Question Flag"] == "1" for r in lab) / len(lab)
    leak = blank_convs & lab_convs
    print(f"\n[G2] gold_acts.csv: {len(rows):,} rows, labelled {len(lab):,} ({100*len(lab)/len(rows):.1f}%) in {len(lab_convs)} convs; sources {dict(src)}; question rate {qrate:.3%}; "
          f"convs with both labelled and blank rows: {len(leak)} (blank rows there = utterances that received no act) -> {'PASS' if .05 <= qrate <= .10 and src.get('nxt') and src.get('swda') else 'FAIL'}")
    ok = ok and .05 <= qrate <= .10
print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}  [{time.time()-t0:.0f}s]")
sys.exit(0 if ok else 1)
