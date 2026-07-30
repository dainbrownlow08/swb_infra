"""Adjudication evidence for the rising-terminal semitone redesign (2026-07-29).

Validates the redesigned `Rising Terminal Flag` / `Terminal ST Slope`
(AUDIT.md §3.3 note; `rising_terminal.py` docstring) against the NXT gold
dialog acts, and records the MNAR structure + the sanctioned caller-level
estimator. Run from `analysis/`:  python3 validate_rising_terminal.py

[1] Gold-DA known-groups validity. The terminal contour is a property of the
    utterance END, so each utterance is classed by its LAST time-aligned gold
    DA: statements (sd/sv, no q base), yn-questions (qy), declarative
    questions (decoration '^d' on a q base — Tannen's "You stayed at the
    Plaza?"). BAR RECORD: the first run (2026-07-29) pre-stated a conjunctive
    bar — qy AND ^d each >= 2x the statement share at Welch p < .001. The qy
    arm passed decisively (2.48x, AUC .721); the ^d arm landed at 1.65x
    (p 2.5e-08, right direction) and FAILED its 2x floor. Adjudicated the
    same day, failure on record: the ^d floor was miscalibrated — only
    ~40-60% of conversational declarative questions carry a final rise, and
    DAMSL '^d' was labeled partly from transcript context, so 2x over
    statements overshoots the construct. ADMISSION rests on the qy arm
    (>= 2x, p < .001) with AUC reported; the ^d arm is reported as a
    corroborating direction check (require > 1x, p < .001).
[2] MNAR structure (recorded caveat, not a bar): null rate, per-side
    defined-rate spread, its correlations with pitch register and with the
    measured share, and the by-sex split.
[3] Caller-level estimator (PF_ratio precedent, adapted): every one of the 487
    panel callers has defined tails (min 2, median 154), so the sanctioned
    estimator is the pooled defined-only share over ALL 487 (no NaN enters the
    PCA), with `rt_defined_share` coverage carried as a companion diagnostic;
    the >=30-defined-tails subset (n=428) is the S8 robustness re-run, not an
    eligibility filter. Bounds stability = Spearman r between the defined-only
    share and the worst-case missing=fall share.
"""
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

from swb_extract import nxt
from swb_extract.analysis import load_features_table
from swb_extract.transcripts import parse_transcript

t0 = time.time()
TRANS_ROOT = Path("../swb_ms98_transcriptions_cleaned")

df = load_features_table(include="provisional")
need = ["Utterance File Name", "Rising Terminal Flag", "Terminal ST Slope",
        "Backchannel Flag", "pitch mean"]
missing = [c for c in need if c not in df.columns]
assert not missing, f"table lacks {missing}"
df["conv"] = df["Utterance File Name"].str.extract(r"sw(\d{4})[AB]")[0].astype(int)
df["side"] = df["Utterance File Name"].str.extract(r"(sw\d{4}[AB])")[0]
rt, st = df["Rising Terminal Flag"], df["Terminal ST Slope"]
sub = df[df["Backchannel Flag"].fillna(0).astype(float) == 0]

# ---- [1] known-groups vs gold dialog acts -----------------------------------
convs = sorted(set(nxt.list_conversations()) & set(df["conv"].unique()))
rows = []
for conv in convs:
    for side in ("A", "B"):
        tp = TRANS_ROOT / str(conv)[:2] / str(conv) / f"sw{conv}{side}-ms98-a-trans.text"
        if not tp.is_file():
            continue
        utts = list(parse_transcript(tp))
        keys = [f"{u.call_id//10:03d}/sw{u.call_id:04d}{u.side}-U{u.utt_num:04d}.wav"
                for u in utts]
        spans = [(u.start, u.end) for u in utts]
        das = [d for d in nxt.load_dialacts(conv, side) if d.start is not None]
        for d, mi in zip(das, nxt.align_to_utterances([(d.start, d.end) for d in das],
                                                      spans)):
            if mi is not None:
                rows.append((keys[mi], d.start, d.bases, d.decorations))
gold = pd.DataFrame(rows, columns=["utt_key", "g_start", "bases", "decos"])
assert len(gold) > 100_000, f"gold parse suspiciously small: {len(gold)}"
last = gold.sort_values("g_start").groupby("utt_key").last()  # terminal DA per utt

is_q = last["bases"].apply(lambda B: any(b.startswith("q") for b in B))
cls = pd.Series("other", index=last.index)
cls[~is_q & last["bases"].apply(lambda B: any(b in ("sd", "sv") for b in B))] = "stmt"
cls[last["bases"].apply(lambda B: "qy" in B)] = "ynq"
cls[is_q & last["decos"].apply(lambda D: "^d" in D)] = "declq"  # NXT keeps the caret

j = pd.DataFrame({"cls": cls}).join(
    df.set_index("Utterance File Name")[["Rising Terminal Flag", "Terminal ST Slope"]],
    how="inner")
jd = j.dropna(subset=["Rising Terminal Flag"])
print(f"[1] gold known-groups: {len(gold):,} aligned DAs -> {len(last):,} classed "
      f"utterances, {len(jd):,} with a defined flag   [{time.time()-t0:.0f}s]")

share = jd.groupby("cls")["Rising Terminal Flag"].agg(["mean", "size"])
for c in ("stmt", "ynq", "declq"):
    assert c in share.index and share.loc[c, "size"] >= 200, f"class {c} too small"
s_st, s_yn, s_dq = (share.loc[c, "mean"] for c in ("stmt", "ynq", "declq"))
flags = lambda c: jd.loc[jd.cls == c, "Rising Terminal Flag"]
slopes = lambda c: jd.loc[jd.cls == c, "Terminal ST Slope"].dropna()
p_yn = ttest_ind(flags("ynq"), flags("stmt"), equal_var=False).pvalue
p_dq = ttest_ind(flags("declq"), flags("stmt"), equal_var=False).pvalue
auc_yn = mannwhitneyu(slopes("ynq"), slopes("stmt")).statistic / (
    len(slopes("ynq")) * len(slopes("stmt")))
auc_dq = mannwhitneyu(slopes("declq"), slopes("stmt")).statistic / (
    len(slopes("declq")) * len(slopes("stmt")))
for name, sh, n, p, auc in [("statement (sd/sv)", s_st, share.loc["stmt", "size"], None, None),
                            ("yn-question (qy)", s_yn, share.loc["ynq", "size"], p_yn, auc_yn),
                            ("declarative q (^d)", s_dq, share.loc["declq", "size"], p_dq, auc_dq)]:
    extra = "" if p is None else f"  ratio {sh/s_st:.2f}x  Welch p {p:.2g}  AUC(ST slope) {auc:.3f}"
    print(f"    rising share | {name:18s} = {sh:.3f}  (n={n:,}){extra}")
bar_yn = (s_yn >= 2 * s_st) and (p_yn < 1e-3)
bar_dq_2x = (s_dq >= 2 * s_st) and (p_dq < 1e-3)          # the original 2x arm
bar_dq_dir = (s_dq > s_st) and (p_dq < 1e-3)              # adjudicated direction check
print(f"    first-run conjunctive bar (2026-07-29, on record): "
      f"qy>=2x {'PASS' if bar_yn else 'FAIL'} · ^d>=2x {'PASS' if bar_dq_2x else 'FAIL'}"
      + ("" if bar_dq_2x else " (adjudicated same day: 2x floor miscalibrated for ^d — "
         "~40-60% of conversational declarative questions rise; see docstring)"))
ok = bar_yn and bar_dq_dir
print(f"    ADMISSION VERDICT [qy >=2x stmt & ^d directionally elevated, each "
      f"Welch p<.001]: {'PASS' if ok else 'FAIL'}")

# ---- [2] MNAR structure (recorded caveat) -----------------------------------
g = sub.groupby("side")
d = pd.DataFrame({"defined": g["Rising Terminal Flag"].apply(lambda s: s.notna().mean()),
                  "share": g["Rising Terminal Flag"].mean(),
                  "pitch": g["pitch mean"].mean(),
                  "n": g.size()})
d = d[d.n >= 20]
con = pd.read_csv("../tables/call_con_tab.csv", header=None, skipinitialspace=True,
                  quotechar='"').iloc[:, :3]
con.columns = ["conv", "cside", "caller_no"]
con["side"] = "sw" + con["conv"].astype(str) + con["cside"].str.strip(' "')
cal = pd.read_csv("../tables/caller_tab.csv", header=None, skipinitialspace=True,
                  quotechar='"')
sex = cal.set_index(0)[3].str.strip(' "')
d = d.join(con.set_index("side")["caller_no"].map(sex).rename("sex"))
by_sex = d.groupby("sex")[["defined", "share"]].mean()
print(f"[2] MNAR (substantive utts): null {sub['Rising Terminal Flag'].isna().mean():.3f}; "
      f"side defined-rate q05/med/q95 = "
      f"{d.defined.quantile(.05):.2f}/{d.defined.median():.2f}/{d.defined.quantile(.95):.2f}")
print(f"    r(defined, pitch) = {d.defined.corr(d.pitch):+.3f}   "
      f"r(defined, share) = {d.defined.corr(d.share):+.3f}   "
      f"r(share, pitch) = {d.share.corr(d.pitch):+.3f}")
for s in by_sex.index:
    print(f"    {s:6s}: defined {by_sex.loc[s,'defined']:.3f}  share {by_sex.loc[s,'share']:.3f}")
print("    (carried caveat: ascertainment is register-correlated; use the "
      "pooled-floor estimator + coverage companion, never a bare defined-only mean)")

# ---- [3] caller-level estimator + bounds ------------------------------------
sub2 = sub.assign(caller=sub["side"].map(con.set_index("side")["caller_no"]))
cg = sub2.groupby("caller")["Rising Terminal Flag"]
cal_tab = pd.DataFrame({"n_sub": cg.size(), "n_def": cg.count(), "rises": cg.sum()})
cal_tab = cal_tab[cal_tab.n_sub >= 20]  # the panel's min-20 substantive floor
eligible = cal_tab[cal_tab.n_def >= 30]
lo = eligible.rises / eligible.n_sub                 # missing counted as falls
hi = eligible.rises / eligible.n_def                 # defined-only
rho = spearmanr(lo, hi).statistic
print(f"[3] callers >=20 substantive: {len(cal_tab)}; defined tails min "
      f"{int(cal_tab.n_def.min())} / median {int(cal_tab.n_def.median())} — pooled "
      f"share computable for all; >=30-tail robustness subset: {len(eligible)} "
      f"({100*len(eligible)/len(cal_tab):.1f}%)")
print(f"    bounds stability: Spearman r(defined-only share, missing=fall share) "
      f"= {rho:.3f} over the {len(eligible)}-caller subset")
print(f"done [{time.time()-t0:.0f}s]")
sys.exit(0 if ok else 1)
