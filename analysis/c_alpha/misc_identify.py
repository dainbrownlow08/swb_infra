"""Method identification on MISC: which of the paper's unstated choices move us onto its numbers.

Targets (Thomas et al. 2018 §4.1, n = 168): GLB .85, Cronbach's α .67, PC1 29% of variance,
Table 2 loadings. Our baseline (misc_variables.py defaults) on the equivalent sample — tasks
2–5 minus the five named outlier participant-tasks, n = 171 — is scored first; then each
unstated choice is varied one at a time, and the best-matching combination is reported.
Every variant is a documented unknown of their method (the paper does not say), evaluated
ONLY on MISC against the published result. Writes misc_identify_variants.csv.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from swb_extract.features.inhouse._text import tokenize  # noqa: E402
from swb_extract.features.inhouse.pronoun_per_second import PERSONAL_PRONOUNS  # noqa: E402
from swb_extract.features.thomas2018 import PAPER_LOADINGS, VARIABLES  # noqa: E402
from swb_extract.features.thomas2018._terms import FILLERS, STOPWORDS  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mv", HERE / "misc_variables.py"); mv = importlib.util.module_from_spec(spec); spec.loader.exec_module(mv)
spec = importlib.util.spec_from_file_location("tm", HERE / "thomas_method.py"); tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
V = list(VARIABLES); T = np.array([PAPER_LOADINGS[v] for v in V])
R = mv.R

LIWC_EXTRA = frozenset({"ya", "u", "ye", "thou", "thee", "thy", "lets", "ur", "yall", "id", "im", "ive", "ill", "youre", "youve", "youll", "youd", "weve", "wed", "were"})  # LIWC2015 i/we/you forms beyond the 16
_STEM = {}


def stemmer(kind):
    if kind not in _STEM:
        from nltk.stem import PorterStemmer, SnowballStemmer
        _STEM[kind] = PorterStemmer() if kind == "porter" else SnowballStemmer("english")
    return _STEM[kind]


def terms(text, kind):
    st = stemmer(kind); out = set()
    for w in tokenize(text):
        w = re.sub(r"_\d+$", "", w)
        if w and w not in FILLERS and w not in STOPWORDS:
            out.add(st.stem(w))
    return out


def count_pron(text, liwc):
    n = 0
    for w in tokenize(text):
        base = w.split("'")[0]
        if w in PERSONAL_PRONOUNS or base in PERSONAL_PRONOUNS or (liwc and (w in LIWC_EXTRA or w.replace("'", "") in LIWC_EXTRA)):
            n += 1
    return n


def pauses(pros, lines, cfg):
    if cfg["pause"] == "vprob":
        voiced = (pros.voice_probability.to_numpy() > 0.5)
    else:
        voiced = (pros.F0.to_numpy() > 0)
    times = pros.time.to_numpy(); n = 0; sec = 0.0; minrun = 2 if cfg["pause"] == "f0zero_min2" else 1
    for s, e in zip(lines.start, lines.end):
        v = voiced[np.searchsorted(times, s):np.searchsorted(times, e)]
        idx = np.flatnonzero(v)
        if idx.size < 2: continue
        g = np.diff(idx) - 1; g = g[g >= minrun]
        n += int(g.size); sec += float(g.sum() * mv.FRAME)
    return n, sec


def pvlv(pros, lines, cfg):
    if cfg["pvlv"] == "vprob":
        m = pros.voice_probability > 0.5
    elif cfg["pvlv"] == "utt":
        times = pros.time.to_numpy(); m = np.zeros(len(pros), bool)
        for s, e in zip(lines.start, lines.end): m[np.searchsorted(times, s):np.searchsorted(times, e)] = True
        m &= (pros.F0 > 0).to_numpy()
    else:
        m = pros.F0 > 0
    return float(pros.F0[m].var()), float(pros.pcm_loudness[m].var())


def olap_voicing(own, partner_pros):
    times = partner_pros.time.to_numpy(); voiced = partner_pros.F0.to_numpy() > 0
    flags = []
    for s in own.start:
        i0, i1 = np.searchsorted(times, s - 0.1), np.searchsorted(times, s)
        flags.append(int(voiced[i0:i1].any()))
    return float(np.mean(flags)) if flags else np.nan


def build(cfg):
    rows = []
    for pair_dir in sorted((R / "Transcripts").iterdir()):
        ps = sorted({int(re.search(r"P(\d+)", f.name).group(1)) for f in pair_dir.glob("P*_T*.tsv")})
        if len(ps) != 2: continue
        for t in (2, 3, 4, 5):
            for p, q in ((ps[0], ps[1]), (ps[1], ps[0])):
                own, partner, pros = mv.load_lines(pair_dir.name, p, t), mv.load_lines(pair_dir.name, q, t), mv.load_prosody(pair_dir.name, p, t)
                words = np.array([len(tokenize(x)) for x in own.transcript]); dur = (own.end - own.start).to_numpy()
                n_p, sec_p = pauses(pros, own, cfg); pv, lv = pvlv(pros, own, cfg)
                ol, pop, naive = mv.onset_timing(own, partner)
                ts = [terms(x, cfg["stem"]) for x in own.transcript]; reps = [len(ts[i] & ts[i-1]) for i in range(1, len(ts))]
                rows.append(dict(pair=pair_dir.name, participant=p, task=t, n_utts=len(own),
                    ppron=sum(count_pron(x, cfg["liwc"]) for x in own.transcript) / words.sum(),
                    wps=words.sum() / dur.sum(), wpu=float(words.mean()), wpp=words.sum() / n_p if n_p else np.nan, boplen=sec_p / n_p if n_p else np.nan,
                    poplen=(float(np.nanmean(naive)) if cfg["poplen"] == "naive" else float(np.mean([x for x in pop if x is not None]))),
                    pv=pv, lv=lv,
                    olap=(olap_voicing(own, mv.load_prosody(pair_dir.name, q, t)) if cfg["olap"] == "voicing" else float(np.mean(ol))),
                    rept=float(np.mean(reps)), repu=float(np.mean([r >= 1 for r in reps]))))
    d = pd.DataFrame(rows); names = mv.task_names()
    d["task_name"] = [names.get((p, t), "?") for p, t in zip(d.participant, d.task)]
    return d[~d.apply(lambda r: (r.participant, r.task_name) in mv.OUTLIERS, axis=1)].dropna(subset=V)


def glb_fa(Rm, nf):
    """psych::glb.fa analogue: 1 − Σ uniqueness / Var(total), uniquenesses from an nf-factor principal-axis fit."""
    k = Rm.shape[0]; h2 = 1 - 1 / np.diag(np.linalg.pinv(Rm))
    for _ in range(200):
        Rr = Rm.copy(); np.fill_diagonal(Rr, h2); w, v = np.linalg.eigh(Rr); order = np.argsort(w)[::-1][:nf]
        L = v[:, order] * np.sqrt(np.clip(w[order], 0, None)); h2n = np.clip((L ** 2).sum(1), 0, 0.999)
        if np.max(np.abs(h2n - h2)) < 1e-7: h2 = h2n; break
        h2 = h2n
    return float(1 - (1 - h2).sum() / Rm.sum())


def alpha_checkkeys(Z):
    """psych::alpha(check.keys=TRUE): reverse items that correlate negatively with the total."""
    keys = np.ones(Z.shape[1])
    for _ in range(5):
        tot = (Z * keys).sum(1); new = np.array([1 if np.corrcoef(Z[:, j], tot)[0, 1] >= 0 else -1 for j in range(Z.shape[1])])
        if (new == keys).all(): break
        keys = new
    return tm.cronbach(Z * keys)[0], keys


def score(d, label):
    Z = tm.zscore(d[V].to_numpy(float)); vec, eig = tm.pc1(Z, V.index("wps"))
    Zk = Z.copy()
    for v in ("boplen", "poplen", "olap"): Zk[:, V.index(v)] *= -1
    a_k, _ = tm.cronbach(Zk); a_ck, keys = alpha_checkkeys(Z); Rm = np.corrcoef(Zk, rowvar=False)
    return dict(variant=label, n=len(d), alpha_keyed=round(a_k, 3), alpha_checkkeys=round(a_ck, 3), checkkeys_flips="".join("-" if k < 0 else "+" for k in keys),
                glb_fa1=round(glb_fa(Rm, 1), 3), glb_fa2=round(glb_fa(Rm, 2), 3), glb_fa3=round(glb_fa(Rm, 3), 3),
                pc1_share=round(float(eig[0] / eig.sum()), 3), phi=round(tm.congruence(vec, T), 4), signs=int((np.sign(vec) == np.sign(T)).sum()),
                abs_loading_gap=round(float(np.abs(vec - T).sum()), 3), **{f"w_{v}": round(float(x), 2) for v, x in zip(V, vec)})


BASE = dict(pause="f0zero", pvlv="f0pos", poplen="blank", olap="transcript", stem="porter", liwc=False)
OPTIONS = dict(pause=["f0zero", "vprob", "f0zero_min2"], pvlv=["f0pos", "vprob", "utt"], poplen=["blank", "naive"], olap=["transcript", "voicing"], stem=["porter", "snowball"], liwc=[False, True])

if __name__ == "__main__":
    out = []
    base = build(BASE); out.append(score(base, "baseline (n=171 sample)"))
    print("baseline:", {k: out[-1][k] for k in ("n", "alpha_keyed", "alpha_checkkeys", "glb_fa1", "glb_fa2", "glb_fa3", "pc1_share", "phi", "signs")})
    # LIWC calibration of ppron on the baseline
    L = pd.read_csv(R / "LIWC" / "LIWC.tsv", sep="\t"); L["liwc_12"] = L["i"] + L["we"] + L["you"]
    m = base.merge(L[["participant", "task", "liwc_12", "ppron"]].rename(columns={"ppron": "liwc_ppron"}), on=["participant", "task"])
    print(f"ppron vs LIWC (i+we+you, % of words): r {np.corrcoef(100*m.ppron, m.liwc_12)[0,1]:.3f}; ours/LIWC median ratio {(100*m.ppron/m.liwc_12).median():.2f} (paper: their list 'captured more terms than did LIWC')")
    for key, opts in OPTIONS.items():
        for o in opts:
            if o == BASE[key]: continue
            cfg = dict(BASE, **{key: o}); out.append(score(build(cfg), f"{key}={o}"))
            print(f"  {key}={str(o):12s}", {k: out[-1][k] for k in ("alpha_keyed", "alpha_checkkeys", "glb_fa2", "pc1_share", "phi", "signs", "abs_loading_gap")})
    # --- sample sensitivity with the baseline routes: which subset of MISC were they on?
    raw = pd.read_csv(HERE / "misc_variables_raw.csv"); raw = raw[raw.task >= 2]
    outl = raw.apply(lambda r: (r.participant, r.task_name) in mv.OUTLIERS, axis=1)
    import wave, glob
    dur = {}
    for f in glob.glob(str(R / "Audio" / "*" / "P*_T*.wav")):
        m_ = re.search(r"P(\d+)_T(\d)", f); w = wave.open(f); dur[(int(m_.group(1)), int(m_.group(2)))] = w.getnframes() / w.getframerate()
    raw["audio_sec"] = [dur.get((p, t), np.nan) for p, t in zip(raw.participant, raw.task)]
    samples = {"all tasks 2-5 (176)": raw, "minus named outliers (171 ≈ their 168)": raw[~outl],
               "<=10 min by transcript, minus outliers (101)": raw[~outl & (raw.last_end <= 600)],
               "<=10 min by audio, minus outliers (99 ≈ their 98)": raw[~outl & (raw.audio_sec <= 600)]}
    print("\nsample sensitivity (baseline routes):")
    for lab, d in samples.items():
        sc = score(d.dropna(subset=V), f"sample: {lab}"); out.append(sc)
        print(f"  {lab:48s} n={sc['n']:3d}  alpha_keyed {sc['alpha_keyed']:.3f}  glb_fa3 {sc['glb_fa3']:.3f}  pc1 {sc['pc1_share']:.3f}  phi {sc['phi']:.4f}  signs {sc['signs']}")
    # --- greedy combination search on the n=171 sample (coordinate ascent on phi; ≤ 2 sweeps)
    cfg = dict(BASE); best = score(base, "greedy start"); tried = 1
    for _sweep in range(2):
        improved = False
        for key, opts in OPTIONS.items():
            for o in opts:
                if o == cfg[key]: continue
                cand = dict(cfg, **{key: o}); sc = score(build(cand), "cand"); tried += 1
                if sc["phi"] > best["phi"] + 1e-4:
                    cfg, best, improved = cand, sc, True
        if not improved: break
    best["variant"] = "greedy best on n=171: " + ", ".join(f"{k}={v}" for k, v in cfg.items()); out.append(best)
    print(f"\ngreedy best ({tried} combinations scored): {cfg}\n  ", {k: best[k] for k in ("alpha_keyed", "alpha_checkkeys", "glb_fa2", "glb_fa3", "pc1_share", "phi", "signs", "abs_loading_gap")})
    pd.DataFrame(out).to_csv(HERE / "misc_identify_variants.csv", index=False)
    print("wrote misc_identify_variants.csv")
