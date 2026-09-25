"""The eleven Thomas et al. (2018) variables computed on the MISC release — their data, our code.

Builds the participant × task table from ``corpus/misc/MISC_Public_Release`` along the
paper's own routes, using the thomas2018 package's pure functions where the route is shared:

  utterance = one STT transcript line (start, end, text)            — their boundaries
  wpu   mean words per line; wps = Σwords / Σ(end − start)            — §3.2 verbatim
  ppron Σ 1st/2nd-person pronouns / Σwords (our 16-form list; theirs unpublished)
  boplen, wpp  between-own pauses = runs of OpenSMILE F0 == 0 frames bounded by F0 > 0
        frames inside a line ("periods where OpenSMILE reports no F0"), 10 ms frames — their
        exact signal (prosodyShs preset, ``Prosodic_Analysis/``)
  pv, lv  variance of F0 / pcm_loudness over all frames with F0 > 0 in the task recording
        ("the variance across the entire recording") — their exact signal
  olap  share of lines that start while the partner's line is still running (pair clocks
        verified shared: voicing anti-correlates at lag 0, tracks cut to equal length)
  poplen  partner's end → own next start, our rule (blank in overlap / own-own gaps);
        ``poplen_naive`` = start − partner's latest end with negatives kept, the route a
        paper silent on overlaps may have taken
  rept, repu  stemmed terms shared with own previous line (``_terms``: NLTK stopwords +
        um/uh/uh-huh removed, Porter)

Cleaning as §3.4: task 1 (warm-up) dropped; any participant-task whose recording runs past
600 s dropped; the named outliers dropped — participant 2 transport, participant 5 migraine
and Olympics, participants 17 and 18 transport. Task names come from Questionnaires/
Per-task.tsv column B (order codes: S/C simple/complex × E/D easy/difficult → SE heroism,
CE migraine, SD Olympics, CD transport, per §3.1). Two tables are written:
misc_variables_all.csv (tasks 2–5, no removals) and misc_variables_paper_clean.csv.

Run from the repo root:  PYTHONPATH=src python3 analysis/c_alpha/misc_variables.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from swb_extract.features.inhouse._text import tokenize  # noqa: E402
from swb_extract.features.inhouse.pronoun_per_second import count_personal_pronouns  # noqa: E402
from swb_extract.features.thomas2018._terms import terms  # noqa: E402

R = Path("corpus/misc/MISC_Public_Release")
HERE = Path(__file__).resolve().parent
FRAME = 0.01
TASK_CODE = {"SE": "heroism", "CE": "migraine", "SD": "olympics", "CD": "transport"}
OUTLIERS = {(2, "transport"), (5, "migraine"), (5, "olympics"), (17, "transport"), (18, "transport")}


def load_lines(pair: str, p: int, t: int) -> pd.DataFrame:
    d = pd.read_csv(R / "Transcripts" / pair / f"P{p:02d}_T{t}.tsv", sep="\t")
    d["transcript"] = d["transcript"].fillna("").astype(str)
    return d.sort_values("start").reset_index(drop=True)


TRACKS = "shipped"  # "shipped" = MISC's Prosodic_Analysis TSVs (the paper's tracks); "v3" = our openSMILE 3.0 re-tracking


def load_prosody(pair: str, p: int, t: int) -> pd.DataFrame:
    if TRACKS == "v3":
        z = np.load(R / ".." / "derived_opensmile_v3" / f"P{p:02d}_T{t}.npz")
        return pd.DataFrame({"time": np.arange(len(z["f0"])) * FRAME, "voice_probability": z["voicing"], "F0": z["f0"], "pcm_loudness": z["loudness"]})
    return pd.read_csv(R / "Prosodic_Analysis" / pair / f"P{p:02d}_T{t}.tsv", sep="\t")


def between_own_pauses(pros: pd.DataFrame, lines: pd.DataFrame) -> tuple[int, float]:
    """(count, seconds) of F0==0 runs bounded by F0>0 frames inside each line."""
    voiced = (pros.F0.to_numpy() > 0)
    times = pros.time.to_numpy()
    n = 0; sec = 0.0
    for s, e in zip(lines.start, lines.end):
        i0, i1 = np.searchsorted(times, s), np.searchsorted(times, e)
        v = voiced[i0:i1]
        idx = np.flatnonzero(v)
        if idx.size < 2:
            continue
        gaps = np.diff(idx) - 1
        gaps = gaps[gaps > 0]
        n += int(gaps.size); sec += float(gaps.sum() * FRAME)
    return n, sec


def onset_timing(own: pd.DataFrame, partner: pd.DataFrame) -> tuple[list[int], list[float | None], list[float]]:
    """Per own line: (olap flag, poplen by our rule, naive poplen)."""
    ev = sorted([(s, e, "own") for s, e in zip(own.start, own.end)] + [(s, e, "p") for s, e in zip(partner.start, partner.end)])
    last_end = {"own": None, "p": None}
    olap, pop, naive = [], [], []
    for s, e, who in ev:
        if who == "own":
            other, mine = last_end["p"], last_end["own"]
            if other is None:
                olap.append(0); pop.append(None); naive.append(np.nan)
            else:
                naive.append(s - other)
                if other > s:
                    olap.append(1); pop.append(None)
                elif mine is not None and mine > other:
                    olap.append(0); pop.append(None)
                else:
                    olap.append(0); pop.append(s - other)
        last_end[who] = e if last_end[who] is None else max(last_end[who], e)
    return olap, pop, naive


def variables(pair: str, p: int, q: int, t: int) -> dict:
    own, partner, pros = load_lines(pair, p, t), load_lines(pair, q, t), load_prosody(pair, p, t)
    toks = [tokenize(x) for x in own.transcript]
    words = np.array([len(w) for w in toks]); dur = (own.end - own.start).to_numpy()
    prons = sum(count_personal_pronouns(x) for x in own.transcript)
    n_p, sec_p = between_own_pauses(pros, own)
    voiced = pros.F0 > 0
    olap, pop, naive = onset_timing(own, partner)
    ts = [terms(x) for x in own.transcript]
    reps = [len(ts[i] & ts[i - 1]) for i in range(1, len(ts))]
    return dict(
        pair=pair, participant=p, task=t, role="seeker" if p % 2 else "intermediary",
        n_utts=len(own), words=int(words.sum()), talk_sec=float(dur.sum()), last_end=float(max(own.end.max(), partner.end.max())),
        ppron=prons / words.sum() if words.sum() else np.nan,
        wps=words.sum() / dur.sum() if dur.sum() > 0 else np.nan,
        wpu=float(words.mean()) if len(words) else np.nan,
        wpp=words.sum() / n_p if n_p else np.nan,
        boplen=sec_p / n_p if n_p else np.nan,
        poplen=float(np.mean([x for x in pop if x is not None])) if any(x is not None for x in pop) else np.nan,
        poplen_naive=float(np.nanmean(naive)) if np.isfinite(naive).any() else np.nan,
        pv=float(pros.F0[voiced].var()) if voiced.sum() > 1 else np.nan,
        lv=float(pros.pcm_loudness[voiced].var()) if voiced.sum() > 1 else np.nan,
        olap=float(np.mean(olap)) if olap else np.nan,
        rept=float(np.mean(reps)) if reps else np.nan,
        repu=float(np.mean([r >= 1 for r in reps])) if reps else np.nan,
        n_pauses=n_p,
    )


def task_names() -> dict[tuple[int, int], str]:
    q = pd.read_csv(R / "Questionnaires" / "Per-task.tsv", sep="\t")
    out = {}
    for _, r in q.iterrows():
        codes = str(r.iloc[1]).split()
        for t, code in zip((2, 3, 4, 5), codes):
            out[(int(r.iloc[0]), t)] = TASK_CODE.get(code, code)
    return out


def main() -> int:
    rows = []
    for pair_dir in sorted((R / "Transcripts").iterdir()):
        ps = sorted({int(re.search(r"P(\d+)", f.name).group(1)) for f in pair_dir.glob("P*_T*.tsv")})
        if len(ps) != 2:
            continue
        for t in (1, 2, 3, 4, 5):
            for p, q in ((ps[0], ps[1]), (ps[1], ps[0])):
                if (pair_dir / f"P{p:02d}_T{t}.tsv").is_file() and (pair_dir / f"P{q:02d}_T{t}.tsv").is_file():
                    rows.append(variables(pair_dir.name, p, q, t))
    d = pd.DataFrame(rows)
    names = task_names()
    d["task_name"] = [names.get((p, t), "warmup" if t == 1 else "?") for p, t in zip(d.participant, d.task)]
    tag = "" if TRACKS == "shipped" else f"_{TRACKS}tracks"
    d.to_csv(HERE / f"misc_variables_raw{tag}.csv", index=False)
    allt = d[d.task >= 2].copy(); allt.to_csv(HERE / f"misc_variables_all{tag}.csv", index=False)
    clean = allt[(allt.last_end <= 600) & ~allt.apply(lambda r: (r.participant, r.task_name) in OUTLIERS, axis=1)].copy()
    clean.to_csv(HERE / f"misc_variables_paper_clean{tag}.csv", index=False)
    print(f"participant-tasks: raw {len(d)} (paper: 220) | tasks 2–5: {len(allt)} (paper: 176 max) | after >10-min + named-outlier removal: {len(clean)} (paper: 98)")
    print("task-name coverage:", d.task_name.value_counts().to_dict())
    V = ["ppron", "wps", "wpu", "wpp", "boplen", "poplen", "pv", "lv", "olap", "rept", "repu"]
    print(clean[V].describe().loc[["mean", "50%", "std"]].round(3).to_string())
    print(f"paper-stated magnitudes: between-own pauses 'on the order of 0.1 s' → ours {clean.boplen.median():.3f}; pitch variation 'on the order of 1000 Hz²' → ours {clean.pv.median():.0f}")
    return 0


if __name__ == "__main__":
    if "--tracks" in sys.argv:
        TRACKS = sys.argv[sys.argv.index("--tracks") + 1]
    sys.exit(main())
