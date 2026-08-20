"""Loudness silence-dilution quantification (AUDIT.md §3 risk 6) — run 2026-08-19.

Question: `loudness mean` is frame-RMS over the whole trans-span slice, silence
included, so it mixes vocal amplitude with speech density. Is it still a valid
loudness measure, or a pausing measure in disguise?

Method: recompute frame RMS exactly as `loudness.py` does (librosa load at
22050 Hz, |stft|, rms), validate the whole-slice mean against the shipped CSV
(known-positive), then restrict to frames whose centers fall inside word-aligned
intervals ("word-tight") and compare.

Recorded results (seed 20260819):
  Utterance level (n=300): CSV validation 300/300 exact.
    log(shipped) ~ 0.95*log(word-tight) + 0.67*log(speech_frac), R^2 .959
    r(log shipped, log tight) .919 / Spearman .913; speech_frac alone R^2 .321
    dilution ratio shipped/tight: median .701, p10 .350
  Side level (40 sides x <=30 utts): side-mean shipped vs word-tight
    r(log) .978 / Spearman .978; r(log shipped, log speech_frac) .343;
    partial r(shipped, frac | tight) .765 — the arithmetic channel is real but
    contributes ~4% of between-side variance. (An earlier independent draw,
    same seed in a standalone script, gave .983/.964/partial .852 — same story.)
  Verdict: dilution shifts values, barely reorders speakers -> keep shipped
  column with the FEATURES.md caveat; word-tight variant not warranted.
  Residual dim-2a threat is telephone channel gain, not dilution.

Full-corpus CSV part needs only pandas; the audio part needs librosa and the
utterances_v2/ slices + swb_ms98_transcriptions_cleaned/ alignments.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "utterances_v2" / "features"
TRANS_ROOT = ROOT / "swb_ms98_transcriptions_cleaned"
sys.path.insert(0, str(ROOT / "src"))

from swb_extract.manifest import parse_rel_path  # noqa: E402

SEED = 20260819
N_UTT_SAMPLE = 300
N_SIDES = 40
UTTS_PER_SIDE = 30


def _index_trans_paths() -> dict[tuple[int, str], Path]:
    out = {}
    for p in TRANS_ROOT.glob("*/*/sw*-ms98-a-trans.text"):
        out[(int(p.name[2:6]), p.name[6])] = p
    return out


def _parse_utt(trans_paths, call, side, utt):
    """(bounds, sorted word intervals) for one utterance, from trans/word.text."""
    tp = trans_paths[(call, side)]
    tag = f"sw{call}{side}-ms98-a-{utt:04d}"
    bounds = None
    with open(tp) as f:
        for line in f:
            parts = line.split(maxsplit=3)
            if len(parts) >= 4 and parts[0] == tag:
                bounds = (float(parts[1]), float(parts[2]))
                break
    words = []
    with open(tp.with_name(f"sw{call}{side}-ms98-a-word.text")) as f:
        for line in f:
            parts = line.split(maxsplit=3)
            if len(parts) >= 4 and parts[0] == tag:
                words.append((float(parts[1]), float(parts[2])))
    return bounds, sorted(words)


def _recompute(trans_paths, rel):
    """(shipped-style whole mean, word-tight mean, speech fraction) or None."""
    import librosa

    call, side, utt = parse_rel_path(rel)
    bounds, words = _parse_utt(trans_paths, call, side, utt)
    if bounds is None or not words:
        return None
    t0, t1 = bounds
    try:
        y, sr = librosa.load(str(ROOT / "utterances_v2" / rel))
    except Exception:
        return None
    if y.size == 0:
        return None
    S, _ = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)[0]
    t = np.arange(len(rms)) * 512 / sr
    mask = np.zeros(len(rms), bool)
    for ws, we in words:
        mask |= (t >= ws - t0) & (t < we - t0)
    if not mask.any():
        return None
    speech = sum(min(we, t1) - max(ws, t0) for ws, we in words if we > t0 and ws < t1)
    return float(np.mean(rms)), float(np.mean(rms[mask])), speech / (t1 - t0)


def main() -> None:
    loud = pd.read_csv(FEAT / "loudness.csv").dropna(subset=["loudness mean"])
    loud = loud[loud["loudness mean"] > 0]
    loud["side"] = loud["Utterance File Name"].str.extract(r"(sw\d+[AB])")
    csv_vals = dict(zip(loud["Utterance File Name"], loud["loudness mean"]))
    trans_paths = _index_trans_paths()
    rng = random.Random(SEED)

    # --- utterance-level sample ---
    sample = rng.sample(loud["Utterance File Name"].tolist(), N_UTT_SAMPLE)
    recs, n_bad = [], 0
    for rel in sample:
        r = _recompute(trans_paths, rel)
        if r is None:
            continue
        whole, tight, frac = r
        if not math.isclose(whole, csv_vals[rel], rel_tol=1e-4, abs_tol=1e-7):
            n_bad += 1
        recs.append((whole, tight, frac))
    s = pd.DataFrame(recs, columns=["whole", "tight", "frac"])
    s = s[(s.whole > 0) & (s.tight > 0)]
    print(f"utterance level n={len(s)}, CSV mismatches={n_bad}")
    lw, lt, lf = np.log(s.whole), np.log(s.tight), np.log(s.frac)
    X = np.column_stack([lt, lf, np.ones(len(s))])
    beta, *_ = np.linalg.lstsq(X, lw, rcond=None)
    r2 = 1 - np.var(lw - X @ beta) / np.var(lw)
    print(f"  r(logW,logT)={lw.corr(lt):.3f}  Spearman={spearmanr(s.whole, s.tight)[0]:.3f}")
    print(f"  log(whole) ~ {beta[0]:.2f}·log(tight) + {beta[1]:.2f}·log(frac)  R²={r2:.3f}")
    print(f"  R²(frac alone)={lw.corr(lf)**2:.3f}  dilution median={np.median(s.whole/s.tight):.3f}")

    # --- side-level sample ---
    sides = rng.sample(sorted(loud["side"].unique()), N_SIDES)
    recs = []
    for sd in sides:
        rows = loud[loud["side"] == sd]["Utterance File Name"].tolist()
        for rel in rng.sample(rows, min(UTTS_PER_SIDE, len(rows))):
            r = _recompute(trans_paths, rel)
            if r is not None and r[0] > 0 and r[1] > 0:
                recs.append((sd, *r))
    df = pd.DataFrame(recs, columns=["side", "whole", "tight", "frac"])
    g = df.groupby("side").agg(whole=("whole", "mean"), tight=("tight", "mean"), frac=("frac", "mean"))
    lw, lt, lf = np.log(g.whole), np.log(g.tight), np.log(g.frac)
    res_w = lw - np.polyval(np.polyfit(lt, lw, 1), lt)
    res_f = lf - np.polyval(np.polyfit(lt, lf, 1), lt)
    print(f"side level n={len(g)}")
    print(f"  r(logW,logT)={lw.corr(lt):.3f}  Spearman={spearmanr(g.whole, g.tight)[0]:.3f}")
    print(f"  r(logW,logFrac)={lw.corr(lf):.3f}  partial r(W,frac|T)={np.corrcoef(res_w, res_f)[0,1]:.3f}")


if __name__ == "__main__":
    main()
