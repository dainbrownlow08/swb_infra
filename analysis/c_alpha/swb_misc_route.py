"""Switchboard's eleven Thomas variables along the MISC route — like-for-like with misc_variables.py.

Same construction as the MISC table, applied to Switchboard with one detector for both corpora:

  utterance = one ms98 transcript line with its own line timestamps (the paper's "line of
              transcript"; NOT word-tight — that is the in-house route)
  wpu, wps  = words per line; Σwords / Σ(line end − start)
  ppron     = Σ 1st/2nd-person pronouns / Σwords (the same list as the MISC run)
  pauses    = runs of openSMILE prosodyShs F0 == 0 frames bounded by F0 > 0 frames inside a
              line (10 ms frames; tracks from utterances_v2/derived/opensmile/<side>.npz,
              produced by openSMILE 3.0 over the whole side recording — the same binary and
              preset used to re-track MISC in corpus/misc/derived_opensmile_v3/)
  pv, lv    = variance of F0 / pcm_loudness over F0 > 0 frames of the whole side recording
              (the paper's "across the entire recording"); *_inlines variants restrict to
              frames inside the side's lines (guards against partner bleed on the channel)
  olap, poplen = onset rules on speech-tight line spans vs the partner's lines (an STT segment's
              times hug the speech, so first-word onset → last-word offset is its analogue; the
              padded ms98 line bounds give the *_trans sensitivity columns)
  rept, repu   = stemmed content terms shared with the speaker's own previous line

Per side also the ingredient sums (words, seconds, pronouns, pause counts/seconds, voiced
frame n/Σ/Σ² for F0 and loudness, overlap counts, poplen sums, repeat sums) so the caller
table pools exactly instead of averaging side values.

Outputs (this folder): swb_misc_route_sides.csv, swb_misc_route_callers.csv.
Run from the repo root:  PYTHONPATH=src python3 analysis/c_alpha/swb_misc_route.py [--limit N]
"""
from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from swb_extract.features.inhouse._text import tokenize  # noqa: E402
from swb_extract.features.inhouse.pronoun_per_second import count_personal_pronouns  # noqa: E402
from swb_extract.features.thomas2018._terms import terms  # noqa: E402
from swb_extract.transcripts import parse_transcript  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mv", HERE / "misc_variables.py"); mv = importlib.util.module_from_spec(spec); spec.loader.exec_module(mv)
TR = Path("swb_ms98_transcriptions_cleaned"); TRACKS = Path("utterances_v2/derived/opensmile"); FRAME = 0.01
V = ["ppron", "wps", "wpu", "wpp", "boplen", "poplen", "pv", "lv", "olap", "rept", "repu"]


def lines_of(conv: int, side: str) -> pd.DataFrame | None:
    """Lines with SPEECH-TIGHT bounds (first-word onset → last-word offset, the analogue of an STT
    segment's span) as start/end, and the padded ms98 line bounds as start_trans/end_trans."""
    tp = TR / f"{conv:04d}"[:2] / f"{conv:04d}" / f"sw{conv:04d}{side}-ms98-a-trans.text"
    if not tp.is_file():
        return None
    u = list(parse_transcript(tp))
    bounds: dict[int, list[float]] = {}
    wp = tp.with_name(tp.name.replace("-trans.text", "-word.text"))
    if wp.is_file():
        for line in open(wp, encoding="utf-8"):
            parts = line.split(maxsplit=3)
            if len(parts) < 4:
                continue
            try:
                utt = int(parts[0].rsplit("-", 1)[1]); s, e = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            b = bounds.setdefault(utt, [s, e]); b[0] = min(b[0], s); b[1] = max(b[1], e)
    d = pd.DataFrame({"start_trans": [x.start for x in u], "end_trans": [x.end for x in u], "transcript": [x.text for x in u]})
    d["start"] = [bounds.get(x.utt_num, [x.start, x.end])[0] for x in u]
    d["end"] = [bounds.get(x.utt_num, [x.start, x.end])[1] for x in u]
    return d


def side_row(conv: int, side: str, own: pd.DataFrame, partner: pd.DataFrame) -> dict | None:
    npz = TRACKS / f"sw{conv:05d}.{side}.npz"
    if not npz.is_file():
        return None
    z = np.load(npz); f0, loud = z["f0"], z["loudness"]; n = len(f0)
    toks = [tokenize(x) for x in own.transcript]; words = np.array([len(w) for w in toks]); dur = (own.end - own.start).to_numpy()
    prons = sum(count_personal_pronouns(x) for x in own.transcript)
    # pauses inside lines, F0==0 runs bounded by voiced frames (10 ms frames from the recording start)
    n_p = 0; sec_p = 0.0; inl = np.zeros(n, bool)
    for s, e in zip(own.start, own.end):
        i0, i1 = min(int(s / FRAME), n), min(int(e / FRAME), n); inl[i0:i1] = True
        idx = np.flatnonzero(f0[i0:i1] > 0)
        if idx.size < 2:
            continue
        g = np.diff(idx) - 1; g = g[g > 0]; n_p += int(g.size); sec_p += float(g.sum() * FRAME)
    voiced = f0 > 0; vin = voiced & inl
    ol, pop, _naive = mv.onset_timing(own, partner)
    ol_t, pop_t, _ = mv.onset_timing(own.rename(columns={"start": "s", "end": "e", "start_trans": "start", "end_trans": "end"}),
                                     partner.rename(columns={"start": "s", "end": "e", "start_trans": "start", "end_trans": "end"}))
    pops_t = [x for x in pop_t if x is not None]; dur_t = (own.end_trans - own.start_trans).to_numpy()
    ts = [terms(x) for x in own.transcript]; reps = [len(ts[i] & ts[i - 1]) for i in range(1, len(ts))]
    pops = [x for x in pop if x is not None]
    fv, lv_ = f0[voiced].astype(float), loud[voiced].astype(float); fvi, lvi = f0[vin].astype(float), loud[vin].astype(float)
    return dict(
        conv=conv, side=side, n_utts=len(own), words=int(words.sum()), talk_sec=float(dur.sum()), prons=int(prons),
        n_pauses=n_p, pause_sec=sec_p, n_voiced=int(voiced.sum()), f0_sum=float(fv.sum()), f0_sq=float((fv ** 2).sum()),
        ld_sum=float(lv_.sum()), ld_sq=float((lv_ ** 2).sum()), n_voiced_in=int(vin.sum()), f0i_sum=float(fvi.sum()), f0i_sq=float((fvi ** 2).sum()),
        ldi_sum=float(lvi.sum()), ldi_sq=float((lvi ** 2).sum()), olap_n=int(sum(ol)), pop_sum=float(sum(pops)), pop_n=len(pops),
        rept_sum=int(sum(reps)), repu_sum=int(sum(r >= 1 for r in reps)), rep_n=len(reps),
        ppron=prons / words.sum() if words.sum() else np.nan, wps=words.sum() / dur.sum() if dur.sum() > 0 else np.nan, wpu=float(words.mean()),
        wpp=words.sum() / n_p if n_p else np.nan, boplen=sec_p / n_p if n_p else np.nan, poplen=float(np.mean(pops)) if pops else np.nan,
        pv=float(fv.var()) if fv.size > 1 else np.nan, lv=float(lv_.var()) if lv_.size > 1 else np.nan,
        pv_inlines=float(fvi.var()) if fvi.size > 1 else np.nan, lv_inlines=float(lvi.var()) if lvi.size > 1 else np.nan,
        olap=float(np.mean(ol)) if ol else np.nan, rept=float(np.mean(reps)) if reps else np.nan, repu=float(np.mean([r >= 1 for r in reps])) if reps else np.nan,
        wps_trans=words.sum() / dur_t.sum() if dur_t.sum() > 0 else np.nan, olap_trans=float(np.mean(ol_t)) if ol_t else np.nan,
        poplen_trans=float(np.mean(pops_t)) if pops_t else np.nan,
    )


def conv_rows(conv: int) -> list[dict]:
    a, b = lines_of(conv, "A"), lines_of(conv, "B")
    if a is None or b is None:
        return []
    return [r for r in (side_row(conv, "A", a, b), side_row(conv, "B", b, a)) if r is not None]


def pooled_var(n, s, sq):
    return np.where(n > 1, sq / n - (s / n) ** 2, np.nan)


def main(limit: int = 0) -> int:
    convs = sorted({int(p.name[2:7]) for p in TRACKS.glob("sw*.npz")})
    if limit:
        convs = convs[:limit]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for r in ex.map(conv_rows, convs, chunksize=16):
            rows.extend(r)
    S = pd.DataFrame(rows); S["side_id"] = "sw" + S.conv.astype(str).str.zfill(4) + S.side
    S.to_csv(HERE / "swb_misc_route_sides.csv", index=False)
    con = pd.read_csv("tables/call_con_tab.csv", header=None, skipinitialspace=True, quotechar='"').iloc[:, :3]; con.columns = ["conv", "cside", "caller"]
    con["side_id"] = "sw" + con.conv.astype(str) + con.cside.str.strip(' "'); S["caller"] = S.side_id.map(con.set_index("side_id").caller)
    g = S.dropna(subset=["caller"]).groupby("caller")
    C = g[["n_utts", "words", "talk_sec", "prons", "n_pauses", "pause_sec", "n_voiced", "f0_sum", "f0_sq", "ld_sum", "ld_sq", "olap_n", "pop_sum", "pop_n", "rept_sum", "repu_sum", "rep_n"]].sum()
    C["n_calls"] = g.conv.nunique()
    C["ppron"] = C.prons / C.words; C["wps"] = C.words / C.talk_sec; C["wpu"] = C.words / C.n_utts; C["wpp"] = C.words / C.n_pauses; C["boplen"] = C.pause_sec / C.n_pauses
    C["poplen"] = C.pop_sum / C.pop_n; C["pv"] = pooled_var(C.n_voiced, C.f0_sum, C.f0_sq); C["lv"] = pooled_var(C.n_voiced, C.ld_sum, C.ld_sq)
    C["olap"] = C.olap_n / C.n_utts; C["rept"] = C.rept_sum / C.rep_n; C["repu"] = C.repu_sum / C.rep_n
    C.reset_index().to_csv(HERE / "swb_misc_route_callers.csv", index=False)
    print(f"sides {len(S):,} ({len(convs)} convs), callers {len(C):,}; side medians:", S[V].median().round(3).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0))
