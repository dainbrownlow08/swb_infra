"""Like-for-like reliability runs: MISC vs Switchboard through the SAME eleven-variable route.

Inputs (built upstream):
  misc_variables_raw.csv            MISC, the tracks Thomas et al. shipped          (misc_variables.py)
  misc_variables_raw_v3tracks.csv   MISC, our openSMILE-3 prosodyShs re-tracking    (misc_variables.py --tracks v3)
  swb_misc_route_sides.csv          Switchboard sides through the MISC route        (swb_misc_route.py)
  swb_misc_route_callers.csv        the same pooled per caller                      (swb_misc_route.py)

Samples: MISC 168 = tasks 2-5 minus the paper's five outlier tasks minus P22's three no-question tasks
(identified in misc_identify.py); MISC 98 = those with audio <= 600 s (97 before the |z|>5 rule).
Switchboard variants: speech-tight line spans (primary), `trans` = the padded transcript spans,
`inlines` = pitch/loudness variation restricted to the speaker's own lines instead of the whole channel.

Run from the repo root:  PYTHONPATH=src python3 analysis/c_alpha/like_for_like.py
"""
import glob, importlib.util, re, subprocess, sys, wave
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tm", HERE / "thomas_method.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm); tm.HERE = HERE
OUTLIERS = {(2, "transport"), (5, "migraine"), (5, "olympics"), (17, "transport"), (18, "transport")}
NO_Q = {(22, 2), (22, 3), (22, 5)}
MAX_AUDIO_S = 600


def misc_duration_s() -> dict:
    out = {}
    for f in glob.glob("corpus/misc/MISC_Public_Release/Audio/*/P*_T*.wav"):
        m = re.search(r"P(\d+)_T(\d)", f)
        with wave.open(f) as w:
            out[(int(m.group(1)), int(m.group(2)))] = w.getnframes() / w.getframerate()
    return out


def score(table: pd.DataFrame, prefix: str, id_cols, min_utts: int, outlier_z: float = 5.0) -> None:
    path = HERE / f"{prefix}_table.csv"
    table.to_csv(path, index=False)
    tm.main(table=path, out_prefix=prefix, id_cols=tuple(id_cols), min_utts=min_utts, outlier_z=outlier_z)


def main() -> int:
    dur = misc_duration_s()
    for tag, name in (("", "shipped"), ("_v3tracks", "v3")):
        d = pd.read_csv(HERE / f"misc_variables_raw{tag}.csv")
        d = d[d.task >= 2]
        keep = [not ((p, tn) in OUTLIERS or (p, t) in NO_Q) for p, tn, t in zip(d.participant, d.task_name, d.task)]
        s168 = d[keep]
        s98 = s168[[dur[(p, t)] <= MAX_AUDIO_S for p, t in zip(s168.participant, s168.task)]]
        assert (len(s168), len(s98)) == (168, 97), (len(s168), len(s98))  # the identified samples (misc_identify.py)
        ids = ("pair", "participant", "task", "task_name")
        score(s168, f"misc_{name}_168", ids, min_utts=0)
        score(s98, f"misc_{name}_98", ids, min_utts=0)

    sides = pd.read_csv(HERE / "swb_misc_route_sides.csv")
    ids = ("conv", "side", "side_id")
    score(sides, "swb_os_sides", ids, min_utts=tm.MIN_UTTS)
    trans = sides.copy()
    for v in ("wps", "olap", "poplen"):
        trans[v] = sides[f"{v}_trans"]
    score(trans, "swb_os_sides_trans", ids, min_utts=tm.MIN_UTTS)
    inl = sides.copy()
    for v in ("pv", "lv"):
        inl[v] = sides[f"{v}_inlines"]
    score(inl, "swb_os_sides_inlines", ids, min_utts=tm.MIN_UTTS)
    callers = pd.read_csv(HERE / "swb_misc_route_callers.csv")
    score(callers, "swb_os_callers", ("caller", "n_calls"), min_utts=tm.MIN_UTTS)
    subprocess.run([sys.executable, str(HERE / "compare_routes.py")], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
