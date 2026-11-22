"""Length-fair ingredients for rept / repu — is Thomas et al.'s re-statement pair utterance length?

rept ("mean number of terms repeated from the same person's previous utterance") and repu
("fraction of utterances with at least one repeated term") are COUNTS of shared terms, so a
longer line shares more terms with its predecessor whether or not the speaker is re-stating
anything. This script rebuilds both from the transcripts (asserting it reproduces the
published tables exactly) and writes, per unit — a Switchboard conversation side, a MISC
participant × task — the ingredients nb_AJ §9 needs to separate re-statement from length:

  n_pairs        utterances with an own previous line (the rows rept / repu average over)
  terms_per_utt  mean content terms per such utterance (stopwords + fillers out, stemmed types)
  rept, repu     the paper's two variables, recomputed here (checked against the tables)
  rept_per_term  Σ shared terms / Σ own terms — the share of an utterance's content terms that
                 repeat the previous line (length of the CURRENT line divided out)
  rept_jaccard   Σ shared / Σ union — both lines' lengths divided out
  rept_chance, repu_chance
                 the same two statistics for a RANDOM pair of the unit's own lines: the exact
                 mean over all unordered pairs. What the unit's line lengths and vocabulary
                 produce with adjacency removed — no "previous utterance" in it at all.
  rept_shuf, repu_shuf
                 one realised re-pairing (seed 0): each utterance keeps its own terms and is
                 scored against a uniformly drawn OTHER own line instead of its predecessor —
                 the like-for-like null, with the sampling noise the real variable has.

``rept − rept_chance`` (formed in the notebook) is adjacency-specific re-statement: what the
previous line adds beyond the unit's lengths and recurring vocabulary. All ratios are ratios of
sums over the unit, never a mean of per-utterance ratios (the thomas2018 package's convention).

Run from the repo root:  PYTHONPATH=src python3 analysis/c_alpha/restatement_length.py
Outputs (here): restatement_length_swb.csv (call_id, side, …), restatement_length_misc.csv
(pair, participant, task, …; tasks 2–5, the rows of misc_variables_all.csv).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from swb_extract.features.inhouse._turn_index import build_text_index  # noqa: E402
from swb_extract.features.thomas2018._io import iter_manifest, key_of  # noqa: E402
from swb_extract.features.thomas2018._terms import terms  # noqa: E402
from swb_extract.manifest import manifest_path  # noqa: E402

import misc_variables as mv  # noqa: E402  (MISC line loader + release path; same cwd contract)

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path("utterances_v2")
TRANSCRIPT_ROOT = Path("swb_ms98_transcriptions_cleaned")
SIDE_TABLE = OUT_ROOT / "derived" / "thomas2018_side.csv"
MISC_TABLE = HERE / "misc_variables_all.csv"
SEED = 0
COLS = ("n_pairs", "terms_per_utt", "rept", "repu", "rept_per_term", "rept_jaccard",
        "rept_chance", "repu_chance", "rept_shuf", "repu_shuf")


def pair_gram(pool: list[set[str]]) -> np.ndarray:
    """n × n matrix of shared-term counts between every two lines of a unit (diagonal = own size)."""
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for i, t in enumerate(pool):
        for w in t:
            rows.append(i); cols.append(vocab.setdefault(w, len(vocab)))
    m = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(pool), max(len(vocab), 1)))
    return (m @ m.T).toarray()


def unit_stats(pool: list[set[str]], pairs: list[tuple[int, int]], rng: np.random.Generator) -> dict[str, float]:
    """The COLS for one unit. ``pool`` = term sets of all the unit's own lines, in order;
    ``pairs`` = (current, previous) indices into pool for the utterances rept / repu average over."""
    nan = dict.fromkeys(COLS, np.nan) | {"n_pairs": len(pairs)}
    if not pairs:
        return nan
    g = pair_gram(pool)
    cur, prev = np.array(pairs).T
    shared = g[cur, prev]; own = g[cur, cur]; union = own + g[prev, prev] - shared
    out = dict(n_pairs=len(pairs), terms_per_utt=float(own.mean()), rept=float(shared.mean()), repu=float((shared >= 1).mean()),
               rept_per_term=float(shared.sum() / own.sum()) if own.sum() else np.nan,
               rept_jaccard=float(shared.sum() / union.sum()) if union.sum() else np.nan)
    n = len(pool)
    if n < 3:                                           # a re-pairing needs a line that is neither current nor itself
        return out | {k: np.nan for k in ("rept_chance", "repu_chance", "rept_shuf", "repu_shuf")}
    off = g[np.triu_indices(n, 1)]                      # every unordered pair of distinct own lines
    other = rng.integers(0, n - 1, len(cur)); other += other >= cur     # uniform over the n − 1 lines that are not current
    return out | dict(rept_chance=float(off.mean()), repu_chance=float((off >= 1).mean()),
                      rept_shuf=float(g[cur, other].mean()), repu_shuf=float((g[cur, other] >= 1).mean()))


def switchboard() -> pd.DataFrame:
    """One row per conversation side in the manifest. Mirrors ``_terms.build_repeat_index``: a side's
    lines are the cleaned-transcript lines ordered by utt_num; an utterance's previous line is the
    preceding transcript line; rept / repu average over the MANIFEST utterances that have one."""
    text_idx = build_text_index(TRANSCRIPT_ROOT)
    lines: dict[tuple[int, str], list[int]] = {}
    for call_id, side, utt_num in text_idx:
        lines.setdefault((call_id, side), []).append(utt_num)
    in_manifest: dict[tuple[int, str], set[int]] = {}
    for rel, _text in iter_manifest(manifest_path(OUT_ROOT)):
        key = key_of(rel)
        if key is not None:
            in_manifest.setdefault((key[0], key[1]), set()).add(key[2])
    rng, rows = np.random.default_rng(SEED), []
    for (call_id, side), wanted in in_manifest.items():                 # manifest order: sides are contiguous
        utts = sorted(lines.get((call_id, side), []))
        pool = [terms(text_idx[(call_id, side, u)]) for u in utts]
        pairs = [(i, i - 1) for i, u in enumerate(utts) if i > 0 and u in wanted]
        rows.append({"call_id": call_id, "side": side, **unit_stats(pool, pairs, rng)})
    return pd.DataFrame(rows)


def misc() -> pd.DataFrame:
    """One row per MISC participant × task (tasks 2–5), lines as ``misc_variables.variables`` orders them."""
    table = pd.read_csv(MISC_TABLE)
    rng, rows = np.random.default_rng(SEED), []
    for pair, p, t in zip(table["pair"], table["participant"], table["task"]):
        pool = [terms(x) for x in mv.load_lines(pair, int(p), int(t)).transcript]
        rows.append({"pair": pair, "participant": int(p), "task": int(t),
                     **unit_stats(pool, [(i, i - 1) for i in range(1, len(pool))], rng)})
    return pd.DataFrame(rows)


def check(new: pd.DataFrame, table: pd.DataFrame, keys: list[str], label: str) -> None:
    """The recomputed rept / repu must BE the published ones — otherwise the ingredients describe another variable."""
    m = table.merge(new, on=keys, suffixes=("_table", ""), validate="one_to_one")
    for v in ("rept", "repu"):
        both = m[[f"{v}_table", v]].dropna()
        gap = float((both[f"{v}_table"] - both[v]).abs().max())
        assert len(both) and gap < 1e-9, f"{label}: recomputed {v} differs from the table by up to {gap:.3g}"
        print(f"  {label}: {v} reproduces the table on {len(both):,} units (max |difference| {gap:.1e})")


def main() -> int:
    swb = switchboard()
    check(swb, pd.read_csv(SIDE_TABLE), ["call_id", "side"], "Switchboard sides")
    swb.to_csv(HERE / "restatement_length_swb.csv", index=False)
    mi = misc()
    check(mi, pd.read_csv(MISC_TABLE), ["pair", "participant", "task"], "MISC participant-tasks")
    mi.to_csv(HERE / "restatement_length_misc.csv", index=False)
    for name, d in (("Switchboard sides", swb), ("MISC participant-tasks", mi)):
        print(f"{name}: {len(d):,} units · medians " + " · ".join(f"{c} {d[c].median():.3f}" for c in COLS[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
