"""Thomas et al. (2018) §3.2–§4.1, copied step for step and applied to Switchboard.

Replicates the reliability + PCA + "involvement" construction of Thomas, Czerwinski,
McDuff, Craswell & Mark, "Style and Alignment in Information-Seeking Conversation"
(CHIIR '18) on our extraction of their eleven variables
(``src/swb_extract/features/thomas2018/``) at their unit — participant × task, which on
Switchboard is one conversation side. Each step quotes the sentence it copies.

 1. Unit table — ``utterances_v2/derived/thomas2018_side.csv`` (``swb-extract
    thomas2018-side``): the eleven pooled per side exactly as §3.2 defines them.
 2. Cleaning (§3.4) — "We examined all apparent outliers, across all the variables …
    and removed a small number of cases." Theirs was manual; here a stated rule: sides
    missing any of the eleven are dropped (their footnote 4: wpu/boplen undefined without
    utterances/pauses), sides with fewer than MIN_UTTS utterances are dropped (the
    project's floor stands in for their ten-minute task), and sides with |z| > OUTLIER_Z
    on any variable are removed. Counts reported; the sensitivity block re-runs without
    the last two rules.
 3. Scaling (§3.2) — "they are each rescaled to zero mean, unit standard deviation.
    Inspection confirmed that each variable is approximately normal, although some have
    long tails." Skewness and excess kurtosis per variable are reported instead of
    inspection.
 4. Reliability (§4.1) — "Revelle's GLB is 0.85 over all eleven variables … (corresponding
    Cronbach's α = 0.67)." α on the raw-signed variables (as a paper that does not mention
    keying would have computed it) AND keyed by the Table 2 signs; mean inter-item
    correlation r̄; α-if-item-deleted. Revelle's glb.fa is an R routine; McDonald's ω_t from
    a one-factor principal-axis fit is reported as the congeneric companion — a stated
    substitution, not a reproduction of the GLB.
 5. PCA (§4.1) — "a principal components analysis and scree plot suggested one more
    important component, explaining 29% of variance, and 2–3 less important components
    each explaining 14% or less." PCA of the correlation matrix of the z-scored eleven;
    all variance shares; PC1 as the unit-norm eigenvector (their Table 2 weights' squares
    sum to 1.00), oriented so that wps is positive. Horn's parallel analysis added.
 6. Involvement — "the involvement shown by a participant, in a task, is the sum of the
    eleven variables above weighted by their loadings on the first principal component."
    Per side, standardized; plus the same sum under THEIR Table 2 weights.
 7. Comparison with Table 2 — sign agreement, Tucker's congruence φ (cosine of the two
    loading vectors), bootstrap 95% CIs of our loadings (B over sides), and a sampling
    yardstick: φ between PC1 of random n=168 subsamples (their PCA n) and the full-data PC1.

Outputs (this folder): thomas_method_sides.csv, thomas_method_loadings.csv,
thomas_method_results.csv. Run from the repo root:
    PYTHONPATH=src python3 analysis/c_alpha/thomas_method.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

sys.path.insert(0, "src")
from swb_extract.features.thomas2018 import PAPER_LOADINGS, VARIABLES  # noqa: E402

HERE = Path(__file__).resolve().parent
SIDE_TABLE = Path("utterances_v2/derived/thomas2018_side.csv")
MIN_UTTS = 20
OUTLIER_Z = 5.0
B_BOOT = 500
N_THOMAS = 168          # observations behind their PCA/GLB
N_THOMAS_CLEAN = 98     # participant-tasks after their cleaning
SEED = 0
KEY_NEGATIVE = ("boplen", "poplen", "olap")   # Table 2's negative signs


def zscore(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0)) / X.std(0, ddof=1)


def cronbach(Z: np.ndarray) -> tuple[float, float]:
    """(alpha, mean inter-item r) on standardized items."""
    k = Z.shape[1]
    R = np.corrcoef(Z, rowvar=False)
    r_bar = R[np.triu_indices(k, 1)].mean()
    return k * r_bar / (1 + (k - 1) * r_bar), r_bar


def omega_1factor(R: np.ndarray, iters: int = 100) -> float:
    """McDonald's omega_total from a one-factor principal-axis fit (iterated communalities)."""
    k = R.shape[0]
    inv = np.linalg.pinv(R)
    h2 = 1 - 1 / np.diag(inv)                       # squared multiple correlations
    for _ in range(iters):
        Rr = R.copy(); np.fill_diagonal(Rr, h2)
        w, v = np.linalg.eigh(Rr)
        lam = np.sqrt(max(w[-1], 0)) * v[:, -1]
        h2_new = np.clip(lam ** 2, 0, 0.999)
        if np.max(np.abs(h2_new - h2)) < 1e-7:
            h2 = h2_new; break
        h2 = h2_new
    lam = np.abs(lam)                               # same-sign loadings for the sum
    return float(lam.sum() ** 2 / (lam.sum() ** 2 + (1 - h2).sum()))


def pc1(Z: np.ndarray, orient_col: int) -> tuple[np.ndarray, np.ndarray]:
    """(unit-norm PC1 eigenvector oriented so Z[:, orient_col] loads positive, eigenvalues desc)."""
    R = np.corrcoef(Z, rowvar=False)
    w, v = np.linalg.eigh(R)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    vec = v[:, 0]
    if vec[orient_col] < 0:
        vec = -vec
    return vec, w


def congruence(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def horn_k(Z: np.ndarray, n_rand: int = 200, rng=None) -> int:
    rng = rng or np.random.default_rng(SEED)
    n, k = Z.shape
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1]
    rand = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, k)), rowvar=False)))[::-1]
                     for _ in range(n_rand)])
    thresh = np.percentile(rand, 95, axis=0)
    return int(np.sum(real > thresh))


def main(table: Path = SIDE_TABLE, out_prefix: str = "thomas_method", id_cols=("call_id", "side"),
         min_utts: int = MIN_UTTS, outlier_z: float = OUTLIER_Z) -> int:
    """Run the method on any participant-task table with columns id_cols + n_utts + the eleven.

    CLI: thomas_method.py [table.csv] [out_prefix] [id_cols,comma,separated] [min_utts] [outlier_z]
    """
    if not table.is_file():
        sys.exit(f"missing {table} — run the eleven extractors and `swb-extract thomas2018-side` first")
    raw = pd.read_csv(table)
    id_cols = list(id_cols)
    global MIN_UTTS, OUTLIER_Z
    MIN_UTTS, OUTLIER_Z = min_utts, outlier_z
    V = list(VARIABLES)
    t_weights = np.array([PAPER_LOADINGS[v] for v in V])
    results: list[tuple[str, object]] = [("n_sides_in_table", len(raw))]

    # --- 2. cleaning
    d = raw.dropna(subset=V).copy(); results.append(("n_after_dropping_missing", len(d)))
    d = d[d.n_utts >= MIN_UTTS].copy(); results.append((f"n_after_min_utts_{MIN_UTTS}", len(d)))
    Z0 = zscore(d[V].to_numpy(float))
    keep = (np.abs(Z0) <= OUTLIER_Z).all(axis=1)
    results.append((f"n_removed_outliers_|z|>{OUTLIER_Z:g}", int((~keep).sum())))
    d = d[keep].copy(); results.append(("n_sides_analysed", len(d)))

    # --- 3. scaling + shape
    X = d[V].to_numpy(float); Z = zscore(X)
    shape = pd.DataFrame({"variable": V, "mean": X.mean(0), "sd": X.std(0, ddof=1),
                          "skew": skew(X, axis=0), "excess_kurtosis": kurtosis(X, axis=0)})

    # --- 4. reliability
    a_raw, r_raw = cronbach(Z)
    Zk = Z.copy()
    for v in KEY_NEGATIVE:
        Zk[:, V.index(v)] *= -1
    a_key, r_key = cronbach(Zk)
    results += [("alpha_raw_signs", round(a_raw, 4)), ("mean_r_raw_signs", round(r_raw, 4)),
                ("alpha_keyed_table2_signs", round(a_key, 4)), ("mean_r_keyed", round(r_key, 4)),
                ("omega_t_1factor_keyed", round(omega_1factor(np.corrcoef(Zk, rowvar=False)), 4)),
                ("thomas_alpha", 0.67), ("thomas_GLB", 0.85), ("thomas_implied_mean_r", round(0.67 / (11 - 10 * 0.67), 4))]
    alpha_del = [cronbach(np.delete(Zk, j, axis=1))[0] for j in range(len(V))]

    # --- 5. PCA + Horn
    vec, eig = pc1(Z, V.index("wps"))
    shares = eig / eig.sum()
    for i, s_ in enumerate(shares[:5], 1):
        results.append((f"pc{i}_variance_share", round(float(s_), 4)))
    results += [("thomas_pc1_variance_share", 0.29), ("horn_K", horn_k(Z))]

    # --- 6. involvement (ours / their weights)
    u_ours = Z @ vec; u_theirs = Z @ t_weights
    u_ours_z = (u_ours - u_ours.mean()) / u_ours.std(ddof=1); u_theirs_z = (u_theirs - u_theirs.mean()) / u_theirs.std(ddof=1)
    results += [("corr_involvement_ours_vs_thomas_weights", round(float(np.corrcoef(u_ours, u_theirs)[0, 1]), 4)),
                ("involvement_skew", round(float(skew(u_ours)), 3)), ("involvement_excess_kurtosis", round(float(kurtosis(u_ours)), 3))]

    # --- 7. comparison with Table 2
    rng = np.random.default_rng(SEED)
    n = Z.shape[0]
    boots = np.array([pc1(Z[rng.integers(0, n, n)], V.index("wps"))[0] for _ in range(B_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    phi = congruence(vec, t_weights)
    signs = int(np.sum(np.sign(vec) == np.sign(t_weights)))
    yard = {}
    for m in (N_THOMAS, N_THOMAS_CLEAN):
        if m >= n:
            results.append((f"yardstick_phi_subsample_n{m}", "n/a (sample not larger than m)")); continue
        phis = [congruence(pc1(Z[rng.choice(n, m, replace=False)], V.index("wps"))[0], vec) for _ in range(B_BOOT)]
        yard[m] = np.percentile(phis, [5, 50, 95])
        results += [(f"yardstick_phi_subsample_n{m}_q05", round(float(yard[m][0]), 4)),
                    (f"yardstick_phi_subsample_n{m}_median", round(float(yard[m][1]), 4))]
    results += [("congruence_phi_ours_vs_thomas_pc1", round(phi, 4)), ("sign_agreement_k_of_11", signs),
                ("phi_within_thomas_sampling_error", bool(phi >= yard[N_THOMAS][0]) if N_THOMAS in yard else "n/a")]

    # --- sensitivity: no outlier rule / no utterance floor
    for label, sub in (("no_outlier_rule", raw.dropna(subset=V)[lambda x: x.n_utts >= MIN_UTTS]),
                       ("no_min_utts", raw.dropna(subset=V))):
        Zs = zscore(sub[V].to_numpy(float)); Zsk = Zs.copy()
        for v in KEY_NEGATIVE:
            Zsk[:, V.index(v)] *= -1
        vs, es = pc1(Zs, V.index("wps"))
        results += [(f"sens_{label}_n", len(sub)), (f"sens_{label}_alpha_keyed", round(cronbach(Zsk)[0], 4)),
                    (f"sens_{label}_pc1_share", round(float(es[0] / es.sum()), 4)),
                    (f"sens_{label}_phi", round(congruence(vs, t_weights), 4))]

    # --- write
    sides = d[id_cols + ["n_utts"] + V].copy()
    for j, v in enumerate(V):
        sides[f"z_{v}"] = Z[:, j]
    sides["involvement_ours"] = u_ours_z; sides["involvement_thomas_weights"] = u_theirs_z
    sides["involvement_pct"] = sides.involvement_ours.rank() / (len(sides) + 1)
    sides.to_csv(HERE / f"{out_prefix}_sides.csv", index=False)
    load = pd.DataFrame({"variable": V, "thomas_table2_weight": t_weights, "our_pc1_weight": vec.round(4),
                         "boot_ci_lo": lo.round(4), "boot_ci_hi": hi.round(4),
                         "sign_match": np.sign(vec) == np.sign(t_weights),
                         "alpha_if_deleted_keyed": np.round(alpha_del, 4)}).merge(shape, on="variable")
    load.to_csv(HERE / f"{out_prefix}_loadings.csv", index=False)
    pd.DataFrame(results, columns=["key", "value"]).to_csv(HERE / f"{out_prefix}_results.csv", index=False)
    print(load.to_string(index=False))
    print("\n" + "\n".join(f"{k:48s} {v}" for k, v in results))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(
        table=Path(a[0]) if a else SIDE_TABLE,
        out_prefix=a[1] if len(a) > 1 else "thomas_method",
        id_cols=a[2].split(",") if len(a) > 2 else ("call_id", "side"),
        min_utts=int(a[3]) if len(a) > 3 else MIN_UTTS,
        outlier_z=float(a[4]) if len(a) > 4 else OUTLIER_Z,
    ))
