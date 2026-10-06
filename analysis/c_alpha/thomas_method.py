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
    correlation r̄; α-if-item-deleted. The GLB is the paper's footnote 2 routine, psych
    1.7.3's ``glb.fa`` (Revelle 2017), ported line for line from the CRAN source
    (``glb_fa``: a one-factor minres fit fixes the factor count as the number of positive
    eigenvalues of the reduced matrix, degrees of freedom permitting; an nf-factor minres
    fit's communalities replace the diagonal; GLB = sum of that matrix / sum of R). Beside
    it, the bound that routine approximates — the algebraic GLB of Jackson & Agunwamba
    (1977), psych's ``glb.algebraic`` — solved exactly as the semidefinite programme it is
    (``glb_algebraic``), and McDonald's ω_t from a one-factor principal-axis fit.
 5. PCA (§4.1) — "a principal components analysis and scree plot suggested one more
    important component, explaining 29% of variance, and 2–3 less important components
    each explaining 14% or less." PCA of the correlation matrix of the z-scored eleven;
    all variance shares; PC1 as the unit-norm eigenvector (their Table 2 weights' squares
    sum to 1.00), oriented so that wps is positive. Horn's parallel analysis added, and
    beside it (2026-10-06, nb_AJ §8) Velicer's MAP (``velicer_map``), the empirical Kaiser
    criterion (``ekc``) and Ruscio & Roche's comparison data (``comparison_data``).
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
from scipy.optimize import minimize
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


def smc(R: np.ndarray) -> np.ndarray:
    """Squared multiple correlations, psych's smc(): 1 - 1/diag(R^-1)."""
    return 1 - 1 / np.diag(np.linalg.inv(R))


def minres_fa(R: np.ndarray, nf: int, exact: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """psych 1.7.3.21 fa(r, nf, fm="minres"), ported from fa.R's fit()/fit.residuals/FAout.

    Uniquenesses by L-BFGS-B (start 1 - smc, bounds [.005, 1], optim's parscale = .01 and
    factr = 1e7, pgtol = 0, maxit = 100) on the squared below-diagonal residuals of the
    nf-eigenvector fit to R with 1 - psi on the diagonal, using psych's own (ML-form)
    gradient; loadings then by FAout (eigen of psi^-1/2 R psi^-1/2). Returns (loadings,
    communality = diag(LL'), values = eigenvalues of R with the communalities on the diagonal).
    ``exact=True`` drops psych's approximate gradient for finite differences with tight
    tolerances — the true minres optimum, the sensitivity companion to the port.
    """
    k = R.shape[0]
    eps = np.finfo(float).eps
    scale = 0.01                                        # optim(parscale = rep(.01, k)) works on psi / .01

    def top(w, v, n):                                   # R's eigen() order: decreasing
        idx = np.argsort(w)[::-1][:n]
        return w[idx], v[:, idx]

    def resid(x):                                       # fit.residuals, fm = "minres"
        S = R.copy(); np.fill_diagonal(S, 1 - x * scale)
        w, v = np.linalg.eigh(S)
        w = np.where(w < eps, 100 * eps, w)
        w, v = top(w, v, nf)
        L = v * np.sqrt(w)
        res = S - L @ L.T
        return float((res[np.tril_indices(k, -1)] ** 2).sum())

    def faout(psi):                                     # FAout / FAgr.minres2 share this
        sc = 1 / np.sqrt(psi)
        w, v = np.linalg.eigh(sc[:, None] * R * sc[None, :])
        w, v = top(w, v, nf)
        return np.sqrt(psi)[:, None] * (v * np.sqrt(np.maximum(w - 1, 0)))

    def grad(x):
        psi = x * scale
        L = faout(psi)
        g = L @ L.T + np.diag(psi) - R
        return np.diag(g) / psi ** 2 * scale

    start = (1 - smc(R)) / scale
    if exact:
        res = minimize(resid, start, method="L-BFGS-B", bounds=[(0.005 / scale, 1 / scale)] * k,
                       options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 5000})
    else:
        res = minimize(resid, start, jac=grad, method="L-BFGS-B", bounds=[(0.005 / scale, 1 / scale)] * k,
                       options={"ftol": 1e7 * eps, "gtol": 0.0, "maxiter": 100, "maxcor": 5})
    psi = res.x * scale
    L = faout(psi)
    h2 = (L ** 2).sum(1)
    S = R.copy(); np.fill_diagonal(S, h2)
    return L, h2, np.sort(np.linalg.eigvalsh(S))[::-1]


def glb_fa(R: np.ndarray, exact: bool = False) -> tuple[float, int]:
    """psych 1.7.3.21 glb.fa(r) — the paper's footnote-2 GLB — ported from glbs.R. (glb, nf).

    nf = number of positive eigenvalues of the reduced matrix from a one-factor minres fit,
    minus one if the nf-factor model has negative degrees of freedom (checked once, as in
    the source); GLB = sum(R with the nf-factor communalities on the diagonal) / sum(R).
    Key the items (D R D) before calling: communalities are sign-invariant, sum(R) is not.
    """
    k = R.shape[0]
    values = minres_fa(R, 1, exact)[2]
    nf = int((values > 0).sum())
    if k * (k - 1) / 2 - nf * k + nf * (nf - 1) / 2 < 0:
        nf -= 1
    h2 = minres_fa(R, nf, exact)[1]
    rr = R.copy(); np.fill_diagonal(rr, h2)
    return float(rr.sum() / R.sum()), nf


def glb_algebraic(R: np.ndarray, tol: float = 1e-10) -> tuple[float, np.ndarray]:
    """The greatest lower bound itself (Jackson & Agunwamba 1977; psych's glb.algebraic):
    max Σψ subject to R − diag(ψ) ⪰ 0 and 0 ≤ ψ ≤ diag(R); GLB = 1 − Σψ / sum(R).
    A semidefinite programme (Rcsdp in psych); here a log-barrier Newton method. (glb, ψ).
    """
    k = R.shape[0]
    up = np.diag(R).astype(float)
    psi = np.full(k, 0.5 * min(np.linalg.eigvalsh(R).min(), up.min()))   # strictly interior

    def f(psi, mu):
        M = R - np.diag(psi)
        w = np.linalg.eigvalsh(M)
        if w.min() <= 0 or psi.min() <= 0 or (up - psi).min() <= 0:
            return np.inf
        return -psi.sum() - mu * (np.log(w).sum() + np.log(psi).sum() + np.log(up - psi).sum())

    mu = 1.0
    while True:
        for _ in range(100):                            # Newton steps at this mu
            Minv = np.linalg.inv(R - np.diag(psi))
            g = -1 + mu * (np.diag(Minv) - 1 / psi + 1 / (up - psi))
            H = mu * (Minv ** 2 + np.diag(1 / psi ** 2 + 1 / (up - psi) ** 2))
            step = -np.linalg.solve(H, g)
            if -g @ step / 2 < tol:
                break
            t, f0 = 1.0, f(psi, mu)
            while f(psi + t * step, mu) > f0 + 0.25 * t * g @ step:
                t /= 2
            psi = psi + t * step
        if 3 * k * mu < tol:                           # duality gap ≤ ν·mu, ν = k (SDP cone) + 2k (bounds)
            break
        mu /= 10
    return float(1 - psi.sum() / R.sum()), psi


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


def velicer_map(R: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Velicer's (1976) minimum average partial test and its revision (Velicer, Eaton & Fava 2000).

    O'Connor's (2000) algorithm: remove the first m principal components (m = 0 … p − 1), rescale
    the remainder to partial correlations, and average their squares (1976) or fourth powers
    (2000) over the off-diagonal; K = the m at the minimum. Returns (K_1976, K_2000, f2, f4), f
    indexed by m. Removing the component of m equicorrelated variables leaves them at partial r
    = −1/(m − 1) whatever their r, so MAP keeps such a component only if r > 1/(m − 1).
    """
    R = np.asarray(R, float)
    p = len(R)
    w, v = np.linalg.eigh(R)
    order = np.argsort(w)[::-1]
    L = v[:, order] * np.sqrt(np.clip(w[order], 0, None))             # principal-component loadings
    off = ~np.eye(p, dtype=bool)
    f2, f4 = [np.mean(R[off] ** 2)], [np.mean(R[off] ** 4)]
    for m in range(1, p):
        C = R - L[:, :m] @ L[:, :m].T                                  # partial covariance after m components
        d = np.sqrt(np.diag(C))
        P = C / np.outer(d, d)
        f2.append(np.mean(P[off] ** 2)); f4.append(np.mean(P[off] ** 4))
    f2, f4 = np.array(f2), np.array(f4)
    return int(f2.argmin()), int(f4.argmin()), f2, f4


def ekc(eig: np.ndarray, n: int) -> tuple[int, np.ndarray]:
    """Braeken & van Assen's (2017) empirical Kaiser criterion, as EFAtools' efa_ekc computes it.

    Reference j = the Marchenko–Pastur upper edge (1 + √(p/n))² × the variance the j − 1 larger
    eigenvalues leave, shared over the p − j + 1 that remain; floored at 1. K = the leading run of
    eigenvalues above their reference. Returns (K, references).
    """
    eig = np.sort(np.asarray(eig, float))[::-1]
    p = len(eig)
    left = p - np.concatenate([[0.0], np.cumsum(eig)[:-1]])
    ref = np.maximum((1 + np.sqrt(p / n)) ** 2 * left / np.arange(p, 0, -1), 1.0)
    return int(np.cumprod(eig > ref).sum()), ref


def _principal_axis(R: np.ndarray, nf: int, max_iter: int = 50, crit: float = 1e-3) -> np.ndarray:
    """Iterated principal-axis loadings, starting from 1s on the diagonal (Ruscio's Factor.Analysis)."""
    C, old = R.copy(), np.full(len(R), 99.0)
    for _ in range(max_iter):
        w, v = np.linalg.eigh(C)
        top = np.argsort(w)[::-1][:nf]
        L = v[:, top] * np.sqrt(np.clip(w[top], 0, None))
        h2 = (L ** 2).sum(1)
        if np.abs(old - h2).max() < crit:
            break
        old = h2
        np.fill_diagonal(C, h2)
    return L


def _comparison_population(X: np.ndarray, nf: int, N: int, rng, max_trials: int = 5) -> np.ndarray:
    """Ruscio & Kaczetow (2008) GenData: N cases from an nf-factor model with X's correlations and X's own
    marginals (bootstrapped, swapped in by rank). The matrix the factors are fitted to is nudged by the
    residual until the reproduced correlations stop improving for max_trials halvings of the step."""
    n, k = X.shape
    margins = np.sort(X[rng.integers(0, n, (N, k)), np.arange(k)], axis=0)
    target = np.corrcoef(X, rowvar=False)
    shared, unique = rng.standard_normal((N, nf)), rng.standard_normal((N, k))
    tril = np.tril_indices(k, -1)

    def build(R_fit):
        L = np.clip(_principal_axis(R_fit, nf), -1, 1)
        L = -L if L[0, 0] < 0 else L
        D = shared @ L.T + unique * np.sqrt(np.clip(1 - (L ** 2).sum(1), 0, None))
        out = np.empty_like(D)
        for j in range(k):
            out[np.argsort(D[:, j]), j] = margins[:, j]
        return out

    R_fit, best, best_rmsr, best_res, trials = target.copy(), None, np.inf, None, 0
    while trials < max_trials:
        res = target - np.corrcoef(build(R_fit), rowvar=False)
        rmsr = np.sqrt(np.mean(res[tril] ** 2))
        if rmsr < best_rmsr:
            best, best_rmsr, best_res, trials = R_fit.copy(), rmsr, res, 0
            R_fit = R_fit + res
        else:
            trials += 1
            R_fit = best + 0.5 ** trials * best_res
    return build(best)


def comparison_data(X: np.ndarray, f_max: int | None = None, n_pop: int = 10_000, n_samples: int = 500,
                    alpha: float = 0.30, seed: int = 0) -> tuple[int, np.ndarray]:
    """Ruscio & Roche's (2012) comparison-data test with their defaults (population 10,000, 500 samples, α .30).

    For f = 1, 2, …: simulate a population with f common factors that reproduces X's correlations and
    marginals, draw n_samples samples of X's size from it, and score each by the RMSR between its eigenvalues
    and X's. Stop at the first f that does not fit significantly better than f − 1 (one-sided Mann–Whitney U):
    K = f − 1, so K ≥ 1 by construction. f_max defaults to the most factors with positive degrees of freedom
    (EFAtools' CD default; 6 for eleven variables). Returns (K, RMSR: samples × factor counts tried).
    """
    from scipy.stats import mannwhitneyu

    X = np.asarray(X, float)
    n, k = X.shape
    f_max = f_max or max(f for f in range(1, k) if (k - f) ** 2 > k + f)
    rng = np.random.default_rng(seed)
    eig = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    rmsr = []
    for f in range(1, f_max + 1):
        pop = _comparison_population(X, f, n_pop, rng)
        e = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(pop[rng.integers(0, n_pop, n)], rowvar=False)))[::-1]
                      for _ in range(n_samples)])
        rmsr.append(np.sqrt(np.mean((e - eig) ** 2, axis=1)))
        if f > 1 and mannwhitneyu(rmsr[-1], rmsr[-2], alternative="less").pvalue >= alpha:
            return f - 1, np.column_stack(rmsr)
    return f_max, np.column_stack(rmsr)


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
                ("omega_t_1factor_keyed", round(omega_1factor(np.corrcoef(Zk, rowvar=False)), 4))]
    R_raw, R_key = np.corrcoef(Z, rowvar=False), np.corrcoef(Zk, rowvar=False)
    (g_raw, nf_raw), (g_key, nf_key) = glb_fa(R_raw), glb_fa(R_key)
    results += [("glb_fa_raw_signs", round(g_raw, 4)), ("glb_fa_raw_signs_nfactors", nf_raw),
                ("glb_fa_keyed_table2_signs", round(g_key, 4)), ("glb_fa_keyed_nfactors", nf_key),
                ("glb_fa_keyed_exact_minres", round(glb_fa(R_key, exact=True)[0], 4)),
                ("glb_algebraic_raw_signs", round(glb_algebraic(R_raw)[0], 4)),
                ("glb_algebraic_keyed_table2_signs", round(glb_algebraic(R_key)[0], 4)),
                ("thomas_alpha", 0.67), ("thomas_GLB_glb.fa_n168", 0.85), ("thomas_implied_mean_r", round(0.67 / (11 - 10 * 0.67), 4))]
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
