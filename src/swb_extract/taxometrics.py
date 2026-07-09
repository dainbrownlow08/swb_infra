"""Taxometric analysis — types vs continuum, formally (AUDIT.md §4A6, Q1 spec).

Pure, tested algorithms only — analysis narrative lives in analysis/07_final.ipynb.
Implements Meehl's MAMBAC and MAXEIG, L-Mode, the Ruscio & Kaczetow (2008)
comparison-data generator, and the Comparison Curve Fit Index. The CCFI logic:
generate B *dimensional* and B *taxonic* comparison populations that both
reproduce the observed indicators' marginals and correlation matrix, differing
only in latent structure; whichever family's mean curve the observed curve
tracks more closely wins. CCFI = RMSR_dim / (RMSR_dim + RMSR_tax):
**< 0.45 dimensional, > 0.55 taxonic, between ambiguous** (Ruscio's bands).

Conventions
-----------
- ``X`` is an (n, k) indicator matrix, indicators sign-aligned so higher =
  more of the putative taxon; NaNs are not handled here — pass complete data.
- All stochastic functions take an explicit ``seed``; identical seeds give
  identical output (numpy default_rng).
- Curves returned by :func:`mambac` / :func:`maxeig` / :func:`lmode` have a
  dataset-independent length for a fixed n (MAMBAC: one value per interior
  cut; MAXEIG: one per window; L-Mode: a fixed density grid), so observed and
  comparison curves are directly comparable.

References: Meehl & Yonce (1994, 1996) — MAMBAC/MAXCOV reports; Waller &
Meehl (1998) *Multivariate Taxometric Procedures*; Ruscio, Ruscio & Meron
(2007) — comparison-data CCFI; Ruscio & Kaczetow (2008) — the GenData
iterative algorithm; Ruscio (2007) — CCFI interpretation bands.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import FactorAnalysis

__all__ = [
    "gen_data",
    "mambac",
    "maxeig",
    "lmode",
    "ccfi",
    "ccfi_suite",
    "ccfi_verdict",
    "LMODE_GRID",
]

LMODE_GRID = np.linspace(-4.0, 4.0, 200)

_OFFDIAG_CLIP = 0.99


def _offdiag_mask(k: int) -> np.ndarray:
    return ~np.eye(k, dtype=bool)


def _rmsr_offdiag(a: np.ndarray, b: np.ndarray) -> float:
    m = _offdiag_mask(a.shape[0])
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _maxdev_offdiag(a: np.ndarray, b: np.ndarray) -> float:
    m = _offdiag_mask(a.shape[0])
    return float(np.max(np.abs(a[m] - b[m])))


def _nearest_psd_corr(R: np.ndarray) -> np.ndarray:
    """Eigen-floor a symmetric matrix and renormalize to unit diagonal."""
    Rs = (R + R.T) / 2.0
    w, V = np.linalg.eigh(Rs)
    w = np.clip(w, 1e-6, None)
    out = (V * w) @ V.T
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def _rank_map_to_marginals(
    Z: np.ndarray, marginals: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Monotone per-column map of Z's ranks onto bootstrap-resampled marginals."""
    n, k = Z.shape
    X = np.empty((n, k))
    for j in range(k):
        boot = np.sort(rng.choice(marginals[j], size=n, replace=True))
        order = np.argsort(Z[:, j], kind="stable")
        X[order, j] = boot
    return X


def _factor_loadings(R: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """One-factor principal-factor loadings of a correlation matrix.

    Iterates communalities on the diagonal; returns loadings ``a`` with
    ``a aᵀ ≈`` the off-diagonal of R. Sign-aligned so the loading sum is
    positive. Clipped to (0.05, 0.95) in absolute value for stability.
    """
    R = np.asarray(R, dtype=float)
    k = R.shape[0]
    m = _offdiag_mask(k)
    h2 = np.array([np.max(np.abs(R[j][m[j]])) for j in range(k)])
    a = np.zeros(k)
    for _ in range(n_iter):
        Rs = R.copy()
        np.fill_diagonal(Rs, h2)
        w, V = np.linalg.eigh(Rs)
        lam, v = w[-1], V[:, -1]
        a = np.sqrt(max(lam, 1e-9)) * v
        if a.sum() < 0:
            a = -a
        h2 = np.clip(a**2, 1e-4, 0.95**2)
    sign = np.sign(a)
    sign[sign == 0] = 1.0
    return sign * np.clip(np.abs(a), 0.05, 0.95)


def gen_data(
    target_corr,
    marginals,
    n: int,
    taxonic: bool = False,
    base_rate: float = 0.45,
    seed: int = 0,
    max_iter: int = 30,
) -> np.ndarray:
    """Ruscio & Kaczetow (2008) comparison data: marginals + correlations, chosen structure.

    Dimensional (``taxonic=False``): iterate an intermediate correlation
    matrix — draw multivariate normal, rank-map each column onto a
    bootstrap-resampled empirical marginal, measure the reproduced
    correlations, feed the residual back — and return the iterate with the
    lowest off-diagonal RMSR to ``target_corr``.

    Taxonic (``taxonic=True``): two latent classes at ``base_rate``,
    **within-class correlations 0**; per-indicator class separations from the
    one-factor decomposition of ``target_corr`` (with unit pooled variance,
    r_jl = π(1−π)·d_j·d_l), then the same rank-mapping onto the pooled
    empirical marginals (monotone, so the class order structure survives) with
    an iteratively rescaled separation to reproduce ``target_corr`` after
    mapping.

    ``marginals`` is a length-k sequence of 1-D arrays of empirical values.
    """
    rng = np.random.default_rng(seed)
    R_target = np.asarray(target_corr, dtype=float)
    marginals = [np.asarray(m, dtype=float).ravel() for m in marginals]
    k = R_target.shape[0]
    if len(marginals) != k:
        raise ValueError(f"{len(marginals)} marginals for a {k}x{k} target_corr")

    best: np.ndarray | None = None
    best_dev = np.inf
    n_cal = max(2000, 2 * n)  # calibration draws: big enough that residuals are ~noise-free
    _N_CAL_ITER = 8

    if not taxonic:
        # Phase 1 — calibrate the intermediate matrix on low-noise draws. Feeding
        # back small-n residuals at full strength makes R_int random-walk with the
        # sampling noise and the loop never converges; the big-n residual is the
        # systematic rank-mapping attenuation only, which is what we must undo.
        R_int = R_target.copy()
        for _ in range(_N_CAL_ITER):
            L = np.linalg.cholesky(_nearest_psd_corr(R_int))
            Zc = rng.standard_normal((n_cal, k)) @ L.T
            Xc = _rank_map_to_marginals(Zc, marginals, rng)
            R_int = R_int + (R_target - np.corrcoef(Xc, rowvar=False))
            np.fill_diagonal(R_int, 1.0)
            R_int = np.clip(R_int, -_OFFDIAG_CLIP, _OFFDIAG_CLIP)
            np.fill_diagonal(R_int, 1.0)
        # Phase 2 — sample candidates at n from the fixed calibrated state; keep
        # the draw with the smallest elementwise deviation (the spec's bound).
        L = np.linalg.cholesky(_nearest_psd_corr(R_int))
        for _ in range(max_iter):
            Z = rng.standard_normal((n, k)) @ L.T
            X = _rank_map_to_marginals(Z, marginals, rng)
            dev = _maxdev_offdiag(np.corrcoef(X, rowvar=False), R_target)
            if dev < best_dev:
                best, best_dev = X, dev
        return best

    # --- taxonic ---------------------------------------------------------
    pq = base_rate * (1.0 - base_rate)
    a = _factor_loadings(R_target)
    m = _offdiag_mask(k)
    mean_tgt = float(np.mean(np.abs(R_target[m])))

    def _tax_draw(nn: int, cls: np.ndarray, scale: float) -> np.ndarray:
        d = np.clip(scale * a, -0.995, 0.995) / np.sqrt(pq)
        sw = np.sqrt(np.clip(1.0 - pq * d**2, 1e-4, None))
        mu = np.where(cls[:, None], (1.0 - base_rate) * d, -base_rate * d)
        Z = mu + rng.standard_normal((nn, k)) * sw
        return _rank_map_to_marginals(Z, marginals, rng)

    def _classes(nn: int) -> np.ndarray:
        cls = np.zeros(nn, dtype=bool)
        cls[: int(round(nn * base_rate))] = True
        rng.shuffle(cls)
        return cls

    # Phase 1 — calibrate the separation scale on low-noise draws (r ∝ scale²,
    # so correct by the square root of the mean-|r| ratio each round).
    scale = 1.0
    cls_cal = _classes(n_cal)
    for _ in range(_N_CAL_ITER):
        R_rep = np.corrcoef(_tax_draw(n_cal, cls_cal, scale), rowvar=False)
        mean_rep = float(np.mean(np.abs(R_rep[m])))
        if mean_rep > 1e-8:
            scale = float(np.clip(scale * np.sqrt(mean_tgt / mean_rep), 0.05, 5.0))
    # Phase 2 — sample candidates at n with the fixed scale; keep the best draw.
    cls = _classes(n)
    for _ in range(max_iter):
        X = _tax_draw(n, cls, scale)
        dev = _maxdev_offdiag(np.corrcoef(X, rowvar=False), R_target)
        if dev < best_dev:
            best, best_dev = X, dev
    return best


def mambac(X, cut_buffer: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Meehl & Yonce MAMBAC: mean-above minus mean-below curves.

    For every ordered indicator pair (input, output): sort cases by the input;
    for each interior cut c (rank ``cut_buffer`` .. ``n − cut_buffer``) record
    mean(output above cut) − mean(output below cut). Returns
    ``(curves, mean_curve)`` with ``curves`` of shape (k·(k−1), n_cuts).
    Taxonic → peaked curves; dimensional → a concave dish rising at the ends.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    cuts = np.arange(cut_buffer, n - cut_buffer + 1)
    if len(cuts) < 3:
        raise ValueError(f"n={n} too small for cut_buffer={cut_buffer}")
    curves = np.empty((k * (k - 1), len(cuts)))
    row = 0
    for i in range(k):
        order = np.argsort(X[:, i], kind="stable")
        for o in range(k):
            if o == i:
                continue
            y = X[order, o]
            csum = np.concatenate(([0.0], np.cumsum(y)))
            total = csum[-1]
            below = csum[cuts] / cuts
            above = (total - csum[cuts]) / (n - cuts)
            curves[row] = above - below
            row += 1
    return curves, curves.mean(axis=0)


def maxeig(X, windows: int = 50, overlap: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """MAXEIG: largest eigenvalue of the other-indicator covariance along each input.

    Per input indicator: sort cases by it; slide ``windows`` overlapping
    windows (fractional ``overlap``) along the sorted order; in each window
    take the covariance matrix of the *other* indicators, zero its diagonal,
    and record the largest eigenvalue. Returns ``(curves, mean_curve)`` with
    ``curves`` of shape (k, windows). Taxonic → an interior peak (hitmax);
    dimensional → flat/monotone drift.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if k < 3:
        raise ValueError("maxeig needs >= 3 indicators")
    w = int(np.floor(n / (1.0 + (windows - 1) * (1.0 - overlap))))
    if w < k:
        raise ValueError(f"window size {w} < indicators {k}; reduce windows/overlap")
    step = (n - w) / (windows - 1) if windows > 1 else 0.0
    curves = np.empty((k, windows))
    for i in range(k):
        order = np.argsort(X[:, i], kind="stable")
        others = X[order][:, [j for j in range(k) if j != i]]
        for t in range(windows):
            lo = int(round(t * step))
            seg = others[lo : lo + w]
            C = np.cov(seg, rowvar=False)
            np.fill_diagonal(C, 0.0)
            curves[i, t] = np.linalg.eigvalsh(C)[-1]
    return curves, curves.mean(axis=0)


def lmode(X, grid: np.ndarray = LMODE_GRID, seed: int = 0):
    """L-Mode: one-factor score density on a fixed grid (+ the scores).

    ``FactorAnalysis(1)`` scores, standardized, Gaussian-KDE (Scott's rule)
    evaluated on ``grid``. Returns ``(density, scores)``. Taxonic → bimodal
    factor-score density; dimensional → unimodal (ties back to §4A1's dip).
    """
    from scipy.stats import gaussian_kde

    X = np.asarray(X, dtype=float)
    fa = FactorAnalysis(n_components=1, random_state=seed).fit(X)
    s = fa.transform(X).ravel()
    sd = s.std()
    if sd <= 0:
        raise ValueError("degenerate factor scores (zero variance)")
    s = (s - s.mean()) / sd
    return gaussian_kde(s)(grid), s


def ccfi(observed_curve, dim_curves, tax_curves) -> float:
    """Comparison Curve Fit Index: RMSR_dim / (RMSR_dim + RMSR_tax).

    ``dim_curves`` / ``tax_curves`` are (B, len(curve)) stacks from
    :func:`gen_data` populations; the observed curve is compared to each
    family's mean curve. < 0.45 ⇒ dimensional, > 0.55 ⇒ taxonic (Ruscio).
    """
    obs = np.asarray(observed_curve, dtype=float)
    dmean = np.asarray(dim_curves, dtype=float).mean(axis=0)
    tmean = np.asarray(tax_curves, dtype=float).mean(axis=0)
    if obs.shape != dmean.shape or obs.shape != tmean.shape:
        raise ValueError("curve length mismatch between observed and comparison stacks")
    rmsr_d = float(np.sqrt(np.mean((obs - dmean) ** 2)))
    rmsr_t = float(np.sqrt(np.mean((obs - tmean) ** 2)))
    if rmsr_d + rmsr_t == 0:
        return 0.5
    return rmsr_d / (rmsr_d + rmsr_t)


def ccfi_verdict(c: float) -> str:
    if c < 0.45:
        return "dimensional"
    if c > 0.55:
        return "taxonic"
    return "ambiguous"


def ccfi_suite(
    X,
    B: int = 100,
    base_rate: float = 0.45,
    seed: int = 0,
    cut_buffer: int = 25,
    windows: int = 50,
    overlap: float = 0.9,
) -> dict:
    """Full CCFI run: observed curves vs B dimensional + B taxonic populations.

    The one composition the notebook and the known-answer tests share, so the
    analysis and its validation cannot drift apart. Comparison populations are
    :func:`gen_data` calls on the observed correlation matrix + marginals
    (dimensional seeds ``seed+b``, taxonic ``seed+b+100000``). Returns
    ``{"ccfi": {procedure: value, "mean": ...}, "observed": {...},
    "dim_curves": {...}, "tax_curves": {...}}`` — the curve stacks feed the
    observed-curve-over-envelopes figure.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    R = np.corrcoef(X, rowvar=False)
    marg = [X[:, j] for j in range(k)]
    obs = {
        "mambac": mambac(X, cut_buffer)[1],
        "maxeig": maxeig(X, windows, overlap)[1],
        "lmode": lmode(X)[0],
    }
    stacks: dict[str, dict[str, list]] = {
        fam: {p: [] for p in obs} for fam in ("dim", "tax")
    }
    for fam, taxo, off in (("dim", False, 0), ("tax", True, 100000)):
        for b in range(B):
            Xc = gen_data(
                R, marg, n, taxonic=taxo, base_rate=base_rate, seed=seed + b + off
            )
            stacks[fam]["mambac"].append(mambac(Xc, cut_buffer)[1])
            stacks[fam]["maxeig"].append(maxeig(Xc, windows, overlap)[1])
            stacks[fam]["lmode"].append(lmode(Xc)[0])
    out = {
        p: ccfi(obs[p], np.array(stacks["dim"][p]), np.array(stacks["tax"][p]))
        for p in obs
    }
    out["mean"] = float(np.mean([out["mambac"], out["maxeig"], out["lmode"]]))
    return {
        "ccfi": out,
        "observed": obs,
        "dim_curves": {p: np.array(v) for p, v in stacks["dim"].items()},
        "tax_curves": {p: np.array(v) for p, v in stacks["tax"].items()},
    }
