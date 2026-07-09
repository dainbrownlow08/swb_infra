"""Formal modality statistics for the continuum battery (AUDIT.md §4A, Q1 spec).

Pure, tested algorithms only — analysis narrative lives in analysis/07_final.ipynb.
Implements the §4A1–A4 test stack: binned Gaussian-KDE mode counting, the critical
bandwidth h_crit, Silverman's smoothed-bootstrap bandwidth test, McLachlan's
parametric-bootstrap GMM likelihood-ratio test, and the skewed-family BIC
comparison that adjudicates "2 Gaussians vs 1 skewed component".

Conventions
-----------
- Inputs are 1-D samples; NaNs are dropped at entry.
- Bandwidths are in **SD units** of the input (samples are standardized
  internally; ``standardize=False`` lets the bootstrap reuse a fixed scale).
- The Gaussian kernel is load-bearing: the mode count of a Gaussian KDE is
  nonincreasing in bandwidth (Silverman 1981), which is what makes h_crit
  well-defined and the bisection valid. Do not swap kernels.
- The Silverman test is known to be conservative at nominal levels
  (Hall & York 2001) — note this wherever its p-values are cited.

References: Silverman (1981) JRSS-B 43(1) 97–99; Hall & York (2001) Statistica
Sinica 11, 515–536; McLachlan (1987) JRSS-C 36(3) 318–324; Efron & Tibshirani
(1993) ch. 16.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture

__all__ = [
    "count_modes",
    "h_crit",
    "silverman_test",
    "gmm_blrt",
    "fit_family_bic",
]

_MIN_N = 10
_MODE_FLOOR_FRAC = 1e-6  # float-plateau guard: ignore maxima below this × peak


def _as_clean_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < _MIN_N:
        raise ValueError(f"need at least {_MIN_N} finite values, got {arr.size}")
    if arr.std() == 0.0:
        raise ValueError("zero-variance sample")
    return arr


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _count_strict_maxima(density: np.ndarray, min_density: float) -> int:
    """Strict local maxima of a 1-D array, plateau-safe.

    Consecutive equal values are compressed into runs first, so a flat-topped
    peak counts once; runs at or below ``min_density`` never count. A boundary
    run counts only if it strictly exceeds its single neighbour.
    """
    d = np.asarray(density, dtype=float)
    keep = np.ones(d.size, dtype=bool)
    keep[1:] = d[1:] != d[:-1]
    v = d[keep]
    if v.size == 1:
        return int(v[0] > min_density)
    left = np.empty(v.size, dtype=bool)
    right = np.empty(v.size, dtype=bool)
    left[0] = True
    left[1:] = v[1:] > v[:-1]
    right[-1] = True
    right[:-1] = v[:-1] > v[1:]
    return int(np.sum((v > min_density) & left & right))


def count_modes(x, h: float, grid_n: int = 4096, standardize: bool = True) -> int:
    """Mode count of the Gaussian KDE of ``x`` at bandwidth ``h`` (SD units).

    Binned implementation: histogram on a grid padded 3h beyond the sample
    range, then a Gaussian filter with sigma = h/binwidth — O(n + grid_n) per
    bandwidth, which is what makes utterance-level bootstraps feasible.
    """
    if h <= 0:
        raise ValueError("bandwidth h must be positive")
    z = np.asarray(x, dtype=float).ravel()
    if standardize:
        z = _standardize(_as_clean_1d(z))
    lo = z.min() - 3.0 * h
    hi = z.max() + 3.0 * h
    counts, _ = np.histogram(z, bins=grid_n, range=(lo, hi))
    binwidth = (hi - lo) / grid_n
    density = gaussian_filter1d(
        counts.astype(float), sigma=h / binwidth, mode="constant", cval=0.0
    )
    return _count_strict_maxima(density, _MODE_FLOOR_FRAC * density.max())


def h_crit(x, k: int = 1, grid_n: int = 4096, max_iter: int = 40) -> float:
    """Smallest bandwidth (SD units) at which the Gaussian KDE has ≤ k modes.

    Valid because the Gaussian-kernel mode count is nonincreasing in h
    (Silverman 1981). Bisection on [1e-3, 2·range(z)]; returns the upper end
    of the final bracket (guaranteed ≤ k modes).
    """
    z = _standardize(_as_clean_1d(x))
    lo = 1e-3
    hi = 2.0 * (z.max() - z.min())
    while count_modes(z, hi, grid_n, standardize=False) > k:  # pragma: no cover
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if count_modes(z, mid, grid_n, standardize=False) <= k:
            hi = mid
        else:
            lo = mid
    return hi


def silverman_test(
    x, k: int = 1, B: int = 999, seed: int = 0, grid_n: int = 4096
) -> tuple[float, float]:
    """Silverman's critical-bandwidth bootstrap test of H0: ≤ k modes.

    Draws B smoothed-bootstrap samples from the KDE at h_crit with the
    variance-preserving rescale y = z̄ + (z_J − z̄ + h·ε)/√(1 + h²/σ̂²) and
    reports p = #{count_modes(y_b, h_crit) > k}/B (Efron & Tibshirani 1993,
    ch. 16). Small p ⇒ reject "≤ k modes". Conservative at nominal levels
    (Hall & York 2001). Returns (h_crit, p).
    """
    z = _standardize(_as_clean_1d(x))
    n = z.size
    h = h_crit(z, k=k, grid_n=grid_n)
    zbar = z.mean()
    var = z.var()
    scale = np.sqrt(1.0 + (h * h) / var)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(B):
        resample = z[rng.integers(0, n, n)]
        y = zbar + (resample - zbar + h * rng.standard_normal(n)) / scale
        if count_modes(y, h, grid_n, standardize=False) > k:
            exceed += 1
    return h, exceed / B


def _fit_gmm(X: np.ndarray, k: int, n_init: int, seed: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=k, covariance_type="full", n_init=n_init, random_state=seed
    ).fit(X)


def gmm_blrt(
    x, k0: int = 1, k1: int = 2, B: int = 999, n_init: int = 10, seed: int = 0
) -> tuple[float, float]:
    """McLachlan (1987) parametric-bootstrap LRT of k0 vs k1 Gaussian components.

    LR_obs = 2n·(score_k1 − score_k0) (sklearn's ``score`` is the *mean*
    per-sample log-likelihood). B datasets are simulated from the fitted k0
    model and both orders refit on each; p = (1 + #{LR_b ≥ LR_obs})/(B + 1).
    Returns (LR_obs, p). This is a test, unlike BIC model selection — it is
    the direct answer to "but BIC picked k=2" (AUDIT.md §4A3).
    """
    xx = _as_clean_1d(x)
    n = xx.size
    X = xx.reshape(-1, 1)
    g0 = _fit_gmm(X, k0, n_init, seed)
    g1 = _fit_gmm(X, k1, n_init, seed)
    lr_obs = 2.0 * n * (g1.score(X) - g0.score(X))

    weights = g0.weights_
    means = g0.means_.ravel()
    sds = np.sqrt(g0.covariances_.ravel())
    rng = np.random.default_rng(seed)
    exceed = 0
    for b in range(B):
        comp = rng.choice(k0, size=n, p=weights)
        sim = rng.normal(means[comp], sds[comp]).reshape(-1, 1)
        s0 = _fit_gmm(sim, k0, n_init, seed + b + 1)
        s1 = _fit_gmm(sim, k1, n_init, seed + b + 1)
        lr_b = 2.0 * n * (s1.score(sim) - s0.score(sim))
        if lr_b >= lr_obs:
            exceed += 1
    return lr_obs, (1 + exceed) / (B + 1)


def fit_family_bic(x, n_init: int = 10, seed: int = 0) -> pd.DataFrame:
    """BIC comparison: single skewed families vs the 2-Gaussian mixture (§4A4).

    MLE fits for norm (2 params), skewnorm (3), lognorm with free loc (3),
    jf_skew_t (4), plus GaussianMixture(1) (≡ norm, cross-check) and
    GaussianMixture(2) (5 params in 1-D). BIC = p·ln n − 2·ll; log-likelihoods
    with non-finite logpdf anywhere (support-edge violations) are recorded as
    −inf and sort last. Returns a DataFrame sorted by BIC ascending.
    """
    xx = _as_clean_1d(x)
    n = xx.size
    rows: list[dict] = []

    def add(family: str, n_params: int, ll: float, params) -> None:
        bic = n_params * np.log(n) - 2.0 * ll if np.isfinite(ll) else np.inf
        rows.append(
            {
                "family": family,
                "n_params": n_params,
                "loglik": ll,
                "bic": bic,
                "params": params,
            }
        )

    def scipy_family(name: str, dist, n_params: int) -> None:
        try:
            with warnings.catch_warnings():
                # scipy's generic MLE explores invalid params en route (log of
                # negatives at lognorm support edges, jf_skew_t overflow); the
                # -inf guard below is the real defense — silence the noise.
                warnings.simplefilter("ignore", RuntimeWarning)
                params = dist.fit(xx)
                lp = dist.logpdf(xx, *params)
            ll = float(lp.sum()) if np.all(np.isfinite(lp)) else -np.inf
            add(name, n_params, ll, tuple(round(float(p), 6) for p in params))
        except Exception as exc:  # fit blow-ups recorded, never raised
            add(name, n_params, -np.inf, f"fit failed: {exc}")

    scipy_family("norm", sps.norm, 2)
    scipy_family("skewnorm", sps.skewnorm, 3)
    scipy_family("lognorm", sps.lognorm, 3)
    scipy_family("jf_skew_t", sps.jf_skew_t, 4)

    X = xx.reshape(-1, 1)
    for k in (1, 2):
        gm = _fit_gmm(X, k, n_init, seed)
        params = {
            "weights": np.round(gm.weights_, 4).tolist(),
            "means": np.round(gm.means_.ravel(), 4).tolist(),
            "sds": np.round(np.sqrt(gm.covariances_.ravel()), 4).tolist(),
        }
        add(f"gmm{k}", 3 * k - 1, float(n * gm.score(X)), params)

    return (
        pd.DataFrame(rows).sort_values("bic", kind="stable").reset_index(drop=True)
    )
