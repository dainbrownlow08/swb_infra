"""Known-answer tests for the taxometrics module (AUDIT.md §4A6, Q1 spec).

Non-negotiable pattern (Q1): every statistic must (a) detect planted structure
and (b) pass a planted null, seeded, before touching real data. The end-to-end
pair (dimensional-generated -> CCFI < 0.45; taxonic at the paper's parameters
-> CCFI > 0.55) doubles as the A7 machinery validation.

"The paper's parameters": the claimed PC1 mixture 0.55*N(-0.31, 0.13^2) +
0.45*N(+0.36, 0.31^2) has base rate pi = 0.45 and a class separation of
(0.36 - (-0.31)) / sqrt(0.55*0.13^2 + 0.45*0.31^2) ~= 2.9 pooled-within SDs.
"""
import numpy as np
import pytest
from scipy.stats import ks_2samp

from swb_extract.taxometrics import (
    ccfi,
    ccfi_suite,
    ccfi_verdict,
    gen_data,
    lmode,
    mambac,
    maxeig,
)

K = 5
PAPER_BASE_RATE = 0.45
PAPER_D = 2.9


def _dimensional_pop(n=600, k=K, seed=11):
    """One-factor continuum with lognormal (skewed) marginals."""
    rng = np.random.default_rng(seed)
    a = np.array([0.8, 0.7, 0.6, 0.7, 0.5][:k])
    f = rng.standard_normal(n)
    X = a * f[:, None] + rng.standard_normal((n, k)) * np.sqrt(1 - a**2)
    return np.exp(0.6 * X)


def _taxonic_pop(n=600, k=K, seed=13, base_rate=PAPER_BASE_RATE, d=PAPER_D):
    """Two classes at the paper's base rate/separation, within-class independent."""
    rng = np.random.default_rng(seed)
    cls = np.zeros(n, dtype=bool)
    cls[: int(round(n * base_rate))] = True
    rng.shuffle(cls)
    return rng.standard_normal((n, k)) + np.where(cls[:, None], d, 0.0)


def _corr_marg(X):
    return np.corrcoef(X, rowvar=False), [X[:, j] for j in range(X.shape[1])]


# --- gen_data -----------------------------------------------------------


def test_gen_data_dimensional_reproduces_corr_and_marginals():
    X = _dimensional_pop()
    R, marg = _corr_marg(X)
    G = gen_data(R, marg, n=len(X), taxonic=False, seed=0)
    Rg = np.corrcoef(G, rowvar=False)
    assert np.max(np.abs((Rg - R)[~np.eye(K, dtype=bool)])) < 0.05
    for j in range(K):
        assert ks_2samp(G[:, j], X[:, j]).statistic < 0.10


def test_gen_data_taxonic_reproduces_corr_and_marginals():
    X = _taxonic_pop()
    R, marg = _corr_marg(X)
    G = gen_data(R, marg, n=len(X), taxonic=True, base_rate=PAPER_BASE_RATE, seed=0)
    Rg = np.corrcoef(G, rowvar=False)
    assert np.max(np.abs((Rg - R)[~np.eye(K, dtype=bool)])) < 0.05
    for j in range(K):
        assert ks_2samp(G[:, j], X[:, j]).statistic < 0.10


def test_gen_data_seeded_reproducible():
    X = _dimensional_pop()
    R, marg = _corr_marg(X)
    g1 = gen_data(R, marg, n=200, seed=7)
    g2 = gen_data(R, marg, n=200, seed=7)
    g3 = gen_data(R, marg, n=200, seed=8)
    assert np.array_equal(g1, g2)
    assert not np.array_equal(g1, g3)


# --- curve shapes: planted structure detected, planted null passed --------


def test_mambac_taxonic_peak_vs_dimensional_dish():
    Xt = _taxonic_pop()
    _, mt = mambac(Xt)
    n_cuts = len(mt)
    # planted taxon: interior peak
    assert n_cuts * 0.2 < np.argmax(mt) < n_cuts * 0.8
    # planted null (correlated normal continuum): ends above the middle (dish)
    rng = np.random.default_rng(5)
    f = rng.standard_normal(600)
    Xd = 0.7 * f[:, None] + rng.standard_normal((600, K)) * np.sqrt(1 - 0.49)
    _, md = mambac(Xd)
    ends = (md[: n_cuts // 10].mean() + md[-n_cuts // 10 :].mean()) / 2
    mid = md[int(n_cuts * 0.4) : int(n_cuts * 0.6)].mean()
    assert ends > mid


def test_maxeig_hitmax_peak_vs_flat():
    Xt = _taxonic_pop()
    _, ct = maxeig(Xt)
    edge_t = np.r_[ct[:5], ct[-5:]].mean()
    assert 50 * 0.2 < np.argmax(ct) < 50 * 0.8 and ct.max() > 2 * edge_t
    Xd = _dimensional_pop()
    _, cd = maxeig(Xd)
    edge_d = np.r_[cd[:5], cd[-5:]].mean()
    assert cd.max() < 2 * abs(edge_d) + 2 * cd.std()  # no hitmax spike


def test_lmode_bimodal_vs_unimodal():
    def modes(dens):
        m = (dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:]) & (dens[1:-1] > 0.1 * dens.max())
        return int(m.sum())

    dens_t, _ = lmode(_taxonic_pop())
    dens_d, _ = lmode(_dimensional_pop())
    assert modes(dens_t) >= 2
    assert modes(dens_d) == 1


# --- CCFI ----------------------------------------------------------------


def test_ccfi_bands():
    assert ccfi_verdict(0.30) == "dimensional"
    assert ccfi_verdict(0.50) == "ambiguous"
    assert ccfi_verdict(0.60) == "taxonic"
    obs = np.zeros(10)
    dim = np.zeros((3, 10))
    tax = np.ones((3, 10))
    assert ccfi(obs, dim, tax) == 0.0
    assert ccfi(obs + 1, dim, tax) == 1.0


@pytest.mark.parametrize("taxonic,side", [(False, "dim"), (True, "tax")])
def test_ccfi_end_to_end(taxonic, side):
    X = _taxonic_pop(seed=13) if taxonic else _dimensional_pop(seed=11)
    res = ccfi_suite(X, B=20, base_rate=PAPER_BASE_RATE, seed=0)
    c = res["ccfi"]
    if side == "dim":
        assert c["mambac"] < 0.45 and c["maxeig"] < 0.45 and c["mean"] < 0.45, c
    else:
        assert c["mambac"] > 0.55 and c["maxeig"] > 0.55 and c["mean"] > 0.55, c
