"""Known-answer tests for the §4A modality stack (AUDIT.md Q1 spec).

Non-negotiable contract: every statistic must (a) detect planted structure and
(b) pass a planted null, seeded, before touching real data.
"""
import numpy as np
import pytest

from swb_extract.stats_modality import (
    _count_strict_maxima,
    count_modes,
    fit_family_bic,
    gmm_blrt,
    h_crit,
    silverman_test,
)


def normal_sample(n=500, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def bimodal_sample(n=500, seed=0, sep=2.0, sd=0.5):
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 2, n)
    return rng.normal(np.where(comp == 0, -sep, sep), sd)


def skewed_sample(n=2000, seed=0, a=4.0):
    from scipy.stats import skewnorm

    return skewnorm.rvs(a, size=n, random_state=seed)


# --- _count_strict_maxima: plateau and edge cases ---------------------------


def test_strict_maxima_plateau_counts_once():
    assert _count_strict_maxima(np.array([0.0, 1.0, 1.0, 0.0]), 0.0) == 1


def test_strict_maxima_two_peaks():
    assert _count_strict_maxima(np.array([0.0, 1.0, 0.1, 2.0, 0.0]), 0.0) == 2


def test_strict_maxima_edge_run_counts():
    # rising to the boundary: the terminal run is a maximum
    assert _count_strict_maxima(np.array([0.0, 1.0, 2.0, 3.0]), 0.0) == 1


def test_strict_maxima_floor_filters_dust():
    d = np.array([0.0, 1.0, 0.0, 1e-9, 0.0])
    assert _count_strict_maxima(d, 1e-6 * d.max()) == 1


# --- count_modes -------------------------------------------------------------


def test_count_modes_detects_bimodality_and_smooths_away():
    x = bimodal_sample(n=2000, seed=1)
    assert count_modes(x, 0.15) == 2
    assert count_modes(x, 3.0) == 1


def test_count_modes_unimodal_normal():
    assert count_modes(normal_sample(n=2000, seed=2), 0.5) == 1


def test_count_modes_nonincreasing_in_h():
    x = bimodal_sample(n=1000, seed=3)
    hs = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    counts = [count_modes(x, h) for h in hs]
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    assert counts[0] >= 2 and counts[-1] == 1


# --- h_crit -------------------------------------------------------------------


def test_h_crit_brackets_the_mode_change():
    x = bimodal_sample(n=1000, seed=4)
    z = (x - x.mean()) / x.std()
    hc = h_crit(x, k=1)
    assert count_modes(z, hc, standardize=False) <= 1
    assert count_modes(z, 0.9 * hc, standardize=False) > 1


def test_h_crit_ordering_bimodal():
    x = bimodal_sample(n=1000, seed=5)
    assert h_crit(x, k=2) < h_crit(x, k=1)


# --- silverman_test -----------------------------------------------------------


def test_silverman_passes_planted_null():
    h, p = silverman_test(normal_sample(n=500, seed=6), k=1, B=199, seed=6)
    assert p > 0.1


def test_silverman_detects_planted_bimodality():
    h, p = silverman_test(bimodal_sample(n=500, seed=7), k=1, B=199, seed=7)
    assert p < 0.01


# --- gmm_blrt -----------------------------------------------------------------


def test_gmm_blrt_passes_planted_null():
    lr, p = gmm_blrt(normal_sample(n=300, seed=8), B=49, n_init=4, seed=8)
    assert p >= 0.05


def test_gmm_blrt_detects_planted_mixture():
    lr, p = gmm_blrt(bimodal_sample(n=300, seed=9), B=49, n_init=4, seed=9)
    assert lr > 0
    assert p == pytest.approx(1.0 / 50.0)


# --- fit_family_bic -----------------------------------------------------------


def test_fit_family_skewed_beats_two_gaussians():
    table = fit_family_bic(skewed_sample(n=2000, seed=10))
    bic = dict(zip(table["family"], table["bic"]))
    assert bic["skewnorm"] < bic["gmm2"]
    # gmm1 is the norm cross-check
    assert abs(bic["gmm1"] - bic["norm"]) < 2.0


def test_fit_family_mixture_wins_on_real_mixture():
    table = fit_family_bic(bimodal_sample(n=2000, seed=11))
    assert table.iloc[0]["family"] == "gmm2"
