"""Known-answer checks for analysis/c_alpha/thomas_method.py (the paper's §4.1 arithmetic)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("tm", REPO / "analysis/c_alpha/thomas_method.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)


def test_cronbach_matches_the_textbook_formula():
    rng = np.random.default_rng(0)
    f = rng.standard_normal(5000)
    Z = np.column_stack([0.6 * f + rng.standard_normal(5000) * 0.8 for _ in range(4)])
    Z = tm.zscore(Z)
    alpha, r_bar = tm.cronbach(Z)
    k = 4
    raw = k / (k - 1) * (1 - Z.var(0, ddof=1).sum() / Z.sum(1).var(ddof=1))   # raw-score alpha
    assert alpha == pytest.approx(raw, abs=1e-6)                                 # identical on z-scores
    assert alpha == pytest.approx(k * r_bar / (1 + (k - 1) * r_bar))
    assert 0.55 < alpha < 0.75                                                    # 4 items at r̄≈.36 → α≈.69


def test_pc1_is_unit_norm_oriented_and_matches_numpy():
    rng = np.random.default_rng(1)
    f = rng.standard_normal(2000)
    Z = tm.zscore(np.column_stack([0.7 * f + rng.standard_normal(2000), -0.7 * f + rng.standard_normal(2000), rng.standard_normal(2000)]))
    vec, eig = tm.pc1(Z, orient_col=0)
    assert np.linalg.norm(vec) == pytest.approx(1.0) and vec[0] > 0 and vec[1] < 0
    assert eig[0] / eig.sum() == pytest.approx(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)).max() / 3)
    assert tm.congruence(vec, -vec) == pytest.approx(-1.0) and tm.congruence(vec, vec) == pytest.approx(1.0)


def test_horn_keeps_nothing_on_pure_noise_and_one_on_a_one_factor_set():
    rng = np.random.default_rng(2)
    assert tm.horn_k(rng.standard_normal((800, 11)), n_rand=100, rng=rng) == 0
    f = rng.standard_normal(800)
    Z = np.column_stack([0.6 * f + 0.8 * rng.standard_normal(800) for _ in range(11)])
    assert tm.horn_k(Z, n_rand=100, rng=rng) == 1


def test_map_matches_the_closed_form_on_two_triplets():
    """Removing the component of m equicorrelated variables leaves them at partial r = −1/(m − 1): a triplet
    survives MAP only if r > .5. Two independent triplets at r .6 and .36 → MAP keeps the first only."""
    R = np.eye(6)
    R[:3, :3] = np.where(np.eye(3) == 1, 1, 0.6); R[3:, 3:] = np.where(np.eye(3) == 1, 1, 0.36)
    k76, k00, f2, f4 = tm.velicer_map(R)
    assert np.allclose(f2[:3], [(3 * .6 ** 2 + 3 * .36 ** 2) / 15, (3 * .5 ** 2 + 3 * .36 ** 2) / 15, 6 * .5 ** 2 / 15])
    assert np.allclose(f4[:3], [(3 * .6 ** 4 + 3 * .36 ** 4) / 15, (3 * .5 ** 4 + 3 * .36 ** 4) / 15, 6 * .5 ** 4 / 15])
    assert k76 == k00 == 1
    assert tm.velicer_map(np.corrcoef(np.random.default_rng(6).standard_normal((2000, 11)), rowvar=False))[:2] == (0, 0)


def test_ekc_reference_values_by_hand():
    K, ref = tm.ekc(np.array([2.5, 1.2, 0.6, 0.4, 0.3]), n=100)                 # five variables, n = 100
    edge = (1 + np.sqrt(5 / 100)) ** 2                                         # Marchenko–Pastur upper edge, 1.497
    assert K == 2 and np.allclose(ref[:3], [edge, max(edge * 2.5 / 4, 1), max(edge * 1.3 / 3, 1)])


def test_comparison_data_recovers_two_planted_factors_with_normal_and_skewed_marginals():
    rng = np.random.default_rng(7)
    f = rng.standard_normal((500, 2))
    X = np.column_stack([0.6 * f[:, j // 5] + 0.8 * rng.standard_normal(500) for j in range(10)])
    assert tm.comparison_data(X, seed=0)[0] == 2
    assert tm.comparison_data(np.exp(X / 2), seed=0)[0] == 2                    # CD reproduces each variable's own shape


def test_omega_brackets_alpha_for_a_congeneric_set():
    rng = np.random.default_rng(3)
    f = rng.standard_normal(5000)
    lams = np.array([0.3, 0.5, 0.7, 0.9])
    Z = tm.zscore(np.column_stack([l * f + np.sqrt(1 - l * l) * rng.standard_normal(5000) for l in lams]))
    alpha, _ = tm.cronbach(Z)
    omega = tm.omega_1factor(np.corrcoef(Z, rowvar=False))
    true_omega = lams.sum() ** 2 / (lams.sum() ** 2 + (1 - lams ** 2).sum())
    assert alpha < omega < 1 and omega == pytest.approx(true_omega, abs=0.03)


def test_glb_algebraic_equals_alpha_under_compound_symmetry():
    """Closed form: with every inter-item r equal, the minimum-trace decomposition is r·11' — GLB = α."""
    k, r = 11, 0.16
    R = np.full((k, k), r); np.fill_diagonal(R, 1)
    glb, psi = tm.glb_algebraic(R)
    assert glb == pytest.approx(k * r / (1 + (k - 1) * r), abs=1e-8)
    assert np.linalg.eigvalsh(R - np.diag(psi)).min() > -1e-8                  # feasible: R − Ψ ⪰ 0


def test_glb_algebraic_bounds_alpha_and_omega_from_above_on_data():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((168, 11)) @ rng.standard_normal((11, 11)) * 0.3 + rng.standard_normal((168, 11))
    Z = tm.zscore(X); R = np.corrcoef(Z, rowvar=False)
    glb = tm.glb_algebraic(R)[0]
    assert tm.cronbach(Z)[0] <= glb <= 1.0                                      # α ≤ GLB (Jackson & Agunwamba)


def test_minres_port_recovers_population_communalities_and_glb_fa_tracks_the_bound():
    lam = np.array([0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    R = np.outer(lam, lam); np.fill_diagonal(R, 1)                               # exact one-factor matrix
    _, h2, values = tm.minres_fa(R, 1)
    assert np.abs(h2 - lam ** 2).max() < 1e-4 and values[0] == pytest.approx((lam ** 2).sum(), abs=1e-3)
    rng = np.random.default_rng(5)                                               # two-cluster sample: nf is unambiguous
    f = rng.standard_normal((2000, 2))
    X = np.column_stack([0.7 * f[:, j // 6] + 0.7 * rng.standard_normal(2000) for j in range(11)])
    Rs = np.corrcoef(X, rowvar=False)
    g, nf = tm.glb_fa(Rs)
    assert 1 <= nf <= 6 and 0 < g <= 1
    assert abs(g - tm.glb_algebraic(Rs)[0]) < 0.1                                # the FA route approximates the bound
    assert abs(g - tm.glb_fa(Rs, exact=True)[0]) < 0.05                          # psych's gradient vs the true minres optimum


@pytest.mark.skipif(not (REPO / "analysis/c_alpha/misc_variables_paper_clean.csv").is_file(), reason="MISC table not built")
def test_method_reproduces_table2_on_misc(tmp_path):
    """External known answer: on the paper's own data the method must land on the paper's component."""
    tm.HERE = tmp_path
    tm.main(table=REPO / "analysis/c_alpha/misc_variables_paper_clean.csv", out_prefix="m", id_cols=("pair", "participant", "task"), min_utts=0, outlier_z=5.0)
    r = pd.read_csv(tmp_path / "m_results.csv").set_index("key").value
    assert float(r["congruence_phi_ours_vs_thomas_pc1"]) >= 0.97
    assert int(r["sign_agreement_k_of_11"]) >= 10
    assert 0.20 <= float(r["pc1_variance_share"]) <= 0.32
    assert float(r["corr_involvement_ours_vs_thomas_weights"]) >= 0.98
