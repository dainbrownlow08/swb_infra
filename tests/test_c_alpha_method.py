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


def test_omega_brackets_alpha_for_a_congeneric_set():
    rng = np.random.default_rng(3)
    f = rng.standard_normal(5000)
    lams = np.array([0.3, 0.5, 0.7, 0.9])
    Z = tm.zscore(np.column_stack([l * f + np.sqrt(1 - l * l) * rng.standard_normal(5000) for l in lams]))
    alpha, _ = tm.cronbach(Z)
    omega = tm.omega_1factor(np.corrcoef(Z, rowvar=False))
    true_omega = lams.sum() ** 2 / (lams.sum() ** 2 + (1 - lams ** 2).sum())
    assert alpha < omega < 1 and omega == pytest.approx(true_omega, abs=0.03)


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
