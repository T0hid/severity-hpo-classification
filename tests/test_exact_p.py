"""Exact P values: agreement with scipy where scipy can represent the P value, and the printed format."""
import math

import numpy as np
from scipy import stats

from hpo_severity import exact_p as XP


def test_chi_square_and_kruskal_match_scipy():
    for stat, dof in ((30.93, 12), (39.88, 3), (721.32, 3), (1020.84, 12), (5.0, 3)):
        assert math.isclose(10 ** XP.chi2(stat, dof), stats.chi2.sf(stat, dof), rel_tol=1e-6)


def test_correlation_and_mann_whitney_match_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 0.3 * x + rng.normal(size=400)
    r, p = stats.pearsonr(x, y)
    assert math.isclose(10 ** XP.r(r, 400), p, rel_tol=1e-6)
    a, b = rng.integers(0, 5, 3000), rng.integers(0, 4, 2500)
    assert math.isclose(10 ** XP.mann_whitney(a, b), stats.mannwhitneyu(a, b, alternative="two-sided").pvalue, rel_tol=1e-6)


def test_mcnemar_matches_exact_binomial():
    p = 2 * sum(math.comb(203, k) for k in range(25)) / 2 ** 203
    assert math.isclose(10 ** XP.mcnemar(179, 24), p, rel_tol=1e-9)


def test_values_below_float_range():
    assert XP.fmt(XP.chi2(3621.84, 48)) == "1.1 × 10⁻⁷³⁴"


def test_format():
    assert XP.fmt(math.log10(0.0057)) == "0.0057"
    assert XP.fmt(math.log10(0.002019)) == "0.0020"
    assert XP.fmt(math.log10(1.13e-8)) == "1.1 × 10⁻⁸"
    assert XP.fmt_tex(math.log10(1.13e-8)) == "$P$ = 1.1 × $10^{-8}$"
