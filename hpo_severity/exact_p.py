"""
Exact P values for the statistical tests reported in the manuscript.

Several P values are smaller than the smallest number a float can hold (about 1e-308), so scipy
returns 0 for them. Each function here returns log10(P), computed on the log scale with the same
test and approximation that scipy uses, so every P value can be reported exactly:

- chi-square tests of independence and Kruskal-Wallis H: chi-square survival function of the
  statistic with its degrees of freedom (as scipy.stats.chi2_contingency and scipy.stats.kruskal);
- Pearson r and the slope of a least-squares line: two-sided t test on n - 2 degrees of freedom (as
  scipy.stats.pearsonr and scipy.stats.linregress);
- Mann-Whitney U: two-sided normal approximation with tie and continuity correction (as
  scipy.stats.mannwhitneyu with method='asymptotic');
- McNemar: exact two-sided binomial test on the discordant pairs.

record() keeps each test so that write() can save them all to exact_p_values.csv, the file the
manuscript P values are copied from and checked against.
"""
import math

import numpy as np
import pandas as pd
from scipy import special, stats

LN10 = math.log(10)
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_ROWS = {}


def _lentz(next_terms, tiny=1e-300, eps=1e-15, max_iter=100000):
    """Modified Lentz evaluation of the continued fraction 1 / (b0 + a1 / (b1 + a2 / (b2 + ...)))."""
    terms = next_terms()
    b = next(terms)
    c, d = 1 / tiny, 1 / (b if abs(b) > tiny else tiny)
    h = d
    for _ in range(max_iter):
        a, b = next(terms)
        d = b + a * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = b + a / c
        c = c if abs(c) > tiny else tiny
        h *= c * d
        if abs(c * d - 1) < eps:
            return h
    raise RuntimeError("continued fraction did not converge")


def _log_gamma_upper(a, x):
    """log of the regularized upper incomplete gamma function Q(a, x)."""
    if x < a + 1:
        return math.log(special.gammaincc(a, x))

    def terms():
        b = x + 1 - a
        yield b
        i = 0
        while True:
            i += 1
            b += 2
            yield -i * (i - a), b
    return -x + a * math.log(x) - math.lgamma(a) + math.log(_lentz(terms))


def _log_beta_lower(a, b, x):
    """log of the regularized incomplete beta function I_x(a, b), for x < (a + 1) / (a + b + 2)."""
    def terms():
        yield 1.0
        m = 0
        while True:
            m += 1
            yield -(a + m - 1) * (a + b + m - 1) * x / ((a + 2 * m - 2) * (a + 2 * m - 1)), 1.0
            yield m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)), 1.0
    front = a * math.log(x) + b * math.log1p(-x) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)) - math.log(a)
    return front + math.log(_lentz(terms))


def chi2(statistic, dof):
    """log10 P of a chi-square statistic (also the Kruskal-Wallis H) with dof degrees of freedom."""
    return min(0.0, _log_gamma_upper(dof / 2, statistic / 2) / LN10)


def r(coefficient, n):
    """log10 of the two-sided P of a correlation coefficient or a regression slope from n points."""
    if abs(coefficient) >= 1:
        return -math.inf
    df = n - 2
    t2 = coefficient ** 2 * df / (1 - coefficient ** 2)
    x = df / (df + t2)
    if x < (df / 2 + 1) / (df / 2 + 0.5 + 2):
        log_two_sided = _log_beta_lower(df / 2, 0.5, x)
    else:
        log_two_sided = math.log(special.betainc(df / 2, 0.5, x))
    return min(0.0, log_two_sided / LN10)


def mann_whitney(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    n = n1 + n2
    ranks = stats.rankdata(np.concatenate([x, y]))
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    u = max(u1, n1 * n2 - u1)
    _, ties = np.unique(ranks, return_counts=True)
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - (ties ** 3 - ties).sum() / (n * (n - 1))))
    z = (u - n1 * n2 / 2 - 0.5) / sd
    return min(0.0, (math.log(2) + float(stats.norm.logsf(z))) / LN10)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0
    return min(0.0, (math.log(2) + float(stats.binom.logcdf(min(b, c), n, 0.5))) / LN10)


def fmt(log10p):
    """P to two significant figures: 0.0057 or 1.7 × 10⁻³⁰."""
    if log10p >= -3:
        p = 10 ** log10p
        if p >= 0.995:
            return "1.0"
        decimals = max(0, 1 - math.floor(math.log10(p)))
        return f"{round(p, decimals):.{decimals}f}"
    e = math.floor(log10p)
    m = round(10 ** (log10p - e), 1)
    if m >= 10:
        m, e = 1.0, e + 1
    return f"{m:.1f} × 10{str(e).translate(_SUP)}"


def fmt_tex(log10p):
    """P for figure text, with the exponent as matplotlib math text: $P$ = 1.1 × $10^{-8}$."""
    text = fmt(log10p)
    if "×" in text:
        mantissa, exp = text.split(" × 10")
        exp = exp.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-"))
        text = f"{mantissa} × $10^{{{exp}}}$"
    return f"$P$ = {text}"


def reset():
    _ROWS.clear()


def record(key, test, statistic, log10p, n=None):
    """Keeps the first result recorded under key and returns its formatted P."""
    _ROWS.setdefault(key, {"key": key, "test": test, "statistic": statistic, "n": n,
                           "log10_p": round(log10p, 4), "p": fmt(log10p)})
    return fmt(log10p)


def write(path):
    pd.DataFrame(list(_ROWS.values())).to_csv(path, index=False, encoding="utf-8")
