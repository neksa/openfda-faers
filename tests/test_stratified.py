"""Validation of the Mantel-Haenszel adjustment and the Breslow-Day homogeneity test.

These are calibration tests against simulated data with a *known* odds ratio, not consistency
checks. They matter because the headline stratified result -- that adjustment removes almost no
signals while Breslow-Day rejects homogeneity for most of them -- is only interpretable if the
homogeneity test has roughly nominal type-I error. A test that always rejects would produce the
same finding for a completely different, and wrong, reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from faers.stats.stratified import breslow_day, mantel_haenszel


def simulate(n_pairs, n_strata, odds_ratios, seed, n_exposed=800, n_unexposed=4000):
    """Build stratified 2x2 tables with a prescribed odds ratio per stratum."""
    rng = np.random.default_rng(seed)
    shape = (n_pairs, n_strata)
    a, b, c, d = (np.zeros(shape) for _ in range(4))
    for i in range(n_pairs):
        for j in range(n_strata):
            baseline = rng.uniform(0.02, 0.08)
            odds0 = baseline / (1 - baseline)
            psi = odds_ratios[j]
            p_exposed = psi * odds0 / (1 + psi * odds0)
            ai = rng.binomial(n_exposed, p_exposed)
            ci = rng.binomial(n_unexposed, baseline)
            a[i, j], b[i, j] = ai, n_exposed - ai
            c[i, j], d[i, j] = ci, n_unexposed - ci
    return a, b, c, d


class TestMantelHaenszel:
    def test_recovers_known_common_odds_ratio(self):
        a, b, c, d = simulate(300, 5, [3.0] * 5, seed=0)
        or_mh, lo, hi, n_strata = mantel_haenszel(a, b, c, d)
        assert np.median(or_mh) == pytest.approx(3.0, rel=0.05)
        assert np.all(n_strata == 5)

    def test_interval_covers_truth_at_roughly_nominal_rate(self):
        a, b, c, d = simulate(300, 4, [2.5] * 4, seed=1)
        _, lo, hi, _ = mantel_haenszel(a, b, c, d)
        covered = ((lo <= 2.5) & (2.5 <= hi)).mean()
        assert 0.90 <= covered <= 0.99

    def test_null_association_recovers_unity(self):
        a, b, c, d = simulate(200, 3, [1.0] * 3, seed=2)
        or_mh, lo, hi, _ = mantel_haenszel(a, b, c, d)
        assert np.median(or_mh) == pytest.approx(1.0, rel=0.08)
        assert ((lo <= 1.0) & (1.0 <= hi)).mean() > 0.90

    def test_zero_cells_stay_estimable(self):
        a = np.array([[0.0, 5.0]])
        b = np.array([[100.0, 95.0]])
        c = np.array([[10.0, 8.0]])
        d = np.array([[890.0, 892.0]])
        or_mh, lo, hi, _ = mantel_haenszel(a, b, c, d)
        assert np.isfinite(or_mh).all() and np.isfinite(lo).all() and np.isfinite(hi).all()


class TestBreslowDay:
    def test_type_one_error_is_near_nominal(self):
        """Homogeneous strata must not be flagged much more than 5% of the time."""
        a, b, c, d = simulate(400, 5, [3.0] * 5, seed=10)
        or_mh, _, _, _ = mantel_haenszel(a, b, c, d)
        _, p = breslow_day(a, b, c, d, or_mh)
        assert 0.01 <= (p < 0.05).mean() <= 0.12

    def test_detects_real_heterogeneity(self):
        a, b, c, d = simulate(400, 5, np.linspace(1.0, 9.0, 5), seed=11)
        or_mh, _, _, _ = mantel_haenszel(a, b, c, d)
        _, p = breslow_day(a, b, c, d, or_mh)
        assert (p < 0.05).mean() > 0.95

    def test_pvalues_are_valid_probabilities(self):
        a, b, c, d = simulate(100, 4, [2.0] * 4, seed=12)
        or_mh, _, _, _ = mantel_haenszel(a, b, c, d)
        stat, p = breslow_day(a, b, c, d, or_mh)
        assert np.all((p >= 0) & (p <= 1))
        assert np.all(stat >= 0)
