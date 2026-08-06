"""Disproportionality measures for spontaneous reporting data.

Everything here operates on the 2x2 table formed for a drug *i* and an event *j*:

===============  ==========================  ==============================
                 event *j*                   any other event
drug *i*         ``a`` = N(i,j)              ``b``
any other drug   ``c``                       ``d``
===============  ==========================  ==============================

Four conventions this module follows, each of which the 2020 notebook did not:

**Lower bounds, not point estimates.** A signal is declared on the lower confidence or credible
bound -- ROR025, PRR025, IC025, EB05 -- never on the raw ratio. A point estimate of 8 computed from
``a = 2`` is noise, and screening on it is the single most common way to generate spurious signals.

**Vectorized throughout.** These run over ~10^7 drug-event pairs. Fisher's exact test uses the
hypergeometric survival function rather than a per-pair call, which is the difference between
seconds and hours.

**No causal language.** These measures quantify *disproportionate reporting*: an event is reported
with a drug more often than the rest of the database would predict. That is a hypothesis-generating
observation about a voluntary reporting corpus, not an estimate of risk, incidence, or causation.

**A stated hypothesis family.** Benjamini-Hochberg is applied once per analysis cohort, over pairs
meeting the minimum-support rule, and the family is recorded alongside the results.

References
----------
van Puijenbroek et al. (2002) *Pharmacoepidemiol Drug Saf* 11:3-10 -- comparison of measures.
DuMouchel (1999) *The American Statistician* 53:177-190 -- gamma-Poisson shrinkage (MGPS).
Noren et al. (2006) *Stat Med* 25:3740-3757 -- IC credible interval approximation.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

#: Below this many co-reports an estimate is too unstable to screen on. Applied *before* multiple
#: testing correction: including the long tail of single-occurrence pairs in the family destroys
#: power without adding information.
DEFAULT_MIN_COUNT = 3

#: Added to zero cells for the asymptotic ratio measures only (Haldane-Anscombe). The Bayesian
#: measures need no such correction -- their priors handle sparsity natively.
HALDANE = 0.5

Z_95 = 1.959963984540054


def _as_float_arrays(*arrays) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(x, dtype=np.float64) for x in arrays)


def contingency(n_ij, n_i, n_j, n_total):
    """Expand marginal counts into the four cells of the 2x2 table.

    ``n_i`` is the total reports mentioning the drug, ``n_j`` the total mentioning the event, and
    ``n_total`` the number of reports in the cohort -- *not* the number of drug-event pairs. Using
    a pair count as the denominator inflates every expected value and is a classic error.
    """
    a, n_i, n_j, n_total = _as_float_arrays(n_ij, n_i, n_j, n_total)
    b = n_i - a
    c = n_j - a
    d = n_total - a - b - c
    return a, b, c, d


def ror(a, b, c, d, correction: float = HALDANE):
    """Reporting odds ratio with a two-sided 95% confidence interval.

    Returns ``(ror, lower, upper)``. Zero cells receive a Haldane-Anscombe correction so the
    estimator stays defined; the original notebook instead asserted all cells were non-zero, which
    silently excluded exactly the sparse pairs that most need care.
    """
    a, b, c, d = _as_float_arrays(a, b, c, d)
    has_zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    ac, bc, cc, dc = (x + np.where(has_zero, correction, 0.0) for x in (a, b, c, d))

    log_ror = np.log(ac) + np.log(dc) - np.log(bc) - np.log(cc)
    se = np.sqrt(1 / ac + 1 / bc + 1 / cc + 1 / dc)
    return np.exp(log_ror), np.exp(log_ror - Z_95 * se), np.exp(log_ror + Z_95 * se)


def prr(a, b, c, d, correction: float = HALDANE):
    """Proportional reporting ratio with a two-sided 95% confidence interval.

    Note the variance: ``1/a - 1/(a+b) + 1/c - 1/(c+d)``. Both subtractions are required. The 2020
    notebook used ``+ 1/(c+d)``, which inflates the standard error and makes the interval
    conservative in a way that varies with the size of the comparator group.
    """
    a, b, c, d = _as_float_arrays(a, b, c, d)
    has_zero = (a == 0) | (c == 0)
    ac, bc, cc, dc = (x + np.where(has_zero, correction, 0.0) for x in (a, b, c, d))

    exposed, comparator = ac + bc, cc + dc
    log_prr = np.log(ac / exposed) - np.log(cc / comparator)
    se = np.sqrt(1 / ac - 1 / exposed + 1 / cc - 1 / comparator)
    return np.exp(log_prr), np.exp(log_prr - Z_95 * se), np.exp(log_prr + Z_95 * se)


def chi2_yates(a, b, c, d):
    """Yates-corrected chi-square, the companion statistic to PRR in the MHRA screening rule."""
    a, b, c, d = _as_float_arrays(a, b, c, d)
    n = a + b + c + d
    num = n * np.maximum(np.abs(a * d - b * c) - n / 2, 0.0) ** 2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def fisher_exact_greater(a, b, c, d):
    """One-sided (over-reporting) Fisher exact p-values, vectorized.

    ``P(X >= a)`` under the hypergeometric null equals ``sf(a - 1)``. Using the survival function
    rather than ``scipy.stats.fisher_exact`` per pair is what makes this tractable at 10^7 pairs,
    and it avoids the underflow that would corrupt the Benjamini-Hochberg ordering.
    """
    a, b, c, d = (np.asarray(x, dtype=np.int64) for x in (a, b, c, d))
    total = a + b + c + d
    return stats.hypergeom.sf(a - 1, total, a + b, a + c)


def information_component(a, n_i, n_j, n_total):
    """Bayesian confidence propagation neural network information component.

    Returns ``(ic, ic025, ic975)``. Uses the Noren (2006) shrinkage form with the standard
    asymptotic credible interval; ``IC025 > 0`` is the screening criterion used by the WHO
    Uppsala Monitoring Centre.
    """
    a, n_i, n_j, n_total = _as_float_arrays(a, n_i, n_j, n_total)
    expected = n_i * n_j / n_total
    shrunk = (a + 0.5) / (expected + 0.5)
    ic = np.log2(shrunk)

    inv_sqrt = (a + 0.5) ** -0.5
    inv_1p5 = (a + 0.5) ** -1.5
    return ic, ic - 3.3 * inv_sqrt - 2.0 * inv_1p5, ic + 2.4 * inv_sqrt - 0.5 * inv_1p5


# ------------------------------------------------------------------------------------------------
# Gamma-Poisson shrinkage (DuMouchel's MGPS)
# ------------------------------------------------------------------------------------------------


def _neg_log_marginal(theta: np.ndarray, n: np.ndarray, e: np.ndarray) -> float:
    """Negative log marginal likelihood of the two-component gamma-Poisson mixture.

    Parameters arrive log-transformed (and the weight logit-transformed) so the optimizer runs
    unconstrained; this is what keeps the five-parameter fit from wandering into invalid regions,
    which is the usual failure mode of a naive box-constrained fit.
    """
    a1, b1, a2, b2 = np.exp(theta[:4])
    w = special.expit(theta[4])

    def log_nb(alpha, beta):
        # NegBin(alpha, beta/(beta+E)) evaluated at n, in log space.
        return (
            special.gammaln(alpha + n)
            - special.gammaln(alpha)
            - special.gammaln(n + 1)
            + alpha * np.log(beta / (beta + e))
            + n * np.log(e / (beta + e))
        )

    ll = np.logaddexp(np.log(w) + log_nb(a1, b1), np.log1p(-w) + log_nb(a2, b2))
    if not np.isfinite(ll).all():
        return np.inf
    return -float(ll.sum())


def fit_gamma_poisson(
    n: np.ndarray, e: np.ndarray, seeds: int = 4, random_state: int = 0
) -> dict:
    """Fit the five hyperparameters by maximum marginal likelihood.

    Multiple starting points are used because the mixture likelihood is multimodal; a single
    start regularly converges to a degenerate solution where both components collapse together.
    The fit is reported with its convergence status so a failure cannot pass silently.
    """
    n, e = _as_float_arrays(n, e)
    ok = np.isfinite(n) & np.isfinite(e) & (e > 0)
    n, e = n[ok], e[ok]
    if n.size == 0:
        raise ValueError("no usable (N, E) pairs for the gamma-Poisson fit")

    rng = np.random.default_rng(random_state)
    starts = [np.array([np.log(0.2), np.log(0.1), np.log(2.0), np.log(4.0), 0.0])]
    for _ in range(seeds - 1):
        starts.append(
            np.array(
                [
                    np.log(rng.uniform(0.05, 1.0)),
                    np.log(rng.uniform(0.05, 1.0)),
                    np.log(rng.uniform(1.0, 8.0)),
                    np.log(rng.uniform(1.0, 8.0)),
                    rng.uniform(-1.0, 1.0),
                ]
            )
        )

    best = None
    for x0 in starts:
        try:
            res = optimize.minimize(
                _neg_log_marginal, x0, args=(n, e), method="Nelder-Mead",
                options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6},
            )
        except (FloatingPointError, ValueError):
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        raise RuntimeError("gamma-Poisson mixture failed to converge from any starting point")

    a1, b1, a2, b2 = np.exp(best.x[:4])
    w = float(special.expit(best.x[4]))
    # Order components so the low-lambda ("background") component is always first, making fitted
    # hyperparameters comparable between runs and cohorts.
    if a1 / b1 > a2 / b2:
        a1, b1, a2, b2, w = a2, b2, a1, b1, 1 - w
    return {
        "alpha1": float(a1), "beta1": float(b1),
        "alpha2": float(a2), "beta2": float(b2),
        "weight": w,
        "neg_log_likelihood": float(best.fun),
        "converged": bool(best.success),
        "n_pairs_fitted": int(n.size),
    }


def _posterior_weight(n, e, p) -> np.ndarray:
    """Posterior probability that a pair came from the first mixture component."""
    def log_nb(alpha, beta):
        return (
            special.gammaln(alpha + n) - special.gammaln(alpha) - special.gammaln(n + 1)
            + alpha * np.log(beta / (beta + e)) + n * np.log(e / (beta + e))
        )

    l1 = np.log(p["weight"]) + log_nb(p["alpha1"], p["beta1"])
    l2 = np.log1p(-p["weight"]) + log_nb(p["alpha2"], p["beta2"])
    return np.exp(l1 - np.logaddexp(l1, l2))


def ebgm(n, e, params: dict, quantiles=(0.05, 0.95)):
    """Empirical Bayes geometric mean and posterior quantiles.

    Returns ``(ebgm, eb05, eb95)``. EBGM is the posterior geometric mean of the reporting ratio;
    ``EB05 >= 2`` is the conventional MGPS screening threshold. Reporting EBGM without EB05 would
    omit precisely the uncertainty that makes the shrinkage worthwhile.
    """
    n, e = _as_float_arrays(n, e)
    q = _posterior_weight(n, e, params)

    a1n, b1e = params["alpha1"] + n, params["beta1"] + e
    a2n, b2e = params["alpha2"] + n, params["beta2"] + e

    log_gm = q * (special.digamma(a1n) - np.log(b1e)) + (1 - q) * (
        special.digamma(a2n) - np.log(b2e)
    )
    point = np.exp(log_gm)

    lo = _mixture_quantile(q, a1n, b1e, a2n, b2e, quantiles[0])
    hi = _mixture_quantile(q, a1n, b1e, a2n, b2e, quantiles[1])
    return point, lo, hi


def _mixture_quantile(q, a1, b1, a2, b2, prob, iters: int = 60) -> np.ndarray:
    """Quantile of a two-component gamma posterior mixture, by vectorized bisection.

    The mixture CDF has no closed-form inverse. Bisection is used rather than a Newton method
    because it cannot diverge, and 60 halvings over a bracket that starts at the component-wise
    quantile range is ample precision for a screening statistic.
    """
    lo = np.minimum(stats.gamma.ppf(prob, a1, scale=1 / b1),
                    stats.gamma.ppf(prob, a2, scale=1 / b2))
    hi = np.maximum(stats.gamma.ppf(prob, a1, scale=1 / b1),
                    stats.gamma.ppf(prob, a2, scale=1 / b2))
    lo = np.nan_to_num(lo, nan=0.0, posinf=0.0)
    hi = np.nan_to_num(hi, nan=0.0, posinf=0.0)
    lo = np.maximum(lo * 0.5, 0.0)
    hi = hi * 2.0 + 1e-9

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = q * stats.gamma.cdf(mid, a1, scale=1 / b1) + (1 - q) * stats.gamma.cdf(
            mid, a2, scale=1 / b2
        )
        too_low = cdf < prob
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------------------------
# Multiplicity
# ------------------------------------------------------------------------------------------------


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), vectorized and monotone.

    BH controls the false discovery rate under independence or positive regression dependence.
    Drug-event tests over a shared corpus are *not* independent -- a report contributes to every
    drug-event pair it contains -- so these q-values are approximate. The report states this rather
    than presenting the level as exact.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out
