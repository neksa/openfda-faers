"""Stage 7: Mantel-Haenszel adjusted reporting odds ratios.

The 2020 notebook identified sex and age as confounders -- "age could be a confounding factor for
the downstream analysis", "a reporting bias may skew the results" -- and then computed every
statistic unadjusted anyway, because the count API could not cross-tabulate. This stage does the
adjustment it called for, and adds calendar period, which the notebook did not consider but which
matters more over a 22-year window than either: reporting volume grew roughly tenfold, and the mix
of reporters and products changed with it.

Design decisions worth stating plainly:

* Each variable is applied as a **separate** stratification, not jointly. Joint sex x age x period
  strata are sparse enough that most pairs would lose estimability, and a pooled estimate over
  mostly-empty strata is worse than no adjustment. Separate adjustments answer "does this signal
  survive controlling for X", one X at a time -- which is the question actually being asked.
* Strata contributing less than ``min_stratum_count`` co-reports are dropped before pooling, so a
  handful of reports cannot dominate an adjusted estimate.
* Breslow-Day tests homogeneity. A pooled MH estimate assumes one common odds ratio across strata;
  when that fails, the pooled number is misleading and the report says so rather than quoting it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


def mantel_haenszel(a, b, c, d):
    """Pooled MH odds ratio with Robins-Breslow-Greenland variance.

    Each input is a 2-D array of shape ``(n_pairs, n_strata)``. Returns
    ``(or_mh, lower, upper, n_informative_strata)``.

    A Haldane-Anscombe half is added only to strata containing a zero cell, which keeps the RBG
    variance defined without perturbing well-populated strata.
    """
    a, b, c, d = (np.asarray(x, dtype=np.float64) for x in (a, b, c, d))
    has_zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a, b, c, d = (x + np.where(has_zero, 0.5, 0.0) for x in (a, b, c, d))

    n = a + b + c + d
    valid = n > 0
    n = np.where(valid, n, 1.0)

    r = np.where(valid, a * d / n, 0.0)
    s = np.where(valid, b * c / n, 0.0)
    r_sum, s_sum = r.sum(axis=1), s.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        or_mh = np.where(s_sum > 0, r_sum / s_sum, np.nan)

    p = np.where(valid, (a + d) / n, 0.0)
    q = np.where(valid, (b + c) / n, 0.0)
    var_num = (
        (p * r).sum(axis=1) / (2 * r_sum**2)
        + ((p * s) + (q * r)).sum(axis=1) / (2 * r_sum * s_sum)
        + (q * s).sum(axis=1) / (2 * s_sum**2)
    )
    se = np.sqrt(np.where(np.isfinite(var_num) & (var_num > 0), var_num, np.nan))

    log_or = np.log(or_mh)
    return (
        or_mh,
        np.exp(log_or - 1.959963984540054 * se),
        np.exp(log_or + 1.959963984540054 * se),
        valid.sum(axis=1),
    )


def breslow_day(a, b, c, d, or_mh):
    """Breslow-Day homogeneity statistic and p-value.

    A small p-value means the odds ratio differs across strata, so the pooled MH estimate is not a
    meaningful summary and the stratum-specific values should be read instead.
    """
    a, b, c, d = (np.asarray(x, dtype=np.float64) for x in (a, b, c, d))
    psi = np.asarray(or_mh, dtype=np.float64)[:, None]

    n1 = a + b  # exposed
    n2 = c + d  # unexposed
    m1 = a + c  # with event
    n = n1 + n2

    # Expected count in cell a under a common odds ratio, via the quadratic root.
    aa = psi - 1.0
    bb = -(psi * (n1 + m1) + (n2 - m1))
    cc = psi * n1 * m1
    disc = np.maximum(bb**2 - 4 * aa * cc, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.where(
            np.abs(aa) < 1e-12,
            np.where(np.abs(bb) > 0, cc / -bb, np.nan),
            (-bb - np.sqrt(disc)) / (2 * aa),
        )
    e_a = np.clip(root, 1e-9, np.minimum(n1, m1) - 1e-9)

    var = 1.0 / np.maximum(
        1.0 / np.maximum(e_a, 1e-9)
        + 1.0 / np.maximum(n1 - e_a, 1e-9)
        + 1.0 / np.maximum(m1 - e_a, 1e-9)
        + 1.0 / np.maximum(n2 - m1 + e_a, 1e-9),
        1e-12,
    )

    usable = np.isfinite(e_a) & (n > 0)
    contrib = np.where(usable, (a - e_a) ** 2 / np.maximum(var, 1e-12), 0.0)
    statistic = contrib.sum(axis=1)
    dof = np.maximum(usable.sum(axis=1) - 1, 1)
    return statistic, stats.chi2.sf(statistic, dof)


def calendar_period(years: int) -> pl.Expr:
    """Bucket the FDA receipt year into fixed-length calendar strata."""
    y = pl.col("fda_dt").dt.year()
    return ((y // years) * years).cast(pl.Int32).cast(pl.String).alias("calendar_period")


def build_strata(demo: pl.LazyFrame, variable: str, period_years: int) -> pl.LazyFrame:
    """Attach the stratum label to each report."""
    if variable == "calendar_period":
        return demo.filter(pl.col("fda_dt").is_not_null()).select(
            "record_id", calendar_period(period_years)
        )
    return demo.select("record_id", pl.col(variable).fill_null("Unknown").alias(variable))


def stratified_counts(
    con, drug_path, reac_path, strata_path, variable: str, pairs: pl.DataFrame, roles
) -> pl.DataFrame:
    """Per-stratum co-report counts and marginals for a chosen set of drug-event pairs.

    Restricting to a candidate pair list is what keeps this tractable: adjusting all 2.3M pairs
    across every stratum would multiply the work by the stratum count for no benefit, since only
    pairs that signal unadjusted are worth re-testing.
    """
    role_clause = ""
    if roles:
        role_clause = "AND role_cod IN (" + ", ".join(f"'{r}'" for r in roles) + ")"

    con.register("candidate_pairs", pairs.to_arrow())
    con.execute(
        f"""
        CREATE OR REPLACE VIEW strat AS
            SELECT record_id, CAST({variable} AS VARCHAR) AS stratum
            FROM read_parquet('{strata_path}');
        CREATE OR REPLACE VIEW dr AS
            SELECT DISTINCT record_id, ingredient FROM read_parquet('{drug_path}')
            WHERE ingredient IS NOT NULL {role_clause};
        CREATE OR REPLACE VIEW rc AS
            SELECT DISTINCT record_id, pt FROM read_parquet('{reac_path}') WHERE pt IS NOT NULL;
        """
    )

    return pl.from_arrow(
        con.execute(
            """
            WITH tot AS (
                SELECT s.stratum, COUNT(DISTINCT s.record_id) AS n_stratum
                FROM strat s
                WHERE s.record_id IN (SELECT record_id FROM dr)
                  AND s.record_id IN (SELECT record_id FROM rc)
                GROUP BY 1
            ),
            di AS (
                SELECT dr.ingredient, s.stratum, COUNT(*) AS n_i
                FROM dr JOIN strat s USING (record_id)
                WHERE dr.ingredient IN (SELECT ingredient FROM candidate_pairs)
                GROUP BY 1, 2
            ),
            rj AS (
                SELECT rc.pt, s.stratum, COUNT(*) AS n_j
                FROM rc JOIN strat s USING (record_id)
                WHERE rc.pt IN (SELECT pt FROM candidate_pairs)
                GROUP BY 1, 2
            ),
            ij AS (
                SELECT dr.ingredient, rc.pt, s.stratum, COUNT(*) AS n_ij
                FROM dr JOIN rc USING (record_id) JOIN strat s USING (record_id)
                JOIN candidate_pairs cp ON cp.ingredient = dr.ingredient AND cp.pt = rc.pt
                GROUP BY 1, 2, 3
            )
            SELECT ij.ingredient, ij.pt, ij.stratum, ij.n_ij, di.n_i, rj.n_j, tot.n_stratum
            FROM ij
            JOIN di ON di.ingredient = ij.ingredient AND di.stratum = ij.stratum
            JOIN rj ON rj.pt = ij.pt AND rj.stratum = ij.stratum
            JOIN tot ON tot.stratum = ij.stratum
            """
        ).arrow()
    )


def adjust(counts: pl.DataFrame, min_stratum_count: int = 3) -> pl.DataFrame:
    """Pool per-stratum tables into an adjusted odds ratio per drug-event pair."""
    counts = counts.filter(pl.col("n_ij") >= min_stratum_count)
    if counts.is_empty():
        return pl.DataFrame()

    strata = sorted(counts["stratum"].unique().to_list())
    pairs = counts.select("ingredient", "pt").unique().sort(["ingredient", "pt"])
    pair_idx = {(r["ingredient"], r["pt"]): i for i, r in enumerate(pairs.to_dicts())}
    strat_idx = {s: j for j, s in enumerate(strata)}

    shape = (pairs.height, len(strata))
    a = np.zeros(shape)
    n_i = np.zeros(shape)
    n_j = np.zeros(shape)
    n_s = np.zeros(shape)

    for row in counts.iter_rows(named=True):
        i = pair_idx[(row["ingredient"], row["pt"])]
        j = strat_idx[row["stratum"]]
        a[i, j] = row["n_ij"]
        n_i[i, j] = row["n_i"]
        n_j[i, j] = row["n_j"]
        n_s[i, j] = row["n_stratum"]

    b = np.maximum(n_i - a, 0.0)
    c = np.maximum(n_j - a, 0.0)
    d = np.maximum(n_s - a - b - c, 0.0)

    or_mh, lo, hi, n_strata = mantel_haenszel(a, b, c, d)
    bd_stat, bd_p = breslow_day(a, b, c, d, or_mh)

    return pairs.with_columns(
        [
            pl.Series("or_mh", or_mh),
            pl.Series("or_mh_lower", lo),
            pl.Series("or_mh_upper", hi),
            pl.Series("n_informative_strata", n_strata),
            pl.Series("breslow_day_stat", bd_stat),
            pl.Series("breslow_day_p", bd_p),
        ]
    ).with_columns(
        (pl.col("or_mh_lower") > 1.0).alias("signal_adjusted"),
        (pl.col("breslow_day_p") < 0.05).alias("heterogeneous"),
    )
