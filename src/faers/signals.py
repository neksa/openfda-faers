"""Stage 6: build drug-event contingency counts and score them.

The join at the centre of this module is the one that decides whether the pipeline runs at all.
For every report we need the set of (ingredient, reaction) pairs it contributes, then a count of
distinct reports per pair. Done naively -- joining the ~90M-row drug table to the ~70M-row reaction
table and only then reducing -- the intermediate is billions of rows and the process dies.

Three things keep it bounded:

* **Deduplicate before joining.** A report that lists the same ingredient three times (different
  products, doses or routes) must contribute *one* edge, not three. Reducing both sides to distinct
  ``(record_id, key)`` pairs first shrinks the input and is also the statistically correct unit --
  the contingency table counts reports, not table rows.
* **Aggregate inside the join.** DuckDB streams the ``GROUP BY`` rather than materializing the
  crossed rows, spilling to disk when needed instead of exhausting RAM.
* **Filter to suspect roles by default.** Restricting to primary and secondary suspect drugs is
  both the pharmacovigilance convention and a large reduction in fan-out. It is a parameter, not a
  hardcoded choice, and the sensitivity of the results to it is reported.

``n_total`` is the number of distinct **reports** in the cohort. Using a pair count instead would
inflate every expected value and quietly invalidate every measure computed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from .stats import disproportionality as D

#: Ceiling on DuckDB's working memory. Left well below physical RAM so the rest of the stage
#: (numpy scoring over the resulting arrays) has room.
DEFAULT_MEMORY_LIMIT = "24GB"


def connect(memory_limit: str = DEFAULT_MEMORY_LIMIT, temp_dir: str | Path = "data/tmp"):
    """A DuckDB connection configured to spill rather than fail."""
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{temp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def measure_fanout(con, drug_path: Path, reac_path: Path, roles: tuple[str, ...] | None) -> dict:
    """Report the join's actual shape before attempting it.

    Cheap insurance: if a handful of pathological reports carry hundreds of drugs and hundreds of
    reactions, the pair count explodes quadratically and it is far better to know that from a
    diagnostic than from an out-of-memory kill an hour later.
    """
    role_filter = _role_clause(roles)
    q = f"""
    WITH d AS (
        SELECT record_id, COUNT(DISTINCT ingredient) AS n_drugs
        FROM read_parquet('{drug_path}') WHERE ingredient IS NOT NULL {role_filter}
        GROUP BY record_id
    ), r AS (
        SELECT record_id, COUNT(DISTINCT pt) AS n_reacs
        FROM read_parquet('{reac_path}') WHERE pt IS NOT NULL
        GROUP BY record_id
    ), j AS (
        SELECT d.record_id, d.n_drugs, r.n_reacs, d.n_drugs * r.n_reacs AS n_pairs
        FROM d JOIN r USING (record_id)
    )
    SELECT COUNT(*) AS n_reports, SUM(n_pairs) AS total_pairs,
           MAX(n_drugs) AS max_drugs, MAX(n_reacs) AS max_reacs, MAX(n_pairs) AS max_pairs,
           quantile_cont(n_drugs, 0.5) AS median_drugs,
           quantile_cont(n_reacs, 0.5) AS median_reacs,
           quantile_cont(n_pairs, 0.99) AS p99_pairs
    FROM j
    """
    row = con.execute(q).fetchone()
    keys = ["n_reports", "total_pairs", "max_drugs", "max_reacs", "max_pairs",
            "median_drugs", "median_reacs", "p99_pairs"]
    return {k: (float(v) if v is not None else None) for k, v in zip(keys, row, strict=True)}


def _role_clause(roles: tuple[str, ...] | None) -> str:
    if not roles:
        return ""
    quoted = ", ".join(f"'{r}'" for r in roles)
    return f"AND role_cod IN ({quoted})"


def build_counts(
    con,
    drug_path: Path,
    reac_path: Path,
    demo_path: Path,
    out_path: Path,
    roles: tuple[str, ...] | None = ("PS", "SS"),
    min_count: int = D.DEFAULT_MIN_COUNT,
) -> dict:
    """Write the (ingredient, reaction, n_ij, n_i, n_j) table and return cohort totals."""
    role_filter = _role_clause(roles)

    con.execute(
        f"""
        CREATE OR REPLACE VIEW cohort AS
            SELECT DISTINCT record_id FROM read_parquet('{demo_path}');
        CREATE OR REPLACE VIEW dr AS
            SELECT DISTINCT record_id, ingredient
            FROM read_parquet('{drug_path}')
            WHERE ingredient IS NOT NULL {role_filter}
              AND record_id IN (SELECT record_id FROM cohort);
        CREATE OR REPLACE VIEW rc AS
            SELECT DISTINCT record_id, pt
            FROM read_parquet('{reac_path}')
            WHERE pt IS NOT NULL
              AND record_id IN (SELECT record_id FROM cohort);
        """
    )

    # n_total must count reports that could contribute a pair at all: those with at least one
    # in-scope drug and at least one reaction. Counting the whole corpus here would understate
    # every expected count for cohorts where the role filter excludes many reports entirely.
    n_total = con.execute(
        "SELECT COUNT(*) FROM (SELECT record_id FROM dr INTERSECT SELECT record_id FROM rc)"
    ).fetchone()[0]

    con.execute(
        f"""
        COPY (
            WITH pair AS (
                SELECT dr.ingredient, rc.pt, COUNT(*) AS n_ij
                FROM dr JOIN rc USING (record_id)
                GROUP BY 1, 2
                HAVING COUNT(*) >= {int(min_count)}
            ),
            di AS (SELECT ingredient, COUNT(*) AS n_i FROM dr GROUP BY 1),
            rj AS (SELECT pt, COUNT(*) AS n_j FROM rc GROUP BY 1)
            SELECT p.ingredient, p.pt, p.n_ij, di.n_i, rj.n_j
            FROM pair p JOIN di USING (ingredient) JOIN rj USING (pt)
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    n_pairs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    return {
        "n_total_reports": int(n_total),
        "n_pairs_retained": int(n_pairs),
        "min_count": int(min_count),
        "roles": list(roles) if roles else "all",
    }


def score(
    counts_path: Path,
    out_path: Path,
    n_total: int,
    params_path: Path | None = None,
    fit_sample: int = 200_000,
    random_state: int = 0,
) -> dict:
    """Attach every disproportionality measure to the contingency table.

    The gamma-Poisson hyperparameters are fitted on a random sample when the pair table is large:
    the fit estimates five numbers describing the *distribution* of reporting ratios, and a
    200k-pair sample pins those down as well as the full table while keeping the Nelder-Mead
    search to seconds. The sample size and seed are recorded so the fit is reproducible.
    """
    df = pl.read_parquet(counts_path)
    a = df["n_ij"].to_numpy().astype(np.float64)
    n_i = df["n_i"].to_numpy().astype(np.float64)
    n_j = df["n_j"].to_numpy().astype(np.float64)

    a_c, b, c, d = D.contingency(a, n_i, n_j, float(n_total))
    expected = n_i * n_j / float(n_total)

    ror, ror_lo, ror_hi = D.ror(a_c, b, c, d)
    prr, prr_lo, prr_hi = D.prr(a_c, b, c, d)
    chi2 = D.chi2_yates(a_c, b, c, d)
    p = D.fisher_exact_greater(a_c, b, c, d)
    q = D.benjamini_hochberg(p)
    ic, ic_lo, ic_hi = D.information_component(a, n_i, n_j, float(n_total))

    rng = np.random.default_rng(random_state)
    idx = (
        rng.choice(a.size, size=min(fit_sample, a.size), replace=False)
        if a.size > fit_sample
        else np.arange(a.size)
    )
    gp = D.fit_gamma_poisson(a[idx], expected[idx], seeds=4, random_state=random_state)
    gp["fit_sample_size"] = int(idx.size)
    gp["fit_random_state"] = int(random_state)
    eb, eb05, eb95 = D.ebgm(a, expected, gp)

    out = df.with_columns(
        [
            pl.Series("expected", expected),
            pl.Series("ror", ror), pl.Series("ror025", ror_lo), pl.Series("ror975", ror_hi),
            pl.Series("prr", prr), pl.Series("prr025", prr_lo), pl.Series("prr975", prr_hi),
            pl.Series("chi2_yates", chi2),
            pl.Series("p_fisher", p), pl.Series("q_bh", q),
            pl.Series("ic", ic), pl.Series("ic025", ic_lo), pl.Series("ic975", ic_hi),
            pl.Series("ebgm", eb), pl.Series("eb05", eb05), pl.Series("eb95", eb95),
        ]
    ).with_columns(
        # The conventional screening rules, kept as explicit flags so the report can compare how
        # much they agree rather than presenting one as the answer.
        (pl.col("ror025") > 1.0).alias("signal_ror"),
        ((pl.col("prr") >= 2.0) & (pl.col("chi2_yates") >= 4.0) & (pl.col("n_ij") >= 3)).alias(
            "signal_prr_mhra"
        ),
        (pl.col("ic025") > 0.0).alias("signal_ic"),
        (pl.col("eb05") >= 2.0).alias("signal_eb05"),
        (pl.col("q_bh") < 0.05).alias("signal_fdr"),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path, compression="zstd")

    summary = {
        "n_pairs": out.height,
        "n_total_reports": int(n_total),
        "gamma_poisson": gp,
        "signals": {
            name: int(out[name].sum())
            for name in ("signal_ror", "signal_prr_mhra", "signal_ic", "signal_eb05", "signal_fdr")
        },
        "concordant_all_four": int(
            (out["signal_ror"] & out["signal_prr_mhra"] & out["signal_ic"] & out["signal_eb05"])
            .sum()
        ),
        "hypothesis_family": (
            "One Benjamini-Hochberg family per cohort, over ingredient-reaction pairs with "
            f"n_ij >= {int(df['n_ij'].min())} in that cohort. q-values are approximate because "
            "pairs share reports and are therefore dependent."
        ),
    }
    if params_path:
        Path(params_path).parent.mkdir(parents=True, exist_ok=True)
        Path(params_path).write_text(json.dumps(summary, indent=2) + "\n")
    return summary
