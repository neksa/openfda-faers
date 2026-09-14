"""Stage 5: the corpus-description questions the 2020 notebook opened with.

Each function answers one question and returns a tidy frame that is written to ``results/`` and
plotted directly. Two differences from the notebook's versions matter:

* These run on the deduplicated corpus, so a case that was resubmitted eight times counts once.
  The notebook's equivalents counted submissions.
* Time series are indexed on **FDA receipt date**, not the quarter of the file a record was found
  in. After deduplication a case lives in the quarter of its *latest* follow-up, so file-quarter
  would systematically shift old cases forward and manufacture a spurious growth trend.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .. import schema as S


def _label(col: str, mapping: dict[str, str], out: str) -> pl.Expr:
    return pl.col(col).replace_strict(mapping, default=pl.col(col)).alias(out)


def reports_per_year(demo: pl.LazyFrame) -> pl.DataFrame:
    """Corpus growth. Indexed on FDA receipt date; see the module note on why not file quarter."""
    return (
        demo.filter(pl.col("fda_dt").is_not_null())
        .with_columns(pl.col("fda_dt").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.len().alias("reports"))
        .sort("year")
        .collect()
    )


def demographics(demo: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """Age and sex composition, overall and over time."""
    sex = demo.group_by("sex").agg(pl.len().alias("reports")).sort("reports", descending=True)
    age = (
        demo.group_by("age_band")
        .agg(pl.len().alias("reports"))
        .sort("reports", descending=True)
    )
    sex_by_year = (
        demo.filter(pl.col("fda_dt").is_not_null())
        .with_columns(pl.col("fda_dt").dt.year().alias("year"))
        .group_by(["year", "sex"])
        .agg(pl.len().alias("reports"))
        .sort(["year", "sex"])
    )
    age_by_year = (
        demo.filter(pl.col("fda_dt").is_not_null())
        .with_columns(pl.col("fda_dt").dt.year().alias("year"))
        .group_by(["year", "age_band"])
        .agg(pl.len().alias("reports"))
        .sort(["year", "age_band"])
    )
    return {
        "sex": sex.collect(),
        "age_band": age.collect(),
        "sex_by_year": sex_by_year.collect(),
        "age_by_year": age_by_year.collect(),
    }


def reporters(demo: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """Who submits reports, and from where."""
    occupation = (
        demo.with_columns(pl.col("occp_cod").fill_null("Unknown"))
        .group_by("occp_cod")
        .agg(pl.len().alias("reports"))
        .sort("reports", descending=True)
        .with_columns(_label("occp_cod", S.OCCP_LABELS, "reporter"))
    )
    country = (
        demo.with_columns(pl.col("occr_country").fill_null("Unknown"))
        .group_by("occr_country")
        .agg(pl.len().alias("reports"))
        .sort("reports", descending=True)
        .head(25)
    )
    rept_type = (
        demo.with_columns(pl.col("rept_cod").fill_null("Unknown"))
        .group_by("rept_cod")
        .agg(pl.len().alias("reports"))
        .sort("reports", descending=True)
        .with_columns(_label("rept_cod", S.REPT_COD_LABELS, "report_type"))
    )
    occupation_by_year = (
        demo.filter(pl.col("fda_dt").is_not_null())
        .with_columns(
            pl.col("fda_dt").dt.year().alias("year"),
            pl.col("occp_cod").fill_null("Unknown"),
        )
        .group_by(["year", "occp_cod"])
        .agg(pl.len().alias("reports"))
        .sort(["year", "occp_cod"])
    )
    return {
        "reporter_occupation": occupation.collect(),
        "reporter_country": country.collect(),
        "report_type": rept_type.collect(),
        "reporter_by_year": occupation_by_year.collect(),
    }


def top_ingredients(drug: pl.LazyFrame, n: int = 100) -> pl.DataFrame:
    """Most-reported active ingredients, counted once per report."""
    return (
        drug.filter(pl.col("ingredient").is_not_null())
        .select("record_id", "ingredient")
        .unique()
        .group_by("ingredient")
        .agg(pl.len().alias("reports"))
        .sort("reports", descending=True)
        .head(n)
        .collect()
    )


def top_reactions(reac: pl.LazyFrame, n: int = 100) -> pl.DataFrame:
    """Most-reported MedDRA preferred terms, counted once per report."""
    return (
        reac.filter(pl.col("pt").is_not_null())
        .select("record_id", "pt")
        .unique()
        .group_by("pt")
        .agg(pl.len().alias("reports"))
        .sort("reports", descending=True)
        .head(n)
        .collect()
    )


def top_indications(indi: pl.LazyFrame, n: int = 100) -> pl.DataFrame:
    return (
        indi.filter(pl.col("indi_pt").is_not_null())
        .with_columns(pl.col("indi_pt").str.to_uppercase())
        .select("record_id", "indi_pt")
        .unique()
        .group_by("indi_pt")
        .agg(pl.len().alias("reports"))
        .sort("reports", descending=True)
        .head(n)
        .collect()
    )


def outcomes(outc: pl.LazyFrame) -> pl.DataFrame:
    """Seriousness, from the outcome codes attached to each report."""
    return (
        outc.filter(pl.col("outc_cod").is_not_null())
        .select("record_id", "outc_cod")
        .unique()
        .group_by("outc_cod")
        .agg(pl.len().alias("reports"))
        .sort("reports", descending=True)
        .with_columns(_label("outc_cod", S.OUTCOME_LABELS, "outcome"))
        .collect()
    )


def report_structure(drug: pl.LazyFrame, reac: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """Drugs and reactions per report.

    The notebook could not compute this at all -- it noted the openFDA count API "does not allow
    analytics other than counting reports with only one variable to group by". With record-level
    data it is a group-by.
    """
    per_drug = (
        drug.filter(pl.col("ingredient").is_not_null())
        .select("record_id", "ingredient")
        .unique()
        .group_by("record_id")
        .agg(pl.len().alias("n_drugs"))
        .group_by("n_drugs")
        .agg(pl.len().alias("reports"))
        .sort("n_drugs")
    )
    per_reac = (
        reac.filter(pl.col("pt").is_not_null())
        .select("record_id", "pt")
        .unique()
        .group_by("record_id")
        .agg(pl.len().alias("n_reactions"))
        .group_by("n_reactions")
        .agg(pl.len().alias("reports"))
        .sort("n_reactions")
    )
    return {"drugs_per_report": per_drug.collect(), "reactions_per_report": per_reac.collect()}


def challenge_table(drug: pl.LazyFrame) -> pl.DataFrame:
    """Joint distribution of drug role, dechallenge and rechallenge outcome.

    This is the notebook's probability-table section, which it left unfinished ("more data
    preprocessing needed") because the API returned the three variables only as separate marginal
    counts. Here they are cross-tabulated directly, with missing values kept as an explicit
    category rather than being conflated with a negative outcome -- the specific ambiguity the
    notebook flagged about the "Unknown" and "Does not apply" codes.
    """
    return (
        drug.select(
            pl.col("role_cod").fill_null("Unknown"),
            pl.col("dechal").fill_null("Missing"),
            pl.col("rechal").fill_null("Missing"),
        )
        .group_by(["role_cod", "dechal", "rechal"])
        .agg(pl.len().alias("drug_records"))
        .sort("drug_records", descending=True)
        .with_columns(
            (pl.col("drug_records") / pl.col("drug_records").sum()).alias("joint_probability"),
            _label("role_cod", S.ROLE_LABELS, "role"),
            _label("dechal", S.CHALLENGE_LABELS, "dechallenge"),
            _label("rechal", S.CHALLENGE_LABELS, "rechallenge"),
        )
        .collect()
    )


def reaction_trend(reac: pl.LazyFrame, demo: pl.LazyFrame, terms: list[str]) -> pl.DataFrame:
    """Yearly report counts for chosen reaction terms, with a share-of-corpus column.

    The share matters: the notebook compared raw counts for Death, Pain and Dyspnoea and read a
    divergence as a reporting bias, but raw counts rise with the corpus. Normalizing by the yearly
    total separates a term becoming more common from the database becoming larger.
    """
    yearly_total = (
        demo.filter(pl.col("fda_dt").is_not_null())
        .with_columns(pl.col("fda_dt").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.len().alias("total_reports"))
    )
    upper = [t.upper() for t in terms]
    return (
        reac.filter(pl.col("pt").is_in(upper))
        .select("record_id", "pt")
        .unique()
        .join(demo.select("record_id", "fda_dt"), on="record_id", how="inner")
        .filter(pl.col("fda_dt").is_not_null())
        .with_columns(pl.col("fda_dt").dt.year().alias("year"))
        .group_by(["year", "pt"])
        .agg(pl.len().alias("reports"))
        .join(yearly_total, on="year", how="left")
        .with_columns((pl.col("reports") / pl.col("total_reports")).alias("share"))
        .sort(["pt", "year"])
        .collect()
    )


def write_all(curated: Path, out_dir: Path, trend_terms: list[str]) -> dict[str, int]:
    """Run every descriptive question and write tidy tables."""
    curated, out_dir = Path(curated), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo = pl.scan_parquet(str(curated / "demo.parquet"))
    drug = pl.scan_parquet(str(curated / "drug_ingredients.parquet"))
    reac = pl.scan_parquet(str(curated / "reac.parquet"))
    outc = pl.scan_parquet(str(curated / "outc.parquet"))
    indi = pl.scan_parquet(str(curated / "indi.parquet"))

    tables: dict[str, pl.DataFrame] = {
        "reports_per_year": reports_per_year(demo),
        "top_ingredients": top_ingredients(drug),
        "top_reactions": top_reactions(reac),
        "top_indications": top_indications(indi),
        "outcomes": outcomes(outc),
        "challenge_table": challenge_table(drug),
        "reaction_trend": reaction_trend(reac, demo, trend_terms),
    }
    tables.update(demographics(demo))
    tables.update(reporters(demo))
    tables.update(report_structure(drug, reac))

    for name, df in tables.items():
        df.write_parquet(out_dir / f"{name}.parquet", compression="zstd")
    return {name: df.height for name, df in tables.items()}
