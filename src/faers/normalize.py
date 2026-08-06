"""Stage 4: resolve reported drug names to active ingredients.

The unit of analysis for signal detection has to be the ingredient, not the reported product
string. FAERS contains ~33k distinct ``drugname`` values in a *single* quarter -- brand names,
misspellings, strengths, dose forms and manufacturer codes ("ONE A DAY /07499601/") -- which would
scatter the reports for one substance across dozens of rows and destroy every marginal count.

The resolution cascade is deliberately license-free. RxNorm-to-ATC and the MedDRA hierarchy both
require licences that cannot ship in a public repository, so instead this **bootstraps FDA's own
harmonization out of the corpus**: from 2014Q3 onward each DRUG row carries both the reported
``drugname`` and FDA's curated ``prod_ai`` active ingredient. Those co-occurrences form an
empirical drugname -> ingredient dictionary, which is then applied backwards to the LAERS era and
to modern rows where ``prod_ai`` is missing.

Cascade, in order of decreasing confidence:

1. ``prod_ai`` present -> use it directly (split on ``\\`` for combination products).
2. Exact match of the raw name in the bootstrapped dictionary.
3. Exact match of the *cleaned* name (strengths, forms and codes stripped) in the dictionary.
4. Unresolved -> keep the cleaned name, flagged, so coverage is measurable rather than hidden.

Combination products matter: 5.6% of ``prod_ai`` values in 2026Q1 hold several ingredients
separated by backslashes. Splitting them into one row per ingredient is required for a drug's
marginal count to include the combination products that contain it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

#: Dose forms, routes and packaging words that carry no ingredient information.
_NOISE_WORDS = (
    r"TABLETS?|CAPSULES?|CAPLETS?|SOFTGELS?|LOZENGES?|SUPPOSITOR(?:Y|IES)|"
    r"INJECTIONS?|INFUSIONS?|SOLUTIONS?|SUSPENSIONS?|EMULSIONS?|SYRUPS?|ELIXIRS?|"
    r"CREAMS?|OINTMENTS?|GELS?|LOTIONS?|PATCHES?|SPRAYS?|DROPS?|POWDERS?|GRANULES?|"
    r"ORAL|TOPICAL|INTRAVENOUS|INTRAMUSCULAR|SUBCUTANEOUS|OPHTHALMIC|NASAL|RECTAL|INHALATION|"
    r"EXTENDED[- ]RELEASE|DELAYED[- ]RELEASE|SUSTAINED[- ]RELEASE|"
    r"\bER\b|\bXR\b|\bSR\b|\bCR\b|\bDR\b|\bIR\b|\bODT\b|\bHCL\b|\bCHW\b|\bUSP\b|\bNF\b"
)

#: A strength expression such as "81MG", "10 MG/ML", "0.5%".
_STRENGTH = r"\d+(?:[.,]\d+)?\s*(?:MG|MCG|UG|G|KG|ML|L|IU|U|MEQ|MMOL|%)\b(?:\s*/\s*\w+)?"

#: A manufacturer / NDC-ish code embedded in the free text, e.g. "/07499601/".
_EMBEDDED_CODE = r"/\s*\d[\d\-]*\s*/"

_CLEAN_STEPS: tuple[tuple[str, str], ...] = (
    (_EMBEDDED_CODE, " "),
    (r"\(.*?\)", " "),  # parenthetical asides: "(as hydrochloride)"
    (_STRENGTH, " "),
    (_NOISE_WORDS, " "),
    (r"[^A-Z0-9 /\\;,+.\-]", " "),
    (r"\s*[.\-]+\s*$", " "),
    (r"\s+", " "),
)

#: Separators that indicate several ingredients in one reported string.
_SPLIT_RE = re.compile(r"[\\;+]|,\s(?=[A-Z])")

#: Strings that are not drugs at all. FAERS uses several placeholder conventions.
_NON_DRUG = {
    "", "UNKNOWN", "UNK", "UNSPECIFIED", "N/A", "NA", "NONE", "NO DRUG",
    "UNSPECIFIED INGREDIENT", "UNKNOWN DRUG", "PRODUCT", "BLINDED", "PLACEBO",
    "UNSPECIFIED MEDICATION", "MEDICATION", "DRUG",
}

#: Free-text placeholders that carry a count or qualifier and so cannot be matched exactly --
#: "12 unspecified medications", "UNKNOWN MEDICATION", "BLINDED THERAPY". Anchored to whole
#: strings so a real ingredient that merely contains one of these words is never dropped.
_NON_DRUG_RE = (
    r"^(?:\d+\s+)?(?:UNSPECIFIED|UNKNOWN|BLINDED|PLACEBO|NON|NO)\b.*"
    r"(?:DRUGS?|MEDICATIONS?|PRODUCTS?|INGREDIENTS?|THERAP(?:Y|IES)|AGENTS?|SUBSTANCES?)$"
    r"|^(?:CONCOMITANT|SUSPECT)\s+(?:DRUGS?|MEDICATIONS?)$"
)


def clean_expr(col: str) -> pl.Expr:
    """Normalize a reported drug string: uppercase, strip strengths, forms and codes."""
    e = pl.col(col).cast(pl.String).str.to_uppercase().str.strip_chars()
    for pattern, repl in _CLEAN_STEPS:
        e = e.str.replace_all(pattern, repl)
    return e.str.strip_chars().str.strip_chars(" .-")


def explode_ingredients(lf: pl.LazyFrame, col: str, out: str = "ingredient") -> pl.LazyFrame:
    """Explode a column of possibly-combination strings into one row per ingredient."""
    return (
        lf.with_columns(
            pl.col(col)
            .cast(pl.String)
            .str.replace_all(r"[;+]", "\\")
            .str.split("\\")
            .alias(out)
        )
        .explode(out)
        .with_columns(pl.col(out).str.strip_chars())
        .filter(pl.col(out).is_not_null() & (pl.col(out).str.len_chars() > 1))
    )


def build_dictionary(drug: pl.LazyFrame, min_support: int = 2) -> pl.DataFrame:
    """Learn a drugname -> ingredient dictionary from rows where FDA supplied both.

    Where one reported name maps to several ingredient strings across the corpus, the most
    frequently observed mapping wins, and only mappings seen at least ``min_support`` times are
    kept so a single mis-keyed row cannot define a mapping.
    """
    both = drug.filter(
        pl.col("prod_ai").is_not_null() & pl.col("drugname").is_not_null()
    ).select(
        clean_expr("drugname").alias("name_clean"),
        pl.col("drugname").cast(pl.String).str.to_uppercase().str.strip_chars().alias("name_raw"),
        pl.col("prod_ai").cast(pl.String).str.to_uppercase().str.strip_chars().alias("prod_ai"),
    )

    # The trailing `prod_ai` sort key is load-bearing, not cosmetic. Where one reported name maps
    # to two ingredient strings with *equal* support, sorting on support alone leaves the winner to
    # whichever row the multithreaded group-by happens to see first, and the pipeline stops being
    # reproducible: two runs over identical inputs differed by ~45 rows and ~800 signals before
    # this key was added. Sorting to a total order makes the choice deterministic.
    counted = (
        both.group_by(["name_raw", "name_clean", "prod_ai"])
        .agg(pl.len().alias("support"))
        .filter(pl.col("support") >= min_support)
        .sort(["name_raw", "support", "prod_ai"], descending=[False, True, False])
        .group_by("name_raw", maintain_order=True)
        .first()
    )
    return counted.collect()


def resolve(drug: pl.LazyFrame, dictionary: pl.DataFrame) -> pl.LazyFrame:
    """Attach a resolved ingredient string and the method that resolved it."""
    by_raw = dictionary.select(
        pl.col("name_raw"), pl.col("prod_ai").alias("dict_by_raw")
    ).unique(subset=["name_raw"])
    by_clean = (
        dictionary.select(pl.col("name_clean"), pl.col("prod_ai").alias("dict_by_clean"),
                          pl.col("support"))
        # Same determinism requirement as build_dictionary: ties on support must not be resolved
        # by row arrival order.
        .sort(["support", "dict_by_clean"], descending=[True, False])
        .unique(subset=["name_clean"], keep="first", maintain_order=True)
        .drop("support")
    )

    return (
        drug.with_columns(
            [
                pl.col("drugname").cast(pl.String).str.to_uppercase().str.strip_chars()
                .alias("name_raw"),
                clean_expr("drugname").alias("name_clean"),
            ]
        )
        .join(by_raw.lazy(), on="name_raw", how="left")
        .join(by_clean.lazy(), on="name_clean", how="left")
        .with_columns(
            pl.coalesce(
                pl.col("prod_ai").cast(pl.String).str.to_uppercase().str.strip_chars(),
                pl.col("dict_by_raw"),
                pl.col("dict_by_clean"),
                pl.col("name_clean"),
            ).alias("ingredient_raw"),
            pl.when(pl.col("prod_ai").is_not_null())
            .then(pl.lit("prod_ai"))
            .when(pl.col("dict_by_raw").is_not_null())
            .then(pl.lit("dict_exact"))
            .when(pl.col("dict_by_clean").is_not_null())
            .then(pl.lit("dict_cleaned"))
            .otherwise(pl.lit("unresolved"))
            .alias("resolution"),
        )
        .drop(["dict_by_raw", "dict_by_clean"])
    )


def normalize_drugs(
    dedup_dir: Path, out_path: Path, stats_path: Path, min_support: int = 2
) -> dict:
    """Run the full cascade and write the ingredient-level drug table."""
    dedup_dir = Path(dedup_dir)
    drug = pl.scan_parquet(str(dedup_dir / "drug.parquet"))

    dictionary = build_dictionary(drug, min_support=min_support)
    resolved = resolve(drug, dictionary)

    exploded = (
        explode_ingredients(resolved, "ingredient_raw", "ingredient")
        .with_columns(pl.col("ingredient").str.strip_chars().str.strip_chars(" .-"))
        .filter(
            ~pl.col("ingredient").is_in(list(_NON_DRUG))
            & ~pl.col("ingredient").str.contains(_NON_DRUG_RE)
        )
    )

    df = exploded.select(
        "record_id", "drug_seq", "role_cod", "drugname", "ingredient", "resolution",
        "dechal", "rechal", "route", "quarter",
    ).collect()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")

    by_method = (
        df.group_by("resolution").agg(pl.len().alias("rows")).sort("rows", descending=True)
    )
    stats = {
        "dictionary_entries": dictionary.height,
        "rows_out": df.height,
        "distinct_ingredients": df["ingredient"].n_unique(),
        "distinct_reported_names": df["drugname"].n_unique(),
        "resolution_counts": {r["resolution"]: r["rows"] for r in by_method.to_dicts()},
        "resolved_fraction": round(
            1 - df.filter(pl.col("resolution") == "unresolved").height / max(df.height, 1), 4
        ),
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats, indent=2) + "\n")
    return stats
