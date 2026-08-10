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

#: A strength expression such as "81MG", "10 MG/ML", "0.5%". The percent alternative carries no
#: trailing word boundary: "%" is a non-word character, so "\b" after it would never match at end
#: of string and "SODIUM CHLORIDE 0.9%" would keep a stray "0.9".
_STRENGTH = (
    r"\d+(?:[.,]\d+)?\s*(?:(?:MG|MCG|UG|G|KG|ML|L|IU|U|MEQ|MMOL)\b(?:\s*/\s*\w+)?|%)"
)

#: A manufacturer / NDC-ish code embedded in the free text, e.g. "/07499601/".
_EMBEDDED_CODE = r"/\s*\d[\d\-]*\s*/"

#: Delivery devices and presentations appended to a brand: "HUMIRA . PEN", "ADVAIR DISKUS",
#: "VENTOLIN HFA". These identify the package, not the substance, and splinter one product across
#: several apparent ingredients if left in place.
_DEVICE_WORDS = (
    r"\bPENS?\b|\bDISKUS\b|\bRESPIMAT\b|\bHFA\b|\bAUTO[- ]?INJECTORS?\b|\bSYRINGES?\b|"
    r"\bINHALERS?\b|\bVIALS?\b|\bKITS?\b|\bPACKS?\b|\bFLEXPEN\b|\bSOLOSTAR\b|\bNEBULES?\b|"
    r"\bAMPOULES?\b|\bAMPULES?\b|\bPREFILLED\b|\bSINGLE[- ]DOSE\b|\bMULTI[- ]?DOSE\b"
)

#: A trailing product/strength designator: "DURAGESIC-100", "PARAGARD 380A", "ADVAIR 100/50".
#:
#: Requires three or more digits, or digits immediately followed by a letter, because a short
#: trailing number is frequently part of the substance name itself -- NONOXYNOL-9, OMEGA-3, COQ-10
#: are ingredients, not product codes. Stripping those would invent new substances.
_TRAILING_DESIGNATOR = r"[\s\-]+(?:\d{3,}[A-Z]?|\d+[A-Z])(?:\s*/\s*\d+[A-Z]?)*\s*$"

_CLEAN_STEPS: tuple[tuple[str, str], ...] = (
    (_EMBEDDED_CODE, " "),
    (r"\(.*?\)", " "),  # parenthetical asides: "(as hydrochloride)"
    (_STRENGTH, " "),
    (_NOISE_WORDS, " "),
    (_DEVICE_WORDS, " "),
    (r"[^A-Z0-9 /\\;,+.\-]", " "),
    (r"\s*[.\-]+\s*$", " "),
    (r"\s+", " "),
    (_TRAILING_DESIGNATOR, ""),
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


def resolve(
    drug: pl.LazyFrame,
    dictionary: pl.DataFrame,
    ndc_index: pl.DataFrame | None = None,
    rxnav_map: dict[str, str] | None = None,
) -> pl.LazyFrame:
    """Attach a resolved ingredient string and the method that resolved it.

    Cascade, in decreasing order of confidence. ``resolution`` records which step fired, and is
    carried all the way into the published signal table so a consumer can tell a curated ingredient
    from a free-text fallback:

    ``prod_ai``       FDA's own active-ingredient field
    ``dict_exact``    reported name seen verbatim alongside a prod_ai elsewhere in the corpus
    ``dict_cleaned``  same, after stripping strengths, forms and devices
    ``ndc_brand``     first token matched an unambiguous brand in FDA's NDC directory
    ``rxnav_brand``   first token resolved to ingredients via NLM RxNav
    ``unresolved``    none of the above; the cleaned name is kept, and labelled
    """
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

    # Brand lookups key on the first token of the cleaned name: NDC brand strings are usually
    # longer than what the reporter wrote ("ParaGard T 380A" vs "PARAGARD 380A"), so neither exact
    # direction matches.
    ndc = (
        ndc_index.filter(pl.col("usable"))
        .select(pl.col("token"), pl.col("ingredients").alias("ndc_ing"))
        if ndc_index is not None and ndc_index.height
        else pl.DataFrame({"token": [], "ndc_ing": []}, schema={"token": pl.String,
                                                                "ndc_ing": pl.String})
    )
    rx = pl.DataFrame(
        {"token": list((rxnav_map or {}).keys()), "rx_ing": list((rxnav_map or {}).values())},
        schema={"token": pl.String, "rx_ing": pl.String},
    )

    return (
        drug.with_columns(
            [
                pl.col("drugname").cast(pl.String).str.to_uppercase().str.strip_chars()
                .alias("name_raw"),
                clean_expr("drugname").alias("name_clean"),
            ]
        )
        .with_columns(
            pl.col("name_clean").str.replace_all(r"[^A-Z0-9 ]", " ").str.strip_chars()
            .str.split(" ").list.first().alias("token")
        )
        .join(by_raw.lazy(), on="name_raw", how="left")
        .join(by_clean.lazy(), on="name_clean", how="left")
        .join(ndc.lazy(), on="token", how="left")
        .join(rx.lazy(), on="token", how="left")
        .with_columns(
            pl.coalesce(
                pl.col("prod_ai").cast(pl.String).str.to_uppercase().str.strip_chars(),
                pl.col("dict_by_raw"),
                pl.col("dict_by_clean"),
                pl.col("ndc_ing"),
                pl.col("rx_ing"),
                pl.col("name_clean"),
            ).alias("ingredient_raw"),
            pl.when(pl.col("prod_ai").is_not_null())
            .then(pl.lit("prod_ai"))
            .when(pl.col("dict_by_raw").is_not_null())
            .then(pl.lit("dict_exact"))
            .when(pl.col("dict_by_clean").is_not_null())
            .then(pl.lit("dict_cleaned"))
            .when(pl.col("ndc_ing").is_not_null())
            .then(pl.lit("ndc_brand"))
            .when(pl.col("rx_ing").is_not_null())
            .then(pl.lit("rxnav_brand"))
            .otherwise(pl.lit("unresolved"))
            .alias("resolution"),
        )
        .drop(["dict_by_raw", "dict_by_clean", "ndc_ing", "rx_ing", "token"])
    )


def unresolved_tokens(
    drug: pl.LazyFrame, dictionary: pl.DataFrame, ndc_index: pl.DataFrame, min_reports: int
) -> list[str]:
    """First tokens still unresolved after the corpus and NDC steps, worth an RxNav lookup.

    Restricted to tokens carrying at least ``min_reports`` drug records. RxNav is queried one name
    at a time over the network, so resolving the whole tail would take hours to recover substances
    that cannot reach the minimum-support threshold anyway.
    """
    partial = resolve(drug, dictionary, ndc_index, None)
    return (
        partial.filter(pl.col("resolution") == "unresolved")
        .with_columns(
            pl.col("name_clean").str.replace_all(r"[^A-Z0-9 ]", " ").str.strip_chars()
            .str.split(" ").list.first().alias("token")
        )
        .filter(pl.col("token").is_not_null() & (pl.col("token").str.len_chars() >= 3))
        .group_by("token")
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") >= min_reports)
        .sort("rows", descending=True)
        .collect()["token"]
        .to_list()
    )


def normalize_drugs(
    dedup_dir: Path,
    out_path: Path,
    stats_path: Path,
    min_support: int = 2,
    ndc_index_path: Path | None = None,
    rxnav_cache: Path | None = None,
    rxnav_min_reports: int = 50,
) -> dict:
    """Run the full cascade and write the ingredient-level drug table."""
    dedup_dir = Path(dedup_dir)
    drug = pl.scan_parquet(str(dedup_dir / "drug.parquet"))

    dictionary = build_dictionary(drug, min_support=min_support)

    ndc_index = (
        pl.read_parquet(ndc_index_path)
        if ndc_index_path and Path(ndc_index_path).exists()
        else None
    )

    rxnav_map: dict[str, str] = {}
    if rxnav_cache is not None and ndc_index is not None:
        from .brands import resolve_via_rxnav

        tokens = unresolved_tokens(drug, dictionary, ndc_index, rxnav_min_reports)
        print(f"  querying RxNav for {len(tokens):,} unresolved tokens", flush=True)
        rxnav_map = resolve_via_rxnav(tokens, rxnav_cache)

    resolved = resolve(drug, dictionary, ndc_index, rxnav_map)

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
    # An ingredient string counts as curated if *any* row reached it by a confident path. This is
    # the number the report quotes, and it is the honest one: the same string is often reached both
    # ways, so counting unresolved rows alone overstates the problem.
    ever_confident = (
        df.group_by("ingredient")
        .agg((pl.col("resolution") != "unresolved").any().alias("confident"))
    )
    n_ing = ever_confident.height
    n_fallback = ever_confident.filter(~pl.col("confident")).height

    stats = {
        "dictionary_entries": dictionary.height,
        "ndc_tokens_usable": (
            int(ndc_index.filter(pl.col("usable")).height) if ndc_index is not None else 0
        ),
        "rxnav_tokens_resolved": len(rxnav_map),
        "rows_out": df.height,
        "distinct_ingredients": n_ing,
        "distinct_reported_names": df["drugname"].n_unique(),
        "resolution_counts": {r["resolution"]: r["rows"] for r in by_method.to_dicts()},
        "resolved_fraction": round(
            1 - df.filter(pl.col("resolution") == "unresolved").height / max(df.height, 1), 4
        ),
        "ingredients_fallback_only": n_fallback,
        "ingredients_fallback_only_fraction": round(n_fallback / max(n_ing, 1), 4),
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats, indent=2) + "\n")
    return stats
