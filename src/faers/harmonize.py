"""Stage 2: parse quarterly archives into one canonical schema spanning both eras.

Parsing quirks this handles, each confirmed against the real archives:

* Members are ``$``-delimited with no quoting, and free-text drug names legitimately contain ``"``
  and ``'``. Quote processing must be disabled or rows silently merge.
* LAERS members contain ragged rows -- more ``$`` fields than the header declares. Polars raises
  rather than truncating unless told otherwise, so a strict read fails outright on 2004Q1.
* Encoding is not UTF-8 throughout; latin-1 bytes appear in manufacturer and drug free text.
* Dates are nominally ``YYYYMMDD`` but include partial values (``YYYY``, ``YYYYMM``) and impossible
  ones -- ``event_dt`` in 2026Q1 alone ranges from year 0001 to 2028.
"""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import polars as pl

from . import schema as S
from .sources import TABLES, Era, Quarter, find_delete_member, find_member

#: Reported dates outside this window are treated as entry errors and nulled.
MIN_YEAR = 1960
MAX_YEAR = date.today().year + 1


def read_table(zf: zipfile.ZipFile, member: str) -> pl.LazyFrame:
    """Read one ``$``-delimited member as all-strings, tolerating the format's rough edges."""
    with zf.open(member) as fh:
        data = fh.read()
    return pl.read_csv(
        data,
        separator="$",
        quote_char=None,
        has_header=True,
        infer_schema_length=0,  # every column stays String; typing happens after harmonization
        truncate_ragged_lines=True,
        encoding="utf8-lossy",
        null_values=["", " "],
    ).lazy()


def parse_fda_date(col: str) -> pl.Expr:
    """Parse an FDA date column into a Date, nulling implausible and unparseable values.

    Partial dates are completed to the first of the period rather than discarded: an ``event_dt``
    of ``201505`` is real information and becomes 2015-05-01. The resulting bias toward
    month-starts is documented in the report rather than hidden.
    """
    s = pl.col(col).cast(pl.String).str.strip_chars()
    digits = s.str.replace_all(r"\D", "")
    n = digits.str.len_chars()

    year = digits.str.slice(0, 4).cast(pl.Int32, strict=False)
    month = (
        pl.when(n >= 6).then(digits.str.slice(4, 2).cast(pl.Int32, strict=False)).otherwise(1)
    )
    day = pl.when(n >= 8).then(digits.str.slice(6, 2).cast(pl.Int32, strict=False)).otherwise(1)

    # Clamp obviously-corrupt components instead of dropping the whole date.
    month = pl.when(month.is_between(1, 12)).then(month).otherwise(1)
    day = pl.when(day.is_between(1, 31)).then(day).otherwise(1)

    valid = n.is_in([4, 6, 8]) & year.is_between(MIN_YEAR, MAX_YEAR)
    return (
        pl.when(valid)
        .then(pl.date(year, month, day))
        .otherwise(None)
        .alias(col)
    )


def age_in_years() -> pl.Expr:
    """Convert (age, age_cod) to years, nulling values beyond human plausibility."""
    age = pl.col("age").cast(pl.Float64, strict=False)
    cod = pl.col("age_cod").cast(pl.String).str.to_uppercase().str.strip_chars()

    expr = pl.lit(None, pl.Float64)
    for code, mult in S.AGE_TO_YEARS.items():
        expr = pl.when(cod == code).then(age * mult).otherwise(expr)
    # A blank unit means years in practice; FDA's own extracts rely on that default.
    expr = pl.when(cod.is_null()).then(age).otherwise(expr)

    return (
        pl.when(expr.is_between(0, S.MAX_PLAUSIBLE_AGE)).then(expr).otherwise(None)
    ).alias("age_years")


#: Age bands used for stratification. Chosen to match the ICH E2B groupings the openFDA
#: ``patientagegroup`` field encodes, so the two sources stay comparable.
AGE_BANDS = [
    (0.0, 0.083, "Neonate"),
    (0.083, 2.0, "Infant"),
    (2.0, 12.0, "Child"),
    (12.0, 18.0, "Adolescent"),
    (18.0, 65.0, "Adult"),
    (65.0, S.MAX_PLAUSIBLE_AGE + 1, "Elderly"),
]


def age_band() -> pl.Expr:
    expr = pl.lit(None, pl.String)
    for lo, hi, label in reversed(AGE_BANDS):
        expr = (
            pl.when(pl.col("age_years").is_not_null() & (pl.col("age_years") >= lo)
                    & (pl.col("age_years") < hi))
            .then(pl.lit(label))
            .otherwise(expr)
        )
    return expr.fill_null("Unknown").alias("age_band")


def harmonize_sex() -> pl.Expr:
    s = pl.col("sex").cast(pl.String).str.to_uppercase().str.strip_chars()
    expr = pl.lit("UNK")
    for src, dst in S.SEX_MAP.items():
        if src:
            expr = pl.when(s == src).then(pl.lit(dst)).otherwise(expr)
    return expr.alias("sex")


def _harmonize_demo(lf: pl.LazyFrame, q: Quarter) -> pl.LazyFrame:
    """DEMO needs the most work: it defines the case key and the version used for dedup."""
    if q.era is Era.FAERS:
        version = pl.col("case_version").cast(pl.Int32, strict=False).fill_null(1)
    else:
        # LAERS has no caseversion. FOLL_SEQ counts follow-ups and is null on the initial report,
        # so shifting by one puts both eras on a common "version 1 is the initial report" scale.
        version = pl.col("foll_seq").cast(pl.Int32, strict=False).fill_null(0) + 1

    names = set(lf.collect_schema().names())
    date_cols = [c for c in ("event_dt", "mfr_dt", "init_fda_dt", "fda_dt", "rept_dt")
                 if c in names]

    return (
        lf.with_columns(
            [
                pl.col("record_id").cast(pl.String).str.strip_chars(),
                # Case keys are zero-padded numerics of differing width across eras; strip leading
                # zeros so a LAERS CASE and the FAERS caseid it became compare equal.
                pl.col("case_id").cast(pl.String).str.strip_chars().str.replace(r"^0+", "")
                .alias("case_id"),
                version.alias("case_version"),
                *[parse_fda_date(c) for c in date_cols],
                harmonize_sex(),
                age_in_years(),
            ]
        )
        .with_columns(age_band())
        .with_columns(
            [
                pl.lit(q.label).alias("quarter"),
                pl.lit(q.era.value).alias("era"),
                pl.lit(q.year).cast(pl.Int32).alias("file_year"),
                pl.col("wt").cast(pl.Float64, strict=False).alias("wt"),
            ]
        )
        .drop([c for c in ("foll_seq",) if c in names])
    )


def _harmonize_child(lf: pl.LazyFrame, table: str, q: Quarter) -> pl.LazyFrame:
    lf = lf.with_columns(pl.col("record_id").cast(pl.String).str.strip_chars())
    if "drug_seq" in lf.collect_schema().names():
        lf = lf.with_columns(pl.col("drug_seq").cast(pl.Int32, strict=False))
    if table == "THER":
        lf = lf.with_columns([parse_fda_date("start_dt"), parse_fda_date("end_dt")])
    if table == "DRUG":
        lf = lf.with_columns(
            pl.col("role_cod").cast(pl.String).str.to_uppercase().str.strip_chars()
        )
    if table == "REAC":
        # MedDRA preferred terms are stored with inconsistent capitalization across quarters.
        lf = lf.with_columns(pl.col("pt").cast(pl.String).str.strip_chars().str.to_uppercase())
    return lf.with_columns(pl.lit(q.label).alias("quarter"))


def harmonize_quarter(zip_path: Path, q: Quarter, out_dir: Path) -> dict:
    """Parse every table in one archive and write canonical parquet.

    Returns per-table row counts.
    """
    out_dir = Path(out_dir)
    stats: dict[str, int] = {}

    with zipfile.ZipFile(zip_path) as zf:
        for table in TABLES:
            member = find_member(zf, table, q)
            if member is None:
                stats[table] = 0
                continue

            lf = read_table(zf, member)
            lf = S.apply_map(lf, table, q.era)
            S.require(lf, table, q.label)
            lf = (
                _harmonize_demo(lf, q)
                if table == "DEMO"
                else _harmonize_child(lf, table, q)
            )

            df = lf.collect()
            dest = out_dir / table.lower()
            dest.mkdir(parents=True, exist_ok=True)
            df.write_parquet(dest / f"{q.label}.parquet", compression="zstd")
            stats[table] = df.height

        # FDA case retractions, FAERS era only.
        member = find_delete_member(zf, q)
        if member is not None:
            with zf.open(member) as fh:
                raw = fh.read().decode("utf-8", errors="replace")
            ids = [
                line.strip().lstrip("0")
                for line in raw.splitlines()
                if line.strip() and not line.strip().lower().startswith("caseid")
            ]
            dest = out_dir / "deleted"
            dest.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"case_id": ids, "quarter": [q.label] * len(ids)}).write_parquet(
                dest / f"{q.label}.parquet", compression="zstd"
            )
            stats["DELETED"] = len(ids)
        else:
            stats["DELETED"] = 0

    return stats
