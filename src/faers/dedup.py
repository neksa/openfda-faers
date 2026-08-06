"""Stage 3: reduce the corpus to one record per case.

This is the correction the 2020 notebook needed but could not make. It worked from the openFDA API,
which exposes reports without the case-version fields, so its drug-ADR matrix counted the same
patient case once per follow-up submission. The notebook noticed the consequence -- a block of
"quite repetitive" rows in its clustergram that it guessed were "duplicates in the database or
other artifacts" -- but had no way to remove them.

Three distinct reductions happen here, and each is counted separately so the report can state what
the corpus actually contains:

1. **Retracted cases.** FAERS quarters ship a ``Deleted/`` list of case ids FDA has withdrawn.
   These are removed outright.
2. **Superseded versions.** A case is resubmitted as follow-up information arrives; each
   submission lands in a different quarter. Only the latest surviving version is kept.
3. **Cross-era continuation.** A case opened under LAERS can be followed up under FAERS. Because
   case identifiers were carried across the 2012Q4 transition, these link on the normalized case
   id, and the FAERS record supersedes the LAERS one.

Note what is *not* attempted: FAERS also contains genuine duplicate cases submitted independently
by different reporters under different case ids. Detecting those needs probabilistic matching on
demographics, dates and drug lists. It is out of scope, and the report says so rather than
implying the corpus is duplicate-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .sources import Era

#: FAERS records supersede LAERS ones for the same case, regardless of version numbering, because
#: the two eras count versions on different scales (caseversion vs a derived FOLL_SEQ).
ERA_RANK = {Era.LAERS.value: 0, Era.FAERS.value: 1}


def build_case_index(harmonized_dir: Path) -> pl.LazyFrame:
    """One row per surviving case, carrying the record_id that represents it."""
    harmonized_dir = Path(harmonized_dir)
    demo = pl.scan_parquet(str(harmonized_dir / "demo" / "*.parquet"))

    quarter_idx = (
        pl.col("quarter").str.slice(0, 4).cast(pl.Int32) * 4
        + pl.col("quarter").str.slice(5, 1).cast(pl.Int32)
        - 1
    ).alias("quarter_idx")

    era_rank = pl.col("era").replace_strict(ERA_RANK, default=0).alias("era_rank")

    return demo.with_columns([quarter_idx, era_rank])


def deleted_case_ids(harmonized_dir: Path) -> pl.LazyFrame:
    """Union of every quarter's retraction list."""
    pattern = Path(harmonized_dir) / "deleted" / "*.parquet"
    files = sorted(Path(harmonized_dir).glob("deleted/*.parquet"))
    if not files:
        return pl.LazyFrame({"case_id": pl.Series([], dtype=pl.String)})
    return pl.scan_parquet(str(pattern)).select("case_id").unique()


def dedup(harmonized_dir: Path, out_dir: Path, stats_path: Path) -> dict:
    """Write the deduplicated DEMO table and the surviving record_id whitelist."""
    harmonized_dir, out_dir = Path(harmonized_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo = build_case_index(harmonized_dir)
    n_raw = demo.select(pl.len()).collect().item()
    n_cases_raw = demo.select(pl.col("case_id").n_unique()).collect().item()

    # 1. drop FDA-retracted cases
    deleted = deleted_case_ids(harmonized_dir)
    n_deleted_listed = deleted.select(pl.len()).collect().item()
    demo = demo.join(deleted.with_columns(pl.lit(True).alias("_del")), on="case_id", how="left")
    n_after_delete = demo.filter(pl.col("_del").is_null()).select(pl.len()).collect().item()
    n_deleted_removed = n_raw - n_after_delete
    demo = demo.filter(pl.col("_del").is_null()).drop("_del")

    # 2+3. keep the latest surviving version of each case. Ordering: FAERS beats LAERS, then the
    # higher case version, then the later quarter, then the later FDA receipt date. record_id is
    # the final deterministic tie-break so the result does not depend on file ordering.
    latest = (
        demo.sort(
            ["case_id", "era_rank", "case_version", "quarter_idx", "fda_dt", "record_id"],
            nulls_last=False,
        )
        .group_by("case_id", maintain_order=False)
        .last()
    )

    result = latest.collect()
    n_final = result.height

    # Cross-era continuations: cases seen in both eras, resolved to the FAERS record.
    era_span = (
        demo.group_by("case_id")
        .agg(pl.col("era").n_unique().alias("n_eras"))
        .filter(pl.col("n_eras") > 1)
        .select(pl.len())
        .collect()
        .item()
    )

    result.write_parquet(out_dir / "demo.parquet", compression="zstd")
    result.select("record_id", "case_id", "quarter", "era").write_parquet(
        out_dir / "surviving_records.parquet", compression="zstd"
    )

    stats = {
        "records_raw": n_raw,
        "cases_raw": n_cases_raw,
        "retracted_ids_listed": n_deleted_listed,
        "records_removed_as_retracted": n_deleted_removed,
        "records_removed_as_superseded": n_after_delete - n_final,
        "cases_spanning_both_eras": era_span,
        "records_final": n_final,
        "duplicate_fraction_of_raw": round(1 - n_final / n_raw, 4) if n_raw else 0.0,
    }
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def filter_children(harmonized_dir: Path, dedup_dir: Path, out_dir: Path) -> dict[str, int]:
    """Restrict the child tables to records that survived deduplication."""
    harmonized_dir, dedup_dir, out_dir = Path(harmonized_dir), Path(dedup_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = pl.scan_parquet(str(dedup_dir / "surviving_records.parquet")).select("record_id")
    counts: dict[str, int] = {}
    for table in ("drug", "reac", "outc", "indi", "ther", "rpsr"):
        src = sorted(harmonized_dir.glob(f"{table}/*.parquet"))
        if not src:
            continue
        lf = pl.scan_parquet(str(harmonized_dir / table / "*.parquet"))
        out = lf.join(keep, on="record_id", how="semi")
        df = out.collect()
        df.write_parquet(out_dir / f"{table}.parquet", compression="zstd")
        counts[table] = df.height
    return counts
