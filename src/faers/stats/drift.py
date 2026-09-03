"""Stage: detect MedDRA vocabulary drift from the corpus, without a dictionary.

MedDRA is revised twice a year, in March and September. Terms are introduced, renamed, split and
merged, so a preferred term's frequency can change sharply for reasons that have nothing to do with
what happened to patients. Over a 22-year window this is not a footnote: a term that did not exist
before 2015 will show a "rising trend" that is purely an artefact of the vocabulary.

The report has always declared this limitation. This measures it instead.

Correcting drift properly needs the MedDRA release history, which is licensed. What does *not*
need a licence is noticing it: a term that goes from absent to common between two adjacent
quarters, or from common to absent, is behaving like a vocabulary change rather than an
epidemiological one. Real clinical trends do not switch on in a single quarter and stay on.

The output is a per-term flag, not a correction. Terms flagged here should be excluded from or
annotated in any temporal claim; they remain perfectly usable for the time-invariant
disproportionality analysis, which pools across the whole window.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

#: A term must appear in at least this many reports overall before its trajectory is worth
#: testing. Rare terms are noisy by construction and would dominate the flag list.
MIN_TOTAL_REPORTS = 200

#: Share-of-corpus below which a term counts as effectively absent in a quarter.
ABSENT_SHARE = 1e-6

#: A step this large between adjacent quarters, measured as a ratio of shares, is not plausible
#: as an epidemiological change in a corpus of this size.
STEP_RATIO = 10.0


def quarter_index(col: str = "quarter") -> pl.Expr:
    """Monotonic quarter counter from a ``2026Q1`` label."""
    return (
        pl.col(col).str.slice(0, 4).cast(pl.Int32) * 4
        + pl.col(col).str.slice(5, 1).cast(pl.Int32)
        - 1
    )


def term_trajectories(reac: pl.LazyFrame, demo: pl.LazyFrame) -> pl.LazyFrame:
    """Share of each quarter's reports mentioning each preferred term.

    Indexed on the quarter of the FDA receipt date rather than the quarter of the file a record was
    found in: after deduplication a case lives in the file of its *latest* follow-up, which would
    shift older cases forward and manufacture drift that is not there.
    """
    dated = demo.filter(pl.col("fda_dt").is_not_null()).select(
        "record_id",
        (pl.col("fda_dt").dt.year() * 4 + pl.col("fda_dt").dt.quarter() - 1).alias("qidx"),
    )
    per_quarter = dated.group_by("qidx").agg(pl.len().alias("quarter_total"))

    return (
        reac.filter(pl.col("pt").is_not_null())
        .select("record_id", "pt")
        .unique()
        .join(dated, on="record_id", how="inner")
        .group_by(["pt", "qidx"])
        .agg(pl.len().alias("reports"))
        .join(per_quarter, on="qidx", how="left")
        .with_columns((pl.col("reports") / pl.col("quarter_total")).alias("share"))
    )


def detect(reac: pl.LazyFrame, demo: pl.LazyFrame) -> pl.DataFrame:
    """Flag terms whose trajectory looks like a vocabulary change.

    Three signatures, any of which is enough:

    ``late_onset``   effectively absent for the first part of the window, then established
    ``discontinued`` established early, then effectively absent
    ``step_change``  an implausibly large jump between adjacent quarters

    A term flagged by none of these is not necessarily drift-free -- a rename between two similar
    terms produces a fall and a rise that this will catch on both sides, but a gradual merge may
    not be visible at all. The flag is a floor on the problem, not a ceiling, and the report says
    so.
    """
    traj = term_trajectories(reac, demo).collect()
    if traj.is_empty():
        return pl.DataFrame()

    totals = traj.group_by("pt").agg(pl.col("reports").sum().alias("total_reports"))
    eligible = totals.filter(pl.col("total_reports") >= MIN_TOTAL_REPORTS)
    traj = traj.join(eligible.select("pt"), on="pt", how="semi")
    if traj.is_empty():
        return pl.DataFrame()

    # The midpoint must come from the *corpus* window, not from the quarters in which the surviving
    # terms happen to appear. Deriving it from the trajectory table lets a term define its own
    # window: a term present only in the second half would have its midpoint fall in the middle of
    # that half, and its onset would be invisible.
    window = (
        demo.filter(pl.col("fda_dt").is_not_null())
        .select(
            (pl.col("fda_dt").dt.year() * 4 + pl.col("fda_dt").dt.quarter() - 1).alias("qidx")
        )
        .select(pl.col("qidx").min().alias("lo"), pl.col("qidx").max().alias("hi"))
        .collect()
        .row(0)
    )
    midpoint = (window[0] + window[1]) / 2

    per_term = traj.sort(["pt", "qidx"]).group_by("pt").agg(
        pl.col("qidx").min().alias("first_seen"),
        pl.col("qidx").max().alias("last_seen"),
        pl.col("share").filter(pl.col("qidx") <= midpoint).mean().alias("share_early"),
        pl.col("share").filter(pl.col("qidx") > midpoint).mean().alias("share_late"),
        pl.col("share").alias("shares"),
        pl.col("qidx").alias("qidxs"),
        pl.col("reports").sum().alias("total_reports"),
    )

    # Largest adjacent-quarter ratio, guarded against divide-by-zero on an absent quarter.
    per_term = per_term.with_columns(
        pl.struct(["shares", "qidxs"])
        .map_elements(_max_step_ratio, return_dtype=pl.Float64)
        .alias("max_step_ratio")
    )

    early = pl.col("share_early").fill_null(0.0)
    late = pl.col("share_late").fill_null(0.0)

    return (
        per_term.with_columns(
            ((early < ABSENT_SHARE) & (late >= ABSENT_SHARE)).alias("late_onset"),
            ((early >= ABSENT_SHARE) & (late < ABSENT_SHARE)).alias("discontinued"),
            (pl.col("max_step_ratio") >= STEP_RATIO).alias("step_change"),
        )
        .with_columns(
            (pl.col("late_onset") | pl.col("discontinued") | pl.col("step_change")).alias(
                "drift_suspect"
            )
        )
        .drop(["shares", "qidxs"])
        .sort("total_reports", descending=True)
    )


def _max_step_ratio(row: dict) -> float:
    """Largest jump between consecutive observed quarters, as a ratio of shares.

    Only consecutive *quarters* count. A gap in the series means the term was absent, which the
    onset/discontinued signatures already cover; treating a gap as a step would double-count it.
    """
    pairs = sorted(zip(row["qidxs"], row["shares"], strict=True))
    worst = 1.0
    for (q0, s0), (q1, s1) in zip(pairs, pairs[1:], strict=False):
        if q1 - q0 != 1:
            continue
        lo, hi = sorted((s0, s1))
        if lo <= 0:
            continue
        worst = max(worst, hi / lo)
    return worst


def write(curated: Path, out_dir: Path, summary_path: Path) -> dict:
    """Run detection and write the per-term table plus a summary."""
    curated, out_dir = Path(curated), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reac = pl.scan_parquet(str(curated / "reac.parquet"))
    demo = pl.scan_parquet(str(curated / "demo.parquet"))

    flags = detect(reac, demo)
    flags.write_parquet(out_dir / "term_drift.parquet", compression="zstd")

    tested = flags.height
    suspect = int(flags["drift_suspect"].sum()) if tested else 0
    reports_affected = (
        int(flags.filter(pl.col("drift_suspect"))["total_reports"].sum()) if tested else 0
    )
    reports_total = int(flags["total_reports"].sum()) if tested else 0

    summary = {
        "terms_tested": tested,
        "min_total_reports": MIN_TOTAL_REPORTS,
        "terms_drift_suspect": suspect,
        "terms_drift_suspect_fraction": round(suspect / max(tested, 1), 4),
        "late_onset": int(flags["late_onset"].sum()) if tested else 0,
        "discontinued": int(flags["discontinued"].sum()) if tested else 0,
        "step_change": int(flags["step_change"].sum()) if tested else 0,
        "report_mentions_affected_fraction": round(
            reports_affected / max(reports_total, 1), 4
        ),
    }
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
    return summary
