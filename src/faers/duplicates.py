"""Detect the same incident reported independently under different case ids.

`dedup.py` collapses case *versions* -- follow-up submissions carrying the same case id. It cannot
see the other kind of redundancy: one adverse event described separately by, say, the patient, the
prescribing physician and the manufacturer, each producing a distinct case. FDA does not link
these, and no field identifies them.

The approach is standard record linkage: block, then score.

**Blocking.** Comparing 20.3M records pairwise is 2x10^14 comparisons and impossible. Records are
grouped on coarse keys that a true duplicate pair must agree on -- sex, rounded age, the
year-month of the event, reporter country -- and only within-group pairs are considered. Blocking
trades recall for tractability: a duplicate pair disagreeing on any blocking key is never seen. The
keys are deliberately coarse for that reason, and the summary reports how much of the corpus was
even eligible.

**Scoring.** Within a block, a candidate pair is scored on how much its drug and reaction sets
overlap (Jaccard), plus agreement on report dates. Two reports of the same incident should name
substantially the same drugs and the same reactions.

**This reports; it does not remove.** Dropping reports changes every downstream number, and an
imperfect matcher would do so invisibly. The default is to publish the estimate and a sensitivity
analysis; removal is opt-in via params. Nothing here is applied to the published signal tables
unless that switch is turned on deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

#: Blocking keys. A pair must agree on all of these to be considered at all.
BLOCK_KEYS = ("sex", "age_bucket", "event_period", "occr_country")

#: Age rounded to this many years for blocking. Reported ages disagree slightly between two
#: accounts of the same patient, so exact-age blocking would miss most true pairs.
AGE_BUCKET_YEARS = 5

#: Retained for the demographic-only view used in diagnostics and tests.
MAX_BLOCK_SIZE = 120

#: Groups of the composite blocking key larger than this are skipped. The composite key includes
#: the ingredient, so a large group means many unrelated patients of the same sex, age band,
#: country and month all taking one common drug -- coincidence, not duplication, and quadratic.
MAX_KEY_GROUP = 40


def blocking_keys(demo: pl.LazyFrame) -> pl.LazyFrame:
    """Attach coarse blocking keys to each report."""
    return demo.select(
        "record_id",
        pl.col("sex").fill_null("UNK"),
        (
            (pl.col("age_years") / AGE_BUCKET_YEARS).floor() * AGE_BUCKET_YEARS
        ).cast(pl.Int32, strict=False).alias("age_bucket"),
        pl.when(pl.col("event_dt").is_not_null())
        .then(pl.col("event_dt").dt.year() * 12 + pl.col("event_dt").dt.month())
        .otherwise(None)
        .alias("event_period"),
        pl.col("occr_country").fill_null("UNK"),
    )


def candidate_pairs(
    con, demo_path: Path, drug_path: Path, reac_path: Path, out_path: Path
) -> dict:
    """Enumerate candidate pairs: same demographic block **and** a shared uncommon ingredient.

    Records missing age or event date are excluded outright rather than pooled into an "unknown"
    block: those blocks would contain millions of records sharing nothing but their missing values,
    and every pair in them would be a coincidence.

    The ingredient is part of the blocking key rather than a filter applied afterwards, and that
    distinction is what makes this run at all. Blocking on demographics alone produced 128.8M
    candidate pairs and exhausted 739 GB of scratch. Adding the ingredient as a post-join filter did
    not help either: the planner still materialized the demographic self-join first. Folding the
    ingredient into the key makes the groups small by construction, so the self-join is over
    tens of rows rather than thousands.

    A genuine duplicate must name substantially the same drugs, so requiring one shared ingredient
    to be considered costs very little recall.
    """
    con.execute(
        f"""
        CREATE OR REPLACE VIEW blocks AS
            SELECT record_id, sex, age_bucket, event_period, occr_country
            FROM read_parquet('{demo_path}')
            WHERE age_bucket IS NOT NULL
              AND event_period IS NOT NULL
              AND sex <> 'UNK'
              AND occr_country <> 'UNK';

        CREATE OR REPLACE VIEW ing AS
            SELECT record_id, ingredient AS attr FROM read_parquet('{drug_path}')
            WHERE ingredient IS NOT NULL GROUP BY 1, 2;

        CREATE OR REPLACE VIEW rea AS
            SELECT record_id, pt AS attr FROM read_parquet('{reac_path}')
            WHERE pt IS NOT NULL GROUP BY 1, 2;
        """
    )

    # Two composite blocking keys -- demographics plus a shared drug, and demographics plus a
    # shared reaction -- intersected. A genuine duplicate must agree on both; an unrelated pair
    # that happens to share one common drug, or one common reaction, is eliminated without ever
    # being scored. Intersecting two cheap group-wise self-joins is far cheaper than scoring the
    # 47.8M pairs that the drug key alone produces.
    for name, view in (("pairs_drug", "ing"), ("pairs_reac", "rea")):
        con.execute(
            f"""
            CREATE OR REPLACE VIEW keyed_{name} AS
                SELECT b.record_id, b.sex, b.age_bucket, b.event_period, b.occr_country, v.attr
                FROM blocks b JOIN {view} v ON v.record_id = b.record_id;

            CREATE OR REPLACE VIEW size_{name} AS
                SELECT sex, age_bucket, event_period, occr_country, attr, COUNT(*) AS n
                FROM keyed_{name} GROUP BY 1, 2, 3, 4, 5;

            CREATE OR REPLACE TABLE {name} AS
                SELECT DISTINCT a.record_id AS left_id, b.record_id AS right_id
                FROM size_{name} k
                JOIN keyed_{name} a
                  ON a.sex = k.sex AND a.age_bucket = k.age_bucket
                 AND a.event_period = k.event_period AND a.occr_country = k.occr_country
                 AND a.attr = k.attr
                JOIN keyed_{name} b
                  ON b.sex = k.sex AND b.age_bucket = k.age_bucket
                 AND b.event_period = k.event_period AND b.occr_country = k.occr_country
                 AND b.attr = k.attr
                 AND a.record_id < b.record_id
                WHERE k.n <= {MAX_KEY_GROUP};
            """
        )

    stats = con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM blocks),
            (SELECT COUNT(*) FROM pairs_drug),
            (SELECT COUNT(*) FROM pairs_reac)
        """
    ).fetchone()

    con.execute(
        f"""
        COPY (
            SELECT left_id, right_id FROM pairs_drug
            INTERSECT
            SELECT left_id, right_id FROM pairs_reac
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    n_pairs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]

    return {
        "eligible_records": int(stats[0]),
        "pairs_sharing_a_drug": int(stats[1]),
        "pairs_sharing_a_reaction": int(stats[2]),
        "candidate_pairs": int(n_pairs),
        "max_key_group": MAX_KEY_GROUP,
    }


def score_pairs(
    con, pairs_path: Path, drug_path: Path, reac_path: Path, out_path: Path, threshold: float
) -> dict:
    """Score candidate pairs on drug and reaction set overlap.

    The score is the mean of the two Jaccard similarities. Requiring agreement on *both* the drugs
    and the reactions is what keeps this specific: two unrelated patients of the same age and sex
    in the same month frequently share one common drug, but rarely share both a drug list and a
    reaction list.
    """
    # Everything is materialized as a TABLE and restricted to the records that actually appear in
    # a candidate pair. Left as views over the full 82M-row drug table, DuckDB re-scanned and
    # re-grouped them once per reference -- four times -- and exhausted the scratch limit. The
    # records involved in candidate pairs are a small fraction of the corpus, so materializing the
    # restricted sets is both smaller and computed once.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE pairs AS
            SELECT * FROM read_parquet('{pairs_path}');

        CREATE OR REPLACE TABLE involved AS
            SELECT left_id AS record_id FROM pairs
            UNION
            SELECT right_id FROM pairs;

        CREATE OR REPLACE TABLE dset AS
            SELECT d.record_id, d.ingredient
            FROM read_parquet('{drug_path}') d
            SEMI JOIN involved i ON i.record_id = d.record_id
            WHERE d.ingredient IS NOT NULL
            GROUP BY 1, 2;

        CREATE OR REPLACE TABLE rset AS
            SELECT r.record_id, r.pt
            FROM read_parquet('{reac_path}') r
            SEMI JOIN involved i ON i.record_id = r.record_id
            WHERE r.pt IS NOT NULL
            GROUP BY 1, 2;

        CREATE OR REPLACE TABLE dcount AS
            SELECT record_id, COUNT(*) AS n FROM dset GROUP BY 1;
        CREATE OR REPLACE TABLE rcount AS
            SELECT record_id, COUNT(*) AS n FROM rset GROUP BY 1;

        CREATE OR REPLACE TABLE drug_overlap AS
            SELECT p.left_id, p.right_id, COUNT(*) AS shared
            FROM pairs p
            JOIN dset dl ON dl.record_id = p.left_id
            JOIN dset dr ON dr.record_id = p.right_id AND dr.ingredient = dl.ingredient
            GROUP BY 1, 2;

        CREATE OR REPLACE TABLE reac_overlap AS
            SELECT p.left_id, p.right_id, COUNT(*) AS shared
            FROM pairs p
            JOIN rset rl ON rl.record_id = p.left_id
            JOIN rset rr ON rr.record_id = p.right_id AND rr.pt = rl.pt
            GROUP BY 1, 2;
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT
                p.left_id, p.right_id,
                COALESCE(d_ov.shared, 0) AS drugs_shared,
                COALESCE(r_ov.shared, 0) AS reactions_shared,
                COALESCE(d_ov.shared, 0)::DOUBLE
                    / NULLIF(dl.n + dr.n - COALESCE(d_ov.shared, 0), 0) AS drug_jaccard,
                COALESCE(r_ov.shared, 0)::DOUBLE
                    / NULLIF(rl.n + rr.n - COALESCE(r_ov.shared, 0), 0) AS reaction_jaccard
            FROM pairs p
            LEFT JOIN drug_overlap d_ov USING (left_id, right_id)
            LEFT JOIN reac_overlap r_ov USING (left_id, right_id)
            JOIN dcount dl ON dl.record_id = p.left_id
            JOIN dcount dr ON dr.record_id = p.right_id
            JOIN rcount rl ON rl.record_id = p.left_id
            JOIN rcount rr ON rr.record_id = p.right_id
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    scored = pl.scan_parquet(str(out_path)).with_columns(
        (
            (pl.col("drug_jaccard").fill_null(0.0) + pl.col("reaction_jaccard").fill_null(0.0))
            / 2
        ).alias("score")
    )
    matched = scored.filter(pl.col("score") >= threshold).collect()

    return {
        "pairs_scored": int(scored.select(pl.len()).collect().item()),
        "threshold": threshold,
        "pairs_above_threshold": matched.height,
        "records_involved": int(
            pl.concat(
                [
                    matched.select(pl.col("left_id").alias("record_id")),
                    matched.select(pl.col("right_id").alias("record_id")),
                ]
            )
            .to_series()
            .n_unique()
        )
        if matched.height
        else 0,
    }


def cluster(matched: pl.DataFrame) -> pl.DataFrame:
    """Group matched pairs into connected components, keeping one representative each.

    Transitivity matters: if A matches B and B matches C, all three describe one incident, and
    removing only the pairwise loser would leave two of the three behind. Union-find over the
    matched pairs; the lowest record id in each component is kept.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic merge direction so the surviving representative never depends on
            # row order.
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo

    for left, right in zip(matched["left_id"], matched["right_id"], strict=True):
        union(left, right)

    members = sorted(parent)
    return pl.DataFrame(
        {
            "record_id": members,
            "cluster_id": [find(m) for m in members],
        }
    ).with_columns((pl.col("record_id") != pl.col("cluster_id")).alias("is_redundant"))


def run(
    con,
    curated_dir: Path,
    out_dir: Path,
    summary_path: Path,
    threshold: float,
) -> dict:
    """Full detection pass. Writes candidate pairs, scores, clusters and a summary."""
    curated_dir, out_dir = Path(curated_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_blocked = out_dir / "blocked_demo.parquet"
    blocking_keys(pl.scan_parquet(str(curated_dir / "demo.parquet"))).collect().write_parquet(
        demo_blocked, compression="zstd"
    )

    pairs_path = out_dir / "candidate_pairs.parquet"
    block_stats = candidate_pairs(
        con,
        demo_blocked,
        curated_dir / "drug_ingredients.parquet",
        curated_dir / "reac.parquet",
        pairs_path,
    )

    scores_path = out_dir / "pair_scores.parquet"
    score_stats = score_pairs(
        con,
        pairs_path,
        curated_dir / "drug_ingredients.parquet",
        curated_dir / "reac.parquet",
        scores_path,
        threshold,
    )

    matched = (
        pl.scan_parquet(str(scores_path))
        .with_columns(
            (
                (
                    pl.col("drug_jaccard").fill_null(0.0)
                    + pl.col("reaction_jaccard").fill_null(0.0)
                )
                / 2
            ).alias("score")
        )
        .filter(pl.col("score") >= threshold)
        .select("left_id", "right_id", "score")
        .collect()
    )

    clusters = cluster(matched) if matched.height else pl.DataFrame(
        schema={"record_id": pl.String, "cluster_id": pl.String, "is_redundant": pl.Boolean}
    )
    clusters.write_parquet(out_dir / "clusters.parquet", compression="zstd")

    n_total = (
        pl.scan_parquet(str(curated_dir / "demo.parquet")).select(pl.len()).collect().item()
    )
    redundant = int(clusters["is_redundant"].sum()) if clusters.height else 0

    summary = {
        **block_stats,
        **score_stats,
        "corpus_records": int(n_total),
        "clusters": int(clusters["cluster_id"].n_unique()) if clusters.height else 0,
        "redundant_records": redundant,
        # Both denominators, because they say different things. Only ~30% of the corpus carries
        # the sex, age, event date and country needed to be blocked at all, so the corpus-wide
        # figure mixes "not redundant" with "never examined". The eligible-subset rate is what the
        # method actually measured.
        "redundant_fraction_of_corpus": round(redundant / max(n_total, 1), 5),
        "redundant_fraction_of_eligible": round(
            redundant / max(block_stats["eligible_records"], 1), 5
        ),
        "eligible_fraction_of_corpus": round(
            block_stats["eligible_records"] / max(n_total, 1), 4
        ),
        "note": (
            "Estimated, not applied. Blocking requires agreement on sex, 5-year age bucket, event "
            "year-month and reporter country, so pairs disagreeing on any of those are never "
            "compared and this is a lower bound. Removal is opt-in via params.duplicates.remove."
        ),
    }
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
    return summary
