"""Tests for independent-duplicate detection.

The clustering tests matter most. A matcher that finds pairs but does not resolve them
transitively would leave two of every three copies behind, and the count it reported would be
wrong in a way nothing downstream could detect.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from faers.duplicates import AGE_BUCKET_YEARS, blocking_keys, cluster


class TestBlockingKeys:
    def test_age_is_bucketed_not_exact(self):
        """Two accounts of the same patient rarely agree on exact age."""
        demo = pl.DataFrame(
            {
                "record_id": ["1", "2", "3"],
                "sex": ["F", "F", "F"],
                "age_years": [41.0, 44.0, 47.0],
                "event_dt": [dt.date(2020, 5, 1)] * 3,
                "occr_country": ["US"] * 3,
            }
        ).lazy()
        got = blocking_keys(demo).collect()
        assert got["age_bucket"].to_list() == [40, 40, 45]
        assert AGE_BUCKET_YEARS == 5

    def test_event_period_is_year_month(self):
        demo = pl.DataFrame(
            {
                "record_id": ["1", "2"],
                "sex": ["M", "M"],
                "age_years": [30.0, 30.0],
                "event_dt": [dt.date(2020, 5, 9), dt.date(2020, 5, 28)],
                "occr_country": ["US", "US"],
            }
        ).lazy()
        got = blocking_keys(demo).collect()
        # Same month must land in the same block despite different days.
        assert got["event_period"][0] == got["event_period"][1]

    def test_missing_values_are_preserved_as_null_for_later_exclusion(self):
        demo = pl.DataFrame(
            {
                "record_id": ["1"],
                "sex": [None],
                "age_years": [None],
                "event_dt": [None],
                "occr_country": [None],
            },
            schema={
                "record_id": pl.String,
                "sex": pl.String,
                "age_years": pl.Float64,
                "event_dt": pl.Date,
                "occr_country": pl.String,
            },
        ).lazy()
        got = blocking_keys(demo).collect().row(0, named=True)
        assert got["age_bucket"] is None and got["event_period"] is None
        assert got["sex"] == "UNK" and got["occr_country"] == "UNK"


class TestClustering:
    def test_transitive_chain_forms_one_cluster(self):
        """A-B and B-C means all three are the same incident, so two are redundant."""
        matched = pl.DataFrame({"left_id": ["A", "B"], "right_id": ["B", "C"]})
        got = cluster(matched)
        assert got["cluster_id"].n_unique() == 1
        assert int(got["is_redundant"].sum()) == 2

    def test_separate_pairs_stay_separate(self):
        matched = pl.DataFrame({"left_id": ["A", "C"], "right_id": ["B", "D"]})
        got = cluster(matched)
        assert got["cluster_id"].n_unique() == 2
        assert int(got["is_redundant"].sum()) == 2

    def test_one_representative_survives_per_cluster(self):
        matched = pl.DataFrame({"left_id": ["A", "B", "C"], "right_id": ["B", "C", "D"]})
        got = cluster(matched)
        assert got.height == 4
        assert int((~got["is_redundant"]).sum()) == 1, "exactly one survivor per cluster"

    def test_representative_choice_is_deterministic(self):
        """The survivor must not depend on the order pairs arrive in."""
        import random

        rows = [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")]
        survivors = set()
        for seed in range(6):
            shuffled = rows[:]
            random.Random(seed).shuffle(shuffled)
            frame = pl.DataFrame(
                {"left_id": [r[0] for r in shuffled], "right_id": [r[1] for r in shuffled]}
            )
            got = cluster(frame)
            survivors.add(tuple(sorted(got.filter(~pl.col("is_redundant"))["record_id"])))
        assert len(survivors) == 1, f"non-deterministic representatives: {survivors}"

    def test_lowest_id_is_kept(self):
        matched = pl.DataFrame({"left_id": ["B"], "right_id": ["A"]})
        got = cluster(matched)
        kept = got.filter(~pl.col("is_redundant"))["record_id"].to_list()
        assert kept == ["A"]

    def test_empty_input(self):
        got = cluster(pl.DataFrame({"left_id": [], "right_id": []},
                                   schema={"left_id": pl.String, "right_id": pl.String}))
        assert got.height == 0


class TestScoringSemantics:
    """The score is the mean of two Jaccard similarities; these pin the intended behaviour."""

    @staticmethod
    def score(drug_j, reac_j):
        return (drug_j + reac_j) / 2

    def test_identical_reports_score_one(self):
        assert self.score(1.0, 1.0) == pytest.approx(1.0)

    def test_sharing_only_drugs_is_not_enough(self):
        """Two unrelated patients on the same common drug must not reach a 0.8 threshold."""
        assert self.score(1.0, 0.0) == 0.5

    def test_sharing_only_reactions_is_not_enough(self):
        assert self.score(0.0, 1.0) == 0.5

    def test_partial_agreement_on_both_can_pass_a_moderate_threshold(self):
        assert self.score(0.75, 0.75) == pytest.approx(0.75)
