"""Tests for drug-drug similarity from reaction profiles.

The validation test is the important one. A similarity measure over adverse-event profiles will
always produce clusters; whether those clusters mean anything is a separate question, and the 2020
notebook did not answer it. Here, planted drug classes must come out more similar to themselves
than to the corpus, or the measure is not capturing pharmacology and nothing downstream of it
should be read.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from faers.stats.similarity import (
    build_profiles,
    cluster_drugs,
    cosine_similarity,
    nearest_neighbours,
    validate_against_known_classes,
)


def profile_frame(rows):
    """A minimal scored-table stand-in: ingredient, reaction, IC, curated flag."""
    return pl.DataFrame(
        rows, schema={"ingredient": pl.String, "pt": pl.String, "ic": pl.Float64}
    ).with_columns(pl.lit(True).alias("ingredient_curated")).lazy()


def synthetic_classes(n_per_class=4, n_shared=30, n_noise=30, seed=0):
    """Drugs in three classes, each class sharing a distinct reaction signature."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(3):
        for d in range(n_per_class):
            drug = f"CLASS{c}_DRUG{d}"
            for r in range(n_shared):  # the class signature
                rows.append({"ingredient": drug, "pt": f"C{c}_REACTION{r}", "ic": 2.0})
            for _ in range(n_noise):  # drug-specific noise from a common pool
                rows.append(
                    {
                        "ingredient": drug,
                        "pt": f"COMMON{rng.integers(0, 200)}",
                        "ic": float(rng.uniform(0.1, 0.5)),
                    }
                )
    return profile_frame(rows)


class TestProfiles:
    def test_matrix_shape_matches_labels(self):
        lf = synthetic_classes()
        matrix, drugs, reactions = build_profiles(lf, min_reactions=5, max_drugs=100)
        assert matrix.shape == (len(drugs), len(reactions))
        assert len(drugs) == 12

    def test_uncurated_ingredients_are_excluded(self):
        """An unmapped product string shares its profile with the substance; including it would
        make both similarities meaningless."""
        lf = (
            pl.DataFrame(
                {
                    "ingredient": ["REAL"] * 6 + ["BRAND . PEN"] * 6,
                    "pt": [f"R{i}" for i in range(6)] * 2,
                    "ic": [1.0] * 12,
                    "ingredient_curated": [True] * 6 + [False] * 6,
                }
            )
            .lazy()
        )
        _, drugs, _ = build_profiles(lf, min_reactions=3, max_drugs=100)
        assert drugs == ["REAL"]

    def test_empty_input(self):
        lf = pl.DataFrame(
            schema={
                "ingredient": pl.String,
                "pt": pl.String,
                "ic": pl.Float64,
                "ingredient_curated": pl.Boolean,
            }
        ).lazy()
        matrix, drugs, reactions = build_profiles(lf)
        assert drugs == [] and reactions == []


class TestCosine:
    def test_identical_profiles_score_one(self):
        from scipy.sparse import csr_matrix

        m = csr_matrix(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
        assert cosine_similarity(m)[0, 1] == pytest.approx(1.0)

    def test_disjoint_profiles_score_zero(self):
        from scipy.sparse import csr_matrix

        m = csr_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        assert cosine_similarity(m)[0, 1] == pytest.approx(0.0)

    def test_is_symmetric_with_unit_diagonal(self):
        from scipy.sparse import csr_matrix

        rng = np.random.default_rng(1)
        m = csr_matrix(rng.random((8, 12)))
        sim = cosine_similarity(m)
        assert np.allclose(sim, sim.T)
        assert np.allclose(np.diag(sim), 1.0)

    def test_zero_profile_does_not_produce_nan(self):
        from scipy.sparse import csr_matrix

        m = csr_matrix(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert np.isfinite(cosine_similarity(m)).all()


class TestValidation:
    def test_planted_classes_are_recovered(self):
        """The core property: same-class drugs must be markedly more similar than average."""
        lf = synthetic_classes()
        matrix, drugs, _ = build_profiles(lf, min_reactions=5, max_drugs=100)
        sim = cosine_similarity(matrix)
        classes = {f"class{c}": (f"CLASS{c}_",) for c in range(3)}
        result = validate_against_known_classes(sim, drugs, classes)
        assert result.height == 3
        for row in result.to_dicts():
            assert row["members_found"] == 4
            assert row["ratio"] > 2.0, f"{row['drug_class']} not recovered: {row}"

    def test_unrelated_drugs_show_no_class_structure(self):
        """The negative control: random profiles must not produce a high ratio."""
        rng = np.random.default_rng(7)
        rows = [
            {
                "ingredient": f"RANDOM_DRUG{d}",
                "pt": f"R{rng.integers(0, 300)}",
                "ic": float(rng.uniform(0.5, 2.0)),
            }
            for d in range(12)
            for _ in range(40)
        ]
        matrix, drugs, _ = build_profiles(profile_frame(rows), min_reactions=5, max_drugs=100)
        sim = cosine_similarity(matrix)
        result = validate_against_known_classes(sim, drugs, {"fake": ("RANDOM_DRUG",)})
        assert result.row(0, named=True)["ratio"] < 2.0

    def test_class_with_one_member_reports_null_not_a_number(self):
        lf = synthetic_classes()
        matrix, drugs, _ = build_profiles(lf, min_reactions=5, max_drugs=100)
        sim = cosine_similarity(matrix)
        result = validate_against_known_classes(sim, drugs, {"lonely": ("CLASS0_DRUG0",)})
        row = result.row(0, named=True)
        assert row["members_found"] == 1
        assert row["mean_within_class"] is None


class TestNeighboursAndClusters:
    def test_nearest_neighbour_is_same_class(self):
        lf = synthetic_classes()
        matrix, drugs, _ = build_profiles(lf, min_reactions=5, max_drugs=100)
        sim = cosine_similarity(matrix)
        nn = nearest_neighbours(sim, drugs, k=1)
        for row in nn.to_dicts():
            assert row["ingredient"].split("_")[0] == row["neighbour"].split("_")[0]

    def test_a_drug_is_never_its_own_neighbour(self):
        lf = synthetic_classes()
        matrix, drugs, _ = build_profiles(lf, min_reactions=5, max_drugs=100)
        nn = nearest_neighbours(cosine_similarity(matrix), drugs, k=3)
        assert not (nn["ingredient"] == nn["neighbour"]).any()

    def test_clustering_separates_planted_classes(self):
        lf = synthetic_classes()
        matrix, drugs, _ = build_profiles(lf, min_reactions=5, max_drugs=100)
        clusters = cluster_drugs(cosine_similarity(matrix), drugs, n_clusters=3)
        by_class = {}
        for row in clusters.to_dicts():
            by_class.setdefault(row["ingredient"].split("_")[0], set()).add(row["cluster"])
        # Each planted class should land in exactly one cluster.
        assert all(len(v) == 1 for v in by_class.values()), by_class

    def test_clustering_is_deterministic(self):
        lf = synthetic_classes()
        matrix, drugs, _ = build_profiles(lf, min_reactions=5, max_drugs=100)
        sim = cosine_similarity(matrix)
        runs = {tuple(cluster_drugs(sim, drugs, 3)["cluster"].to_list()) for _ in range(4)}
        assert len(runs) == 1
