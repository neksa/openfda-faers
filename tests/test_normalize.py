"""Tests for drug name resolution.

The determinism test here guards a bug that reproducibility claims live or die on. Where a reported
drug name mapped to two ingredient strings with equal support, the winner was decided by whichever
row Polars' multithreaded group-by saw first. Two runs of the pipeline over byte-identical inputs
produced different ingredient tables, and the difference propagated all the way to the signal
counts. Nothing errored; the numbers just moved.
"""

from __future__ import annotations

import polars as pl
import pytest

from faers.normalize import _NON_DRUG_RE, build_dictionary, clean_expr, explode_ingredients


def clean_one(text: str) -> str:
    return (
        pl.DataFrame({"drugname": [text]})
        .with_columns(clean_expr("drugname").alias("c"))["c"][0]
    )


class TestCleaning:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ASPIRIN 81MG TABLET", "ASPIRIN"),
            ("SIMETHICONE CHW 80MG", "SIMETHICONE"),
            ("METFORMIN HCL ER 500 MG", "METFORMIN"),
            ("ONE A DAY       /07499601/", "ONE A DAY"),
            ("LIPITOR (ATORVASTATIN CALCIUM)", "LIPITOR"),
            ("  ibuprofen  ", "IBUPROFEN"),
        ],
    )
    def test_strips_strength_form_and_codes(self, raw, expected):
        assert clean_one(raw) == expected

    def test_preserves_combination_separators_for_later_splitting(self):
        # Splitting happens in explode_ingredients; cleaning must not destroy the separator.
        assert "\\" in clean_one("INSULIN GLARGINE\\LIXISENATIDE")
        assert ";" in clean_one("MENTHOL;ZINC OXIDE")

    def test_does_not_mangle_hyphenated_names(self):
        assert clean_one("AMOXICILLIN-BIOCHEMIE") == "AMOXICILLIN-BIOCHEMIE"


class TestExplodeIngredients:
    def test_splits_combination_products(self):
        lf = pl.DataFrame({"ingredient_raw": ["INSULIN GLARGINE\\LIXISENATIDE"]}).lazy()
        out = explode_ingredients(lf, "ingredient_raw").collect()
        assert out["ingredient"].to_list() == ["INSULIN GLARGINE", "LIXISENATIDE"]

    def test_single_ingredient_yields_one_row(self):
        lf = pl.DataFrame({"ingredient_raw": ["ASPIRIN"]}).lazy()
        assert explode_ingredients(lf, "ingredient_raw").collect().height == 1

    def test_semicolons_and_plus_are_separators(self):
        lf = pl.DataFrame({"ingredient_raw": ["MENTHOL;ZINC OXIDE", "A+B"]}).lazy()
        out = explode_ingredients(lf, "ingredient_raw").collect()
        assert "MENTHOL" in out["ingredient"].to_list()
        assert "ZINC OXIDE" in out["ingredient"].to_list()


class TestNonDrugFilter:
    @pytest.mark.parametrize(
        "text",
        [
            "12 UNSPECIFIED MEDICATIONS",
            "UNKNOWN MEDICATION",
            "UNSPECIFIED INGREDIENT",
            "BLINDED THERAPY",
            "CONCOMITANT DRUGS",
        ],
    )
    def test_matches_placeholders(self, text):
        got = pl.DataFrame({"x": [text]}).select(
            pl.col("x").str.contains(_NON_DRUG_RE).alias("m")
        )["m"][0]
        assert got, f"{text!r} should be filtered as a non-drug placeholder"

    @pytest.mark.parametrize(
        "text",
        ["ASPIRIN", "INSULIN GLARGINE", "NONOXYNOL-9", "NORETHINDRONE ACETATE"],
    )
    def test_does_not_match_real_ingredients(self, text):
        got = pl.DataFrame({"x": [text]}).select(
            pl.col("x").str.contains(_NON_DRUG_RE).alias("m")
        )["m"][0]
        assert not got, f"{text!r} is a real ingredient and must survive"


class TestRxcuiLookup:
    """Ingredient identifiers are joined by name, not carried positionally through the explode."""

    def test_pairs_parallel_ingredient_and_rxcui_lists(self):
        from faers.normalize import rxcui_lookup

        got = rxcui_lookup(
            {
                "HUMIRA": {"ingredients": "ADALIMUMAB", "rxcuis": "642036"},
                "ADVAIR": {
                    "ingredients": "FLUTICASONE\\SALMETEROL",
                    "rxcuis": "41126\\36117",
                },
            }
        )
        as_dict = dict(zip(got["ingredient"], got["ingredient_rxcui"], strict=True))
        assert as_dict["ADALIMUMAB"] == "642036"
        # Combination products must not cross-assign identifiers between their ingredients.
        assert as_dict["FLUTICASONE"] == "41126"
        assert as_dict["SALMETEROL"] == "36117"

    def test_ignores_entries_without_identifiers(self):
        from faers.normalize import rxcui_lookup

        got = rxcui_lookup({"X": {"ingredients": "SOMETHING", "rxcuis": ""}})
        assert got.height == 0

    def test_tolerates_mismatched_list_lengths(self):
        """A truncated RxCUI list must drop the unpaired names, not misalign them."""
        from faers.normalize import rxcui_lookup

        got = rxcui_lookup({"X": {"ingredients": "AAA\\BBB\\CCC", "rxcuis": "1\\2"}})
        as_dict = dict(zip(got["ingredient"], got["ingredient_rxcui"], strict=True))
        assert as_dict == {"AAA": "1", "BBB": "2"}

    def test_handles_empty_input(self):
        from faers.normalize import rxcui_lookup

        assert rxcui_lookup(None).height == 0
        assert rxcui_lookup({}).height == 0


class TestDictionaryDeterminism:
    def test_equal_support_ties_resolve_identically_across_row_orders(self):
        """The regression test: a tie must not be decided by row arrival order."""
        rows = (
            [{"drugname": "TIEBRAND", "prod_ai": "ZEBRAINE"}] * 3
            + [{"drugname": "TIEBRAND", "prod_ai": "ALPHAINE"}] * 3
            + [{"drugname": "CLEARBRAND", "prod_ai": "CLEARINE"}] * 5
        )
        base = pl.DataFrame(rows)
        winners = {
            build_dictionary(
                base.sample(fraction=1.0, shuffle=True, seed=s).lazy(), min_support=2
            )
            .filter(pl.col("name_raw") == "TIEBRAND")["prod_ai"][0]
            for s in range(8)
        }
        assert len(winners) == 1, f"non-deterministic tie-breaking: {winners}"

    def test_higher_support_still_wins(self):
        rows = (
            [{"drugname": "BRAND", "prod_ai": "ZZZ_MAJORITY"}] * 9
            + [{"drugname": "BRAND", "prod_ai": "AAA_MINORITY"}] * 2
        )
        d = build_dictionary(pl.DataFrame(rows).lazy(), min_support=2)
        # Alphabetical order must not override the support ranking.
        assert d.filter(pl.col("name_raw") == "BRAND")["prod_ai"][0] == "ZZZ_MAJORITY"

    def test_min_support_excludes_single_observations(self):
        rows = [{"drugname": "ONEOFF", "prod_ai": "TYPO_INGREDIENT"}]
        assert build_dictionary(pl.DataFrame(rows).lazy(), min_support=2).height == 0
