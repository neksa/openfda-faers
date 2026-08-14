"""Tests for brand-to-ingredient resolution.

The unambiguity guard is the load-bearing part. First-token matching is only safe because a token
whose NDC records disagree about the ingredient is *declined* rather than resolved to the most
common answer. Without it, `SODIUM CHLORIDE` would inherit whichever sodium formulation happened to
be most numerous in the directory — a fabricated mapping applied to hundreds of thousands of rows.
"""

from __future__ import annotations

import json
import zipfile

import polars as pl
import pytest

from faers.brands import DOMINANCE_THRESHOLD, STOPWORD_TOKENS, build_ndc_index


def make_ndc_zip(tmp_path, records):
    payload = {"results": records}
    path = tmp_path / "ndc.json.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("drug-ndc.json", json.dumps(payload))
    return path


def rec(brand, ingredients, base=None):
    return {
        "brand_name": brand,
        "brand_name_base": base if base is not None else brand,
        "active_ingredients": [{"name": i} for i in ingredients],
    }


class TestNdcIndex:
    def test_unanimous_token_resolves(self, tmp_path):
        z = make_ndc_zip(tmp_path, [rec("ParaGard T 380A", ["COPPER"])] * 4)
        idx = build_ndc_index(z)
        row = idx.filter(pl.col("token") == "PARAGARD").row(0, named=True)
        assert row["ingredients"] == "COPPER"
        assert row["dominance"] == 1.0
        assert row["usable"]

    def test_ambiguous_token_is_declined(self, tmp_path):
        """The critical safety case: disagreeing records must produce no mapping."""
        z = make_ndc_zip(
            tmp_path,
            [rec("Mixco Alpha", ["ALPHA"])] * 5 + [rec("Mixco Beta", ["BETA"])] * 5,
        )
        idx = build_ndc_index(z)
        row = idx.filter(pl.col("token") == "MIXCO").row(0, named=True)
        assert row["dominance"] == 0.5
        assert not row["usable"], "a 50/50 split must not resolve"

    def test_dominant_token_resolves_despite_minority(self, tmp_path):
        z = make_ndc_zip(
            tmp_path,
            [rec("Domco One", ["MAIN"])] * 19 + [rec("Domco Two", ["OTHER"])],
        )
        idx = build_ndc_index(z)
        row = idx.filter(pl.col("token") == "DOMCO").row(0, named=True)
        assert row["dominance"] >= DOMINANCE_THRESHOLD
        assert row["usable"] and row["ingredients"] == "MAIN"

    def test_generic_tokens_are_never_usable(self, tmp_path):
        z = make_ndc_zip(tmp_path, [rec("Sodium Chloride Injection", ["SODIUM CHLORIDE"])] * 50)
        idx = build_ndc_index(z)
        row = idx.filter(pl.col("token") == "SODIUM").row(0, named=True)
        assert row["dominance"] == 1.0
        assert not row["usable"], "generic substance words must be excluded regardless of dominance"
        assert "SODIUM" in STOPWORD_TOKENS

    def test_records_without_ingredients_are_skipped(self, tmp_path):
        """Biologics often carry no active_ingredients; they must not create empty mappings."""
        z = make_ndc_zip(tmp_path, [rec("Humira", []), rec("Lipitor", ["ATORVASTATIN"])])
        idx = build_ndc_index(z)
        assert "HUMIRA" not in idx["token"].to_list()
        assert "LIPITOR" in idx["token"].to_list()

    def test_multi_ingredient_uses_backslash_convention(self, tmp_path):
        z = make_ndc_zip(
            tmp_path, [rec("Advair Diskus", ["SALMETEROL XINAFOATE", "FLUTICASONE PROPIONATE"])] * 3
        )
        idx = build_ndc_index(z)
        got = idx.filter(pl.col("token") == "ADVAIR").row(0, named=True)["ingredients"]
        # Sorted and backslash-joined, matching FDA's own prod_ai convention.
        assert got == "FLUTICASONE PROPIONATE\\SALMETEROL XINAFOATE"

    def test_short_tokens_rejected(self, tmp_path):
        z = make_ndc_zip(tmp_path, [rec("K2 Tablets", ["POTASSIUM"])] * 10)
        idx = build_ndc_index(z)
        row = idx.filter(pl.col("token") == "K2")
        if row.height:
            assert not row.row(0, named=True)["usable"]

    def test_index_is_deterministic(self, tmp_path):
        recs = [rec("Tieco A", ["AAA"])] * 5 + [rec("Tieco B", ["BBB"])] * 5
        z = make_ndc_zip(tmp_path, recs)
        picks = {build_ndc_index(z).filter(pl.col("token") == "TIECO")
                 .row(0, named=True)["ingredients"] for _ in range(5)}
        assert len(picks) == 1


class TestRxNavQueryConstruction:
    """Regression test for a silent failure.

    Passing ``tty="IN+MIN"`` to urlencode escapes the plus to %2B, so RxNav receives one term type
    named "IN+MIN" — which does not exist — and returns an empty result. Nothing raised: every
    brand simply resolved to None, and 250 lookups produced zero ingredients before it was noticed.
    The term types must be space-separated so they encode to a real "+" on the wire.
    """

    def test_tty_parameter_encodes_to_a_plus_not_percent_2b(self):
        import inspect
        import urllib.parse

        from faers import brands

        source = inspect.getsource(brands.rxnav_lookup)
        assert 'tty="IN MIN"' in source, "term types must be space-separated, not '+'-joined"
        assert 'tty="IN+MIN"' not in source

        # And confirm why: urlencode turns a literal plus into %2B, a space into a plus.
        assert urllib.parse.urlencode({"tty": "IN+MIN"}) == "tty=IN%2BMIN"
        assert urllib.parse.urlencode({"tty": "IN MIN"}) == "tty=IN+MIN"

    def test_lookup_failures_return_none_rather_than_raising(self, monkeypatch):
        """A network failure must degrade to 'unresolved', never abort the stage."""
        from faers import brands

        monkeypatch.setattr(brands, "_rxnav_json", lambda *a, **k: None)
        assert brands.rxnav_ingredients("ANYTHING") is None


class TestCleaningForBrandLookup:
    """The cleaner must reduce a product string to a token the brand index can match."""

    @pytest.mark.parametrize(
        ("raw", "expected_token"),
        [
            ("HUMIRA . PEN", "HUMIRA"),
            ("DURAGESIC-100", "DURAGESIC"),
            ("ADVAIR DISKUS 100/50", "ADVAIR"),
            ("PARAGARD 380A", "PARAGARD"),
            ("LANTUS SOLOSTAR", "LANTUS"),
            ("VENTOLIN HFA", "VENTOLIN"),
        ],
    )
    def test_product_strings_reduce_to_brand_token(self, raw, expected_token):
        from faers.normalize import clean_expr

        cleaned = (
            pl.DataFrame({"d": [raw]}).with_columns(clean_expr("d").alias("c"))["c"][0]
        )
        assert cleaned.split(" ")[0] == expected_token

    @pytest.mark.parametrize("raw", ["NONOXYNOL-9", "OMEGA-3", "COQ-10", "VITAMIN B-12"])
    def test_trailing_numbers_in_real_ingredients_survive(self, raw):
        """Stripping these would invent substances that do not exist."""
        from faers.normalize import clean_expr

        cleaned = (
            pl.DataFrame({"d": [raw]}).with_columns(clean_expr("d").alias("c"))["c"][0]
        )
        assert cleaned == raw

    def test_percent_strength_is_stripped(self):
        from faers.normalize import clean_expr

        cleaned = (
            pl.DataFrame({"d": ["SODIUM CHLORIDE 0.9%"]})
            .with_columns(clean_expr("d").alias("c"))["c"][0]
        )
        assert cleaned == "SODIUM CHLORIDE"
