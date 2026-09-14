"""Contract tests for the quarter index and the two-era column mapping.

The mapping layer is where this pipeline is most likely to break silently rather than loudly. FDA
renamed ``gndr_cod`` to ``sex`` partway through the FAERS era; a mapping that named only ``sex``
parsed 2012Q4 without error and produced 100% "UNK" patient sex. Nothing raised. These tests pin the
behaviours that would let that recur.
"""

from __future__ import annotations

import polars as pl
import pytest

from faers import schema as S
from faers.sources import Era, Quarter, parse_quarter, quarters


class TestQuarterIndex:
    def test_era_boundary_is_2012q3_to_2012q4(self):
        assert Quarter(2012, 3).era is Era.LAERS
        assert Quarter(2012, 4).era is Era.FAERS

    def test_url_stem_follows_era(self):
        assert "aers_ascii_2012q3" in Quarter(2012, 3).url
        assert "faers_ascii_2012q4" in Quarter(2012, 4).url
        # The legacy stem must not be a substring match of the modern one.
        assert not Quarter(2012, 4).url.split("/")[-1].startswith("aers_")

    def test_full_coverage_is_90_quarters(self):
        qs = quarters("2004Q1", "2026Q2")
        assert len(qs) == 90
        assert qs[0].label == "2004Q1" and qs[-1].label == "2026Q2"
        assert sum(q.era is Era.LAERS for q in qs) == 35
        assert sum(q.era is Era.FAERS for q in qs) == 55

    def test_quarters_are_contiguous_and_ordered(self):
        qs = quarters("2011Q1", "2013Q4")
        assert [q.label for q in qs][:5] == [
            "2011Q1", "2011Q2", "2011Q3", "2011Q4", "2012Q1"
        ]
        assert all(b.index == a.index + 1 for a, b in zip(qs, qs[1:], strict=False))

    def test_two_digit_year_used_in_member_names(self):
        assert Quarter(2004, 1).yy == "04"
        assert Quarter(2026, 2).yy == "26"

    @pytest.mark.parametrize("bad", ["2026Q5", "26Q1", "2026", "2026-Q1", ""])
    def test_rejects_malformed_labels(self, bad):
        with pytest.raises(ValueError):
            parse_quarter(bad)

    def test_rejects_reversed_range(self):
        with pytest.raises(ValueError):
            quarters("2026Q2", "2004Q1")


class TestColumnMapping:
    def test_canonical_columns_match_across_eras(self):
        """LAERS and FAERS outputs must be concatenable, or the 90 files cannot be scanned."""
        for table in S.MAPS:
            cols = S.canonical_columns(table)
            assert "foll_seq" not in cols
            assert len(cols) == len(set(cols))

    def test_gndr_cod_resolves_to_sex(self):
        """The specific defect: early FAERS quarters use gndr_cod, later ones use sex."""
        early = pl.DataFrame(
            {"primaryid": ["1"], "caseid": ["1"], "gndr_cod": ["F"]}
        ).lazy()
        out = S.apply_map(early, "DEMO", Era.FAERS).collect()
        assert out["sex"][0] == "F"

        late = pl.DataFrame({"primaryid": ["1"], "caseid": ["1"], "sex": ["M"]}).lazy()
        assert S.apply_map(late, "DEMO", Era.FAERS).collect()["sex"][0] == "M"

    def test_absent_columns_become_null_not_missing(self):
        """A LAERS DRUG table has no prod_ai; the column must exist and be null."""
        laers = pl.DataFrame(
            {"ISR": ["1"], "DRUG_SEQ": ["1"], "DRUGNAME": ["ASPIRIN"]}
        ).lazy()
        out = S.apply_map(laers, "DRUG", Era.LAERS).collect()
        assert "prod_ai" in out.columns
        assert out["prod_ai"].null_count() == out.height
        assert out["drugname"][0] == "ASPIRIN"

    def test_header_case_drift_is_tolerated(self):
        upper = pl.DataFrame({"PRIMARYID": ["7"], "CASEID": ["7"], "SEX": ["F"]}).lazy()
        out = S.apply_map(upper, "DEMO", Era.FAERS).collect()
        assert out["record_id"][0] == "7" and out["sex"][0] == "F"

    def test_missing_join_key_raises(self):
        """A missing key must fail loudly -- it silently empties every downstream join."""
        broken = pl.DataFrame({"drugname": ["ASPIRIN"]}).lazy()
        mapped = S.apply_map(broken, "DRUG", Era.LAERS)
        mapped = mapped.drop("record_id")
        with pytest.raises(ValueError, match="missing required column"):
            S.require(mapped, "DRUG", "2004Q1")

    def test_laers_keeps_foll_seq_for_version_derivation(self):
        laers = pl.DataFrame({"ISR": ["1"], "CASE": ["9"], "FOLL_SEQ": ["2"]}).lazy()
        out = S.apply_map(laers, "DEMO", Era.LAERS).collect()
        assert out["foll_seq"][0] == "2"


class TestValueVocabularies:
    def test_suspect_roles_are_primary_and_secondary(self):
        assert S.SUSPECT_ROLES == ("PS", "SS")
        assert all(r in S.ROLE_LABELS for r in S.SUSPECT_ROLES)

    def test_sex_vocabulary_collapses_to_three_values(self):
        assert set(S.SEX_MAP.values()) == {"F", "M", "UNK"}

    def test_age_units_cover_the_faers_codes(self):
        for code in ("DEC", "YR", "MON", "WK", "DY", "HR"):
            assert code in S.AGE_TO_YEARS
        assert S.AGE_TO_YEARS["YR"] == 1.0
        assert S.AGE_TO_YEARS["DEC"] == 10.0
        assert S.AGE_TO_YEARS["MON"] == pytest.approx(1 / 12)
