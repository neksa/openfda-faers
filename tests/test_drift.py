"""Tests for empirical MedDRA vocabulary-drift detection.

Built on synthetic corpora where the right answer is known by construction: a term that is planted
as stable must not be flagged, and a term planted with a vocabulary-change signature must be. This
matters because the detector's whole purpose is to separate vocabulary artefacts from real trends,
and a detector that flags everything is as useless as one that flags nothing.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from faers.stats.drift import ABSENT_SHARE, STEP_RATIO, detect, term_trajectories


def build_corpus(term_plan, quarters=40, reports_per_quarter=500, seed=0):
    """Synthesize DEMO and REAC frames from a per-term share schedule.

    ``term_plan`` maps a term to a callable taking the quarter index (0-based) and returning the
    share of that quarter's reports mentioning it.
    """
    rng = __import__("random").Random(seed)
    demo_rows, reac_rows = [], []
    rid = 0
    start_year = 2006
    for q in range(quarters):
        year = start_year + q // 4
        month = 1 + 3 * (q % 4)
        for _ in range(reports_per_quarter):
            rid += 1
            demo_rows.append({"record_id": str(rid), "fda_dt": dt.date(year, month, 15)})
            for term, schedule in term_plan.items():
                if rng.random() < schedule(q):
                    reac_rows.append({"record_id": str(rid), "pt": term})
    demo = pl.DataFrame(demo_rows, schema={"record_id": pl.String, "fda_dt": pl.Date}).lazy()
    reac = pl.DataFrame(reac_rows, schema={"record_id": pl.String, "pt": pl.String}).lazy()
    return reac, demo


def flags_for(result, term):
    row = result.filter(pl.col("pt") == term)
    assert row.height == 1, f"{term} was not tested"
    return row.row(0, named=True)


class TestTrajectories:
    def test_shares_are_computed_per_quarter(self):
        reac, demo = build_corpus({"STABLE": lambda q: 0.5}, quarters=8, reports_per_quarter=200)
        traj = term_trajectories(reac, demo).collect()
        assert traj["qidx"].n_unique() == 8
        # Planted at 0.5; sampling noise around it, never wildly off.
        assert traj["share"].mean() == pytest.approx(0.5, abs=0.08)


class TestDetection:
    def test_stable_term_is_not_flagged(self):
        """The false-positive case that matters: a genuinely steady term must pass through."""
        reac, demo = build_corpus({"STABLE": lambda q: 0.30}, seed=1)
        got = flags_for(detect(reac, demo), "STABLE")
        assert not got["drift_suspect"]
        assert not got["late_onset"] and not got["discontinued"] and not got["step_change"]

    def test_late_onset_term_is_flagged(self):
        """A term that does not exist before the midpoint, then becomes common."""
        reac, demo = build_corpus({"NEWTERM": lambda q: 0.0 if q < 20 else 0.30}, seed=2)
        got = flags_for(detect(reac, demo), "NEWTERM")
        assert got["late_onset"] and got["drift_suspect"]

    def test_discontinued_term_is_flagged(self):
        reac, demo = build_corpus({"OLDTERM": lambda q: 0.30 if q < 20 else 0.0}, seed=3)
        got = flags_for(detect(reac, demo), "OLDTERM")
        assert got["discontinued"] and got["drift_suspect"]

    def test_abrupt_step_is_flagged(self):
        """Present throughout, so neither onset nor discontinuation — caught by the step rule."""
        reac, demo = build_corpus({"JUMPY": lambda q: 0.01 if q < 20 else 0.40}, seed=4)
        got = flags_for(detect(reac, demo), "JUMPY")
        assert got["step_change"] and got["drift_suspect"]
        assert got["max_step_ratio"] >= STEP_RATIO

    def test_gradual_trend_is_not_flagged_as_drift(self):
        """A real epidemiological trend rises smoothly and must survive.

        This is the discrimination the detector exists for: same total change as the step case,
        spread across the window instead of landing in one quarter.
        """
        reac, demo = build_corpus({"GRADUAL": lambda q: 0.02 + 0.38 * (q / 39)}, seed=5)
        got = flags_for(detect(reac, demo), "GRADUAL")
        assert not got["drift_suspect"], "a smooth trend must not be mistaken for vocabulary drift"

    def test_rename_flags_both_sides(self):
        """A term renamed mid-window shows as one discontinued and one late-onset term."""
        reac, demo = build_corpus(
            {
                "OLDNAME": lambda q: 0.25 if q < 20 else 0.0,
                "NEWNAME": lambda q: 0.0 if q < 20 else 0.25,
            },
            seed=6,
        )
        result = detect(reac, demo)
        assert flags_for(result, "OLDNAME")["discontinued"]
        assert flags_for(result, "NEWNAME")["late_onset"]

    def test_rare_terms_are_excluded_from_testing(self):
        """Below the support floor a trajectory is noise; testing it would swamp the flag list."""
        reac, demo = build_corpus(
            {"COMMON": lambda q: 0.30, "RARE": lambda q: 0.0002},
            quarters=8,
            reports_per_quarter=200,
            seed=7,
        )
        result = detect(reac, demo)
        assert "COMMON" in result["pt"].to_list()
        assert "RARE" not in result["pt"].to_list()

    def test_empty_corpus_returns_empty(self):
        empty = pl.DataFrame(schema={"record_id": pl.String, "pt": pl.String}).lazy()
        demo = pl.DataFrame(schema={"record_id": pl.String, "fda_dt": pl.Date}).lazy()
        assert detect(empty, demo).is_empty()

    def test_absent_share_threshold_is_small_enough_to_mean_absent(self):
        assert ABSENT_SHARE < 1e-4
