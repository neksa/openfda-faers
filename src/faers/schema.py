"""Canonical schema and the two-era column mappings.

Column sets drift *within* eras as well as between them (``reporter_country`` appears partway
through LAERS; ``caseversion`` only exists in FAERS), so every mapping is applied tolerantly: a
source column that is absent in a given quarter yields nulls rather than an error. What must never
be tolerated silently is a *key* column going missing, which :func:`require` enforces.

Join semantics, easy to get wrong: child tables (DRUG/REAC/...) link to DEMO by the **record**
identifier -- ``primaryid`` in FAERS, ``ISR`` in LAERS -- not by the case identifier. A case
accumulates several records over time as follow-ups arrive; collapsing to one record per case is
the job of the dedup stage, not this one.
"""

from __future__ import annotations

import polars as pl

from .sources import Era

# --------------------------------------------------------------------------------------------
# Column mappings: canonical_name -> source column name, per era.
# --------------------------------------------------------------------------------------------

# A mapping value may be a single source column or a tuple of candidates tried in order.
# FDA renamed columns mid-era without changing the archive format: ``gndr_cod`` became ``sex`` in
# 2014Q3, and ``occr_country`` appears only from 2014Q3. Testing 2012Q4 with a single-name mapping
# silently produced 100% "UNK" sex, so candidates are matched positionally, first present wins.

DEMO_MAP: dict[Era, dict[str, str | tuple[str, ...]]] = {
    Era.FAERS: {
        "record_id": "primaryid",
        "case_id": "caseid",
        "case_version": "caseversion",
        "i_f_code": "i_f_code",
        "event_dt": "event_dt",
        "mfr_dt": "mfr_dt",
        "init_fda_dt": "init_fda_dt",
        "fda_dt": "fda_dt",
        "rept_dt": "rept_dt",
        "rept_cod": "rept_cod",
        "mfr_sndr": "mfr_sndr",
        "age": "age",
        "age_cod": "age_cod",
        "age_grp": "age_grp",
        # Renamed from gndr_cod in 2014Q3; both spellings occur across the FAERS era.
        "sex": ("sex", "gndr_cod"),
        "wt": "wt",
        "wt_cod": "wt_cod",
        "occp_cod": "occp_cod",
        "reporter_country": "reporter_country",
        "occr_country": "occr_country",
        "e_sub": "e_sub",
        "to_mfr": "to_mfr",
    },
    Era.LAERS: {
        "record_id": "ISR",
        "case_id": "CASE",
        # LAERS has no caseversion; FOLL_SEQ is the follow-up counter and is derived in harmonize.
        "foll_seq": "FOLL_SEQ",
        "i_f_code": "I_F_COD",
        "event_dt": "EVENT_DT",
        "mfr_dt": "MFR_DT",
        "fda_dt": "FDA_DT",
        "rept_dt": "REPT_DT",
        "rept_cod": "REPT_COD",
        "mfr_sndr": "MFR_SNDR",
        "age": "AGE",
        "age_cod": "AGE_COD",
        "sex": "GNDR_COD",
        "wt": "WT",
        "wt_cod": "WT_COD",
        "occp_cod": "OCCP_COD",
        "reporter_country": "REPORTER_COUNTRY",
        "e_sub": "E_SUB",
        "to_mfr": "TO_MFR",
        "death_dt": "DEATH_DT",
    },
}

DRUG_MAP: dict[Era, dict[str, str]] = {
    Era.FAERS: {
        "record_id": "primaryid",
        "drug_seq": "drug_seq",
        "role_cod": "role_cod",
        "drugname": "drugname",
        "prod_ai": "prod_ai",
        "route": "route",
        "dechal": "dechal",
        "rechal": "rechal",
        "nda_num": "nda_num",
        "dose_amt": "dose_amt",
        "dose_unit": "dose_unit",
        "dose_form": "dose_form",
        "dose_freq": "dose_freq",
        "val_vbm": "val_vbm",
    },
    Era.LAERS: {
        "record_id": "ISR",
        "drug_seq": "DRUG_SEQ",
        "role_cod": "ROLE_COD",
        "drugname": "DRUGNAME",
        # No PROD_AI in LAERS: active ingredient must be resolved from free text downstream.
        "route": "ROUTE",
        "dechal": "DECHAL",
        "rechal": "RECHAL",
        "nda_num": "NDA_NUM",
        "val_vbm": "VAL_VBM",
    },
}

REAC_MAP: dict[Era, dict[str, str]] = {
    Era.FAERS: {"record_id": "primaryid", "pt": "pt", "drug_rec_act": "drug_rec_act"},
    Era.LAERS: {"record_id": "ISR", "pt": "PT"},
}

OUTC_MAP: dict[Era, dict[str, str]] = {
    Era.FAERS: {"record_id": "primaryid", "outc_cod": "outc_cod"},
    Era.LAERS: {"record_id": "ISR", "outc_cod": "OUTC_COD"},
}

INDI_MAP: dict[Era, dict[str, str]] = {
    Era.FAERS: {"record_id": "primaryid", "drug_seq": "indi_drug_seq", "indi_pt": "indi_pt"},
    Era.LAERS: {"record_id": "ISR", "drug_seq": "DRUG_SEQ", "indi_pt": "INDI_PT"},
}

THER_MAP: dict[Era, dict[str, str]] = {
    Era.FAERS: {
        "record_id": "primaryid",
        "drug_seq": "dsg_drug_seq",
        "start_dt": "start_dt",
        "end_dt": "end_dt",
        "dur": "dur",
        "dur_cod": "dur_cod",
    },
    Era.LAERS: {
        "record_id": "ISR",
        "drug_seq": "DRUG_SEQ",
        "start_dt": "START_DT",
        "end_dt": "END_DT",
        "dur": "DUR",
        "dur_cod": "DUR_COD",
    },
}

RPSR_MAP: dict[Era, dict[str, str]] = {
    Era.FAERS: {"record_id": "primaryid", "rpsr_cod": "rpsr_cod"},
    Era.LAERS: {"record_id": "ISR", "rpsr_cod": "RPSR_COD"},
}

MAPS = {
    "DEMO": DEMO_MAP,
    "DRUG": DRUG_MAP,
    "REAC": REAC_MAP,
    "OUTC": OUTC_MAP,
    "INDI": INDI_MAP,
    "THER": THER_MAP,
    "RPSR": RPSR_MAP,
}

#: Columns whose absence is a hard error -- everything downstream joins on them.
REQUIRED = {
    "DEMO": ("record_id", "case_id"),
    "DRUG": ("record_id", "drug_seq"),
    "REAC": ("record_id", "pt"),
    "OUTC": ("record_id", "outc_cod"),
    "INDI": ("record_id", "drug_seq"),
    "THER": ("record_id", "drug_seq"),
    "RPSR": ("record_id", "rpsr_cod"),
}


def canonical_columns(table: str) -> list[str]:
    """The full canonical column set for a table, unioned across both eras.

    Emitting the same columns regardless of era is what lets 91 quarterly parquet files be scanned
    as one dataset. Without it, LAERS files would simply lack ``prod_ai`` and any concatenation
    would fail on a schema mismatch.
    """
    seen: dict[str, None] = {}
    for era_map in MAPS[table].values():
        for dst in era_map:
            if dst != "foll_seq":  # era-internal helper, consumed during harmonization
                seen[dst] = None
    return list(seen)


def apply_map(lf: pl.LazyFrame, table: str, era: Era) -> pl.LazyFrame:
    """Rename source columns to canonical names, filling absent ones with nulls.

    ``foll_seq`` is retained when present because LAERS harmonization derives ``case_version``
    from it.
    """
    mapping = MAPS[table][era]
    present = set(lf.collect_schema().names())
    lowered = {c.lower(): c for c in present}

    def resolve(src: str | tuple[str, ...]) -> str | None:
        for cand in (src,) if isinstance(src, str) else src:
            if cand in present:
                return cand
            if cand.lower() in lowered:  # header case drifts between quarters
                return lowered[cand.lower()]
        return None

    wanted = list(canonical_columns(table))
    if "foll_seq" in mapping:
        wanted.append("foll_seq")

    exprs = []
    for dst in wanted:
        src = mapping.get(dst)
        col = resolve(src) if src is not None else None
        exprs.append(
            pl.col(col).cast(pl.String).alias(dst)
            if col is not None
            else pl.lit(None, pl.String).alias(dst)
        )
    return lf.select(exprs)


def require(lf: pl.LazyFrame, table: str, quarter_label: str) -> None:
    """Fail loudly when a join key did not survive the mapping."""
    names = set(lf.collect_schema().names())
    missing = [c for c in REQUIRED[table] if c not in names]
    if missing:
        raise ValueError(f"{quarter_label} {table}: missing required column(s) {missing}")


# --------------------------------------------------------------------------------------------
# Value harmonization
# --------------------------------------------------------------------------------------------

#: FAERS uses F/M/UNK/NS; LAERS GNDR_COD adds blanks. Everything else becomes UNK.
SEX_MAP = {"F": "F", "M": "M", "UNK": "UNK", "NS": "UNK", "": "UNK"}

#: Multipliers converting the age unit code to years.
AGE_TO_YEARS = {
    "DEC": 10.0,
    "YR": 1.0,
    "MON": 1.0 / 12,
    "WK": 1.0 / 52.1775,
    "DY": 1.0 / 365.25,
    "DAY": 1.0 / 365.25,
    "HR": 1.0 / 8766.0,
}

#: Reported ages above this are treated as data entry errors rather than real observations.
MAX_PLAUSIBLE_AGE = 120.0

#: Report source qualification (primary reporter occupation).
OCCP_LABELS = {
    "MD": "Physician",
    "PH": "Pharmacist",
    "OT": "Other health professional",
    "LW": "Lawyer",
    "CN": "Consumer",
    "HP": "Health professional",
    "RN": "Registered nurse",
}

#: Drug role in the report. PS/SS are the "suspect" roles used for signal detection.
ROLE_LABELS = {
    "PS": "Primary suspect",
    "SS": "Secondary suspect",
    "C": "Concomitant",
    "I": "Interacting",
    "DN": "Drug not administered",
}

SUSPECT_ROLES = ("PS", "SS")

#: Outcome codes attached to a report.
OUTCOME_LABELS = {
    "DE": "Death",
    "LT": "Life-threatening",
    "HO": "Hospitalization",
    "DS": "Disability",
    "CA": "Congenital anomaly",
    "RI": "Required intervention",
    "OT": "Other serious",
}

#: Dechallenge / rechallenge outcome codes.
CHALLENGE_LABELS = {
    "Y": "Positive",
    "N": "Negative",
    "U": "Unknown",
    "D": "Does not apply",
}

#: Report type.
REPT_COD_LABELS = {
    "EXP": "Expedited",
    "PER": "Periodic (non-expedited)",
    "DIR": "Direct",
    "30DAY": "30-day",
    "SDY": "Study",
    "OTH": "Other",
}
