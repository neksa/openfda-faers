---
license: cc-by-4.0
pretty_name: FAERS disproportionality signals, 2004-2026
annotations_creators:
  - no-annotation
language:
  - en
tags:
  - pharmacovigilance
  - drug-safety
  - adverse-events
  - signal-detection
  - fda
  - faers
task_categories:
  - tabular-classification
---

# FAERS disproportionality signals, 2004Q1–2026Q2

Drug–event reporting associations computed over **every adverse event report FDA has published
since 2004** — 20.3 million distinct cases after deduplication, spanning both the legacy LAERS and
current FAERS eras.

Produced by a fully reproducible DVC pipeline: <https://github.com/neksa/openfda-faers>

> **These are reporting associations, not evidence of harm.** A drug–event pair scoring highly here
> is reported together more often than the rest of the database predicts. That is a hypothesis to
> investigate. It is not causation, and it is not a rate — the number of people who took the drug
> without incident is unknown, so incidence cannot be computed from these data. See *Limitations*.

## What is here

| File | Rows | Description |
|---|---|---|
| `scored.parquet` | ~2.3M | One row per drug–event pair, with contingency counts and five disproportionality measures |
| `descriptive/` | small | Corpus description: growth, demographics, reporters, top terms, report structure |
| `*.json` | small | Stage metrics: deduplication, normalization, signal counts, drift, duplicates |

## Columns — `scored.parquet`

### Identity

| Column | Type | Meaning |
|---|---|---|
| `ingredient` | string | Active ingredient, uppercase. Resolved from the reported drug name; see *Drug resolution* |
| `ingredient_rxcui` | string | RxNorm identifier for the ingredient, where known. Join key to RxNorm/OMOP |
| `ingredient_curated` | bool | `false` means the name is an unmapped product string, not a resolved substance. **Filter on this** |
| `pt` | string | MedDRA Preferred Term for the reaction, as published by FDA in FAERS |

### Contingency counts

All counts are of **reports**, not of table rows: a report naming the same ingredient three times
contributes once.

| Column | Type | Meaning |
|---|---|---|
| `a` | int | Reports with this ingredient **and** this reaction |
| `b` | int | Reports with this ingredient, without this reaction |
| `c` | int | Reports with this reaction, without this ingredient |
| `d` | int | Reports with neither |
| `n_ij`, `n_i`, `n_j` | int | Co-reports, ingredient total, reaction total (`a`, `a+b`, `a+c`) |
| `expected` | float | `n_i × n_j / N` — expected co-reports under independence |

### Measures

Every measure is reported with its interval. **Screen on the lower bound, never the point
estimate**: a ratio of 8 computed from two co-reports is noise.

| Column | Meaning |
|---|---|
| `ror`, `ror025`, `ror975` | Reporting odds ratio, 95% CI (Haldane–Anscombe corrected) |
| `prr`, `prr025`, `prr975` | Proportional reporting ratio, 95% CI |
| `chi2_yates` | Yates-corrected χ², the companion to PRR in the MHRA rule |
| `p_fisher`, `q_bh` | One-sided Fisher exact p-value; Benjamini–Hochberg q-value |
| `ic`, `ic025`, `ic975` | Information component (BCPNN), Norén 2006 shrinkage |
| `ebgm`, `eb05`, `eb95` | Empirical Bayes geometric mean (MGPS/DuMouchel), posterior 5th/95th percentiles |

### Screening flags

| Column | Rule | Convention |
|---|---|---|
| `signal_ror` | `ror025 > 1` | common |
| `signal_prr_mhra` | `prr ≥ 2` and `χ² ≥ 4` and `a ≥ 3` | MHRA |
| `signal_ic` | `ic025 > 0` | WHO-UMC |
| `signal_eb05` | `eb05 ≥ 2` | FDA / MGPS — the most conservative |
| `signal_fdr` | `q_bh < 0.05` | multiplicity-controlled |

No flag is authoritative. They are published side by side because their disagreement is
informative; `signal_eb05` is very nearly a subset of the others.

## Method summary

Source: FDA's quarterly FAERS/LAERS ASCII extracts (public domain), 90 archives, each pinned by
SHA-256 in `source_manifest.json`.

1. **Harmonize** both eras into one schema. They differ substantially — LAERS is `ISR`-keyed with
   no active-ingredient field; FAERS is `primaryid`/`caseid`/`caseversion`-keyed.
2. **Deduplicate.** 16.7% of raw records are redundant: 4.0M superseded case versions and 67,838
   FDA-retracted cases. 82,134 cases span the 2012 era boundary.
3. **Resolve drug names to ingredients** (98.6% of rows) via FDA's `prod_ai`, a dictionary
   bootstrapped from the corpus, FDA's NDC directory, and NLM RxNav.
4. **Score** every pair with ≥3 co-reports, restricted to primary/secondary suspect drugs.

## Limitations

Read these before using the data.

- **No causality, no rates.** Reporting associations only. The unexposed denominator is unknown.
- **Highly-ranked pairs are frequently indication bias, not harm.** The top-scoring pairs in this
  corpus include a radioisotope tracer with "radioisotope scan abnormal" and an antipsychotic with
  a feature of the psychosis it treats. They rank highly *because* the association is real and
  near-deterministic — which is precisely why disproportionality cannot distinguish harm from
  reason-for-prescribing.
- **~24% of distinct drug labels in the signal table are product names, not substances.** These
  have `ingredient_curated = false`. Their marginals are split, which both understates the parent
  substance and inflates the fragment. Filter them for anything quantitative.
- **MedDRA version drift is flagged, not corrected.** Terms are revised twice yearly; see
  `drift_summary.json` and the `drift_suspect` flag. Correcting it requires the licensed MedDRA
  release. Temporal claims should exclude flagged terms.
- **Independent duplicates are estimated, not removed.** The same incident reported separately
  under different case ids is detected by blocked record linkage and reported in
  `duplicates_summary.json`, but not removed from these tables. Blocking makes the estimate a
  lower bound.
- **q-values are approximate.** Benjamini–Hochberg assumes independence or positive dependence;
  drug–event tests share reports and are dependent.
- **Partial dates** are completed to the start of the period, biasing them toward month-starts.

## Relationship to OFFSIDES / TWOSIDES

[nSIDES](https://nsides.io/) is the closest prior work and this is a complement, not a replacement.
OFFSIDES covers FAERS **through 2014** and reports **PRR** with a propensity-score-matched design.
This covers **2004Q1–2026Q2 across both eras, deduplicated**, and reports five measures with
intervals plus FDR. OFFSIDES' propensity-score adjustment is methodologically stronger on
confounding than the Mantel–Haenszel stratification used here; this dataset is broader in coverage
and in measures. Use both.

## Licence and terminology

This is a **non-commercial academic project**.

**The derived tables are released CC BY 4.0.** That covers what is actually this project's work:
the contingency counts, the computed measures, the resolution provenance and the code that produced
them.

**It does not, and cannot, relicense the underlying vocabularies.** A licence granted here applies
only to this project's contribution.

| Component | Origin | Terms |
|---|---|---|
| Contingency counts, measures, flags | this project | CC BY 4.0 |
| Underlying reports | FDA FAERS | public domain (US federal work) |
| `ingredient`, `ingredient_rxcui` | RxNorm (NLM) | public domain |
| `pt` | **MedDRA**, as published by FDA in the public FAERS release | see below |

MedDRA is a licensed terminology owned by IFPMA and maintained by the MSSO. The preferred terms in
`pt` reach this dataset through FDA's public-domain FAERS release rather than from the MedDRA
distribution, and FDA publishes them to everyone without a licence check. Whether onward
republication in a derived table is covered is not settled by that fact alone: the MedDRA EULA
contains no explicit carve-out for terms obtained from public regulatory sources.

**If you intend to use the `pt` column, confirm your own position.** MedDRA subscriptions are free
of charge for non-profit, non-commercial and academic organisations, and for regulatory
authorities; commercial users pay on a revenue-scaled tariff. Nothing else in this dataset carries
that constraint — the drug side is RxNorm, and the statistics are CC BY 4.0 — so a consumer without
a MedDRA position can drop one column and use everything else.

## Citation

```bibtex
@misc{goncearenco_faers_2026,
  author = {Goncearenco, Alexander},
  title  = {FAERS disproportionality signals, 2004--2026},
  year   = {2026},
  url    = {https://github.com/neksa/openfda-faers}
}
```

## References

van Puijenbroek et al. (2002) *Pharmacoepidemiol Drug Saf* 11:3–10 ·
DuMouchel (1999) *Am Stat* 53:177–190 ·
Norén et al. (2006) *Stat Med* 25:3740–3757 ·
Robins, Breslow & Greenland (1986) *Biometrics* 42:311–323
