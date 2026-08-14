# FAERS: the complete FDA adverse event corpus, as a reproducible pipeline

[![DVC](https://img.shields.io/badge/pipeline-DVC-13ADC7.svg)](https://dvc.org)
[![Report](https://img.shields.io/badge/report-Quarto-75AADB.svg)](https://neksa.github.io/openfda-faers/)
[![Tests](https://img.shields.io/badge/tests-123%20passing-brightgreen.svg)](tests/)

A DVC pipeline over **every adverse event report the FDA has published since 2004** — 20.3 million
distinct cases after deduplication, spanning both the legacy LAERS and current FAERS eras — with
disproportionality analysis, confounder adjustment, and a rendered report.

**[Read the report →](https://neksa.github.io/openfda-faers/)**

## What it answers

| | |
|---|---|
| **How much of the database is distinct?** | 16.7% of raw records are redundant: 4.0M superseded case versions and 67,838 FDA-retracted cases. 82,134 cases span the 2012 era boundary. |
| **What does a report contain?** | Mean 3.18 drugs and 2.98 reactions, median 1 drug — with a tail reaching 171 drugs on one report. |
| **Who reports?** | Consumers (8.9M) now file more than physicians (4.5M). |
| **Which drug–event pairs report disproportionately?** | 2.28M pairs scored; 822,472 flagged by EB05 ≥ 2, the most conservative rule. |
| **Do signals survive confounder adjustment?** | 98.5–100% survive Mantel–Haenszel adjustment, but Breslow–Day rejects homogeneity for 50–76% — the confounders modify signals rather than create them. |
| **How much of the corpus supports a causal reading?** | 0.147% of drug records are a suspect drug with both positive dechallenge and positive rechallenge. |
| **How much is vocabulary drift rather than epidemiology?** | 42% of tested reaction terms look like MedDRA revisions, touching 24% of reaction mentions. Flagged, not corrected. |
| **How many reports are the same incident twice?** | 9.1% of linkable records, in 287k clusters — beyond the 16.7% version-based redundancy. Estimated, not removed. |

## Quick start

```bash
uv sync
dvc repro
```

No credentials, no cloud storage, no API key. FDA hosts the source archives; `dvc.lock` pins every
intermediate by hash. Full run is ~25 minutes on a 10-core laptop, dominated by the 3.5 GB download.

```bash
faers verify      # check archives against committed SHA-256 checksums
pytest            # statistical validation suite
dvc metrics show  # corpus, resolution and signal counts
```

## Pipeline

```
download → harmonize → dedup ─┐             ┌→ describe ──┐
                              ├→ normalize ─┼→ signals ───┤
brands ───────────────────────┘             ├→ stratify ──┼→ figures → report
                                            ├→ duplicates ┤
                                            └→ drift ─────┘
```

| Stage | What it does |
|---|---|
| `download` | Fetch 90 quarterly archives, record SHA-256 per file |
| `harmonize` | Parse both eras into one canonical schema |
| `dedup` | One record per case; drop retracted and superseded |
| `brands` | Brand→ingredient index from FDA's NDC directory |
| `normalize` | Resolve drug names to ingredients via prod_ai, corpus, NDC, RxNav (98.6%) |
| `describe` | Growth, demographics, reporters, top terms, report structure |
| `signals` | ROR, PRR, IC/BCPNN, EBGM/MGPS with Fisher + BH-FDR |
| `stratify` | Mantel–Haenszel by sex, age band, calendar period |
| `duplicates` | Record linkage for independent duplicate reports (estimated, not removed) |
| `drift` | Flags MedDRA vocabulary change without a licensed dictionary |
| `figures` | Vega-Lite specs + PNG |
| `report` | Quarto site |

Every choice that moves a number lives in [`params.yaml`](params.yaml) — role filter, minimum
support, stratification variables — so a sensitivity analysis is a params edit, not a code change.

## Why FAERS ASCII rather than the openFDA API

The previous version of this repository was a notebook built on the openFDA web API. That API caps
pagination at 25,000 records per query, so its record-level analysis ran on a 10,000-report sample,
and it exposes no case-version fields, so deduplication was impossible.

| Source | Coverage | Size |
|---|---|---|
| openFDA API | 20.7M reports | rate-limited, `skip` capped at 25k |
| openFDA bulk JSON | 20.7M reports | **113 GB**, 1,767 partitions |
| **FAERS/LAERS ASCII** | 90 quarters, 2004Q1–2026Q2 | **3.5 GB** |

The ASCII dumps are 34× smaller for the same reports, and carry `caseid`/`caseversion` — the only
clean basis for deduplication.

## Statistical approach

Signals are screened on **lower confidence or credible bounds** (ROR₀₂₅, PRR₀₂₅, IC₀₂₅, EB05), never
point estimates. Four rules are reported side by side because none is authoritative:

- **ROR** with 95% CI, Haldane–Anscombe corrected
- **PRR** with the MHRA χ² ≥ 4 criterion
- **IC/BCPNN** (Norén 2006 shrinkage), WHO-UMC's `IC₀₂₅ > 0`
- **EBGM/MGPS** (DuMouchel 1999) gamma-Poisson shrinkage with EB05/EB95
- **Fisher exact** (vectorized hypergeometric) with Benjamini–Hochberg FDR

Validated against nine established label warnings — Metformin–Lactic acidosis, Clozapine–
Agranulocytosis, Amiodarone–Pulmonary toxicity and others — **all nine flagged by all five rules**,
with implausible pairings correctly suppressed (Aspirin–Alopecia, EB05 = 0.19). Mantel–Haenszel and
Breslow–Day are calibrated against simulated data with a known common odds ratio: MH recovers
OR = 3.01 against a true 3, and Breslow–Day shows 6.8% type-I error against a nominal 5%.

### Corrections to the previous analysis

- **PRR variance.** The notebook used `1/a − 1/(a+b) + 1/c + 1/(c+d)`; the final term is a
  subtraction. [`tests/test_disproportionality.py`](tests/test_disproportionality.py) pins this.
- **Dechallenge/rechallenge prevalence.** Estimated at ~2% from marginal counts treated as
  independent; the joint distribution gives 0.147%.
- **Reaction trends.** Raw yearly counts conflate a term becoming more common with the database
  growing. Reported here as a share of each year's reports.
- **Duplicates.** The repetitive clustergram block the notebook attributed to "duplicates or other
  artifacts" was 4.0M superseded case versions.

## What these data cannot show

Reports do not link a drug to a reaction — that field does not exist — and no report is validated.
Disproportionality measures reporting associations, not causation or incidence: the number of people
exposed without an event is unknown.

The highest-scoring pairs in the whole corpus illustrate this. Technetium–"Radioisotope scan
abnormal" and Zuclopenthixol–Anosognosia rank at the top *because* the association is real and
near-deterministic — the tracer is given in order to perform the scan; anosognosia is a feature of
the treated psychosis. Strong disproportionality distinguishes association from noise, not harm from
indication.

**Drug names are resolved to ingredients at 98.6%**, via FDA's `prod_ai`, a dictionary bootstrapped
from the corpus, FDA's NDC directory and NLM's RxNav. The residue — discontinued, foreign and
misspelled products — keeps its reported name and is flagged `ingredient_curated = false` in the
published table. 23.5% of distinct drug labels in the signal table are still product names rather
than substances, which splits marginals; treat those signals as provisional.

Further limitations — MedDRA version drift, absent drug-class hierarchy,
approximate q-values under dependence — are stated in the
[report](https://neksa.github.io/openfda-faers/#limitations).

## Data

FDA Adverse Event Reporting System quarterly extracts, 2004Q1–2026Q2, from
[fis.fda.gov](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html). Public domain.

FDA republishes quarterly extracts as snapshots and can revise them in place, so
`results/source_manifest.json` pins a SHA-256 per archive and `faers verify` reports drift rather
than letting it silently change results.

## References

van Puijenbroek et al. (2002) *Pharmacoepidemiol Drug Saf* 11:3–10 · DuMouchel (1999)
*Am Stat* 53:177–190 · Norén et al. (2006) *Stat Med* 25:3740–3757 · Robins, Breslow & Greenland
(1986) *Biometrics* 42:311–323
