"""Drug-drug similarity from adverse reaction profiles.

Two drugs that provoke the same pattern of reported reactions may share a mechanism or a target.
That is the premise the 2020 notebook explored with UMAP and DBSCAN, and it is the one section of
that notebook this rewrite had not reproduced.

Three deliberate departures from the notebook's approach:

**Similarity on the information component, not raw counts.** Raw co-report counts make every drug
look like every other drug, because both profiles are dominated by the same handful of ubiquitous
terms -- nausea, fatigue, death. The IC is already normalized against what the corpus would predict,
so a drug's profile becomes what is *distinctive* about it rather than what is merely common.

**Cosine similarity and hierarchical clustering rather than UMAP into DBSCAN.** UMAP is a nonlinear
projection tuned by hyperparameters that change the answer; clustering its output means clustering
an artefact of those hyperparameters, and the notebook's cluster labels were not reproducible
across runs for that reason. Cosine distance on the profile matrix is computed directly, is
deterministic, and needs no embedding step at all. It also drops the umap-learn dependency, which
pins an old llvmlite that will not build on current Python.

**Validated against known pharmacology.** The notebook clustered, described two clusters, and left
interpretation as "out of scope". Here the output is checked against drug classes whose members
should resemble each other -- statins, ACE inhibitors, TNF inhibitors, SSRIs. If those do not come
out as mutual near-neighbours, the similarity measure is not working and nothing further should be
read into it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster import hierarchy
from scipy.sparse import csr_matrix

#: Drugs need a reasonably rich profile before a similarity is meaningful.
MIN_REACTIONS_PER_DRUG = 20

#: Cap on drugs entering the analysis, taken by report volume. The full matrix is ~8.7k drugs and
#: would be fine, but the pairwise distance matrix grows quadratically and most of the tail carries
#: too little evidence to place confidently.
MAX_DRUGS = 2000

#: Drug classes whose members should resemble each other if the measure works. Names are matched as
#: prefixes against resolved ingredients, since FAERS records salts ("ATORVASTATIN CALCIUM").
VALIDATION_CLASSES: dict[str, tuple[str, ...]] = {
    "statins": ("ATORVASTATIN", "SIMVASTATIN", "ROSUVASTATIN", "PRAVASTATIN", "LOVASTATIN"),
    "ace_inhibitors": ("LISINOPRIL", "ENALAPRIL", "RAMIPRIL", "CAPTOPRIL", "QUINAPRIL"),
    "tnf_inhibitors": ("ADALIMUMAB", "ETANERCEPT", "INFLIXIMAB", "GOLIMUMAB", "CERTOLIZUMAB"),
    "ssris": ("FLUOXETINE", "SERTRALINE", "PAROXETINE", "CITALOPRAM", "ESCITALOPRAM"),
    "bisphosphonates": ("ALENDRONATE", "RISEDRONATE", "IBANDRONATE", "ZOLEDRONIC"),
}


def build_profiles(
    scored: pl.LazyFrame,
    min_reactions: int = MIN_REACTIONS_PER_DRUG,
    max_drugs: int = MAX_DRUGS,
) -> tuple[csr_matrix, list[str], list[str]]:
    """Drug x reaction matrix weighted by the information component.

    Only curated ingredients are included: an unmapped product string has a profile split with the
    substance it belongs to, so its similarities would be meaningless in both directions.
    """
    df = (
        scored.filter(pl.col("ingredient_curated") & (pl.col("ic") > 0))
        .select("ingredient", "pt", "ic")
        .collect()
    )
    if df.is_empty():
        return csr_matrix((0, 0)), [], []

    counts = df.group_by("ingredient").agg(pl.len().alias("n"))
    keep = (
        counts.filter(pl.col("n") >= min_reactions)
        .sort(["n", "ingredient"], descending=[True, False])
        .head(max_drugs)["ingredient"]
        .to_list()
    )
    df = df.filter(pl.col("ingredient").is_in(keep))

    drugs = sorted(df["ingredient"].unique().to_list())
    reactions = sorted(df["pt"].unique().to_list())
    drug_ix = {d: i for i, d in enumerate(drugs)}
    reac_ix = {r: i for i, r in enumerate(reactions)}

    rows = np.fromiter((drug_ix[d] for d in df["ingredient"]), dtype=np.int32, count=df.height)
    cols = np.fromiter((reac_ix[r] for r in df["pt"]), dtype=np.int32, count=df.height)
    vals = df["ic"].to_numpy().astype(np.float64)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(drugs), len(reactions)))
    return matrix, drugs, reactions


def cosine_similarity(matrix: csr_matrix) -> np.ndarray:
    """Dense cosine similarity between rows, with zero-norm rows handled."""
    dense = matrix.toarray()
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = dense / norms
    sim = unit @ unit.T
    np.fill_diagonal(sim, 1.0)
    return np.clip(sim, -1.0, 1.0)


def nearest_neighbours(sim: np.ndarray, drugs: list[str], k: int = 5) -> pl.DataFrame:
    """The k most similar drugs to each drug."""
    rows = []
    for i, drug in enumerate(drugs):
        order = np.argsort(-sim[i])
        rank = 0
        for j in order:
            if j == i:
                continue
            rows.append(
                {
                    "ingredient": drug,
                    "neighbour": drugs[j],
                    "rank": rank + 1,
                    "cosine": float(sim[i, j]),
                }
            )
            rank += 1
            if rank >= k:
                break
    return pl.DataFrame(rows)


def cluster_drugs(sim: np.ndarray, drugs: list[str], n_clusters: int = 40) -> pl.DataFrame:
    """Average-linkage hierarchical clustering on cosine distance.

    Average linkage rather than Ward: Ward assumes Euclidean geometry, which cosine distance does
    not provide, and applying it anyway is a common and quiet error.
    """
    if len(drugs) < 2:
        return pl.DataFrame({"ingredient": drugs, "cluster": [0] * len(drugs)})
    distance = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    condensed = _condense(distance)
    linkage = hierarchy.linkage(condensed, method="average")
    labels = hierarchy.fcluster(linkage, t=min(n_clusters, len(drugs)), criterion="maxclust")
    return pl.DataFrame({"ingredient": drugs, "cluster": labels.astype(int)})


def _condense(square: np.ndarray) -> np.ndarray:
    """Upper triangle of a symmetric distance matrix, in scipy's condensed form."""
    n = square.shape[0]
    iu = np.triu_indices(n, k=1)
    return square[iu]


def validate_against_known_classes(
    sim: np.ndarray, drugs: list[str], classes: dict[str, tuple[str, ...]] | None = None
) -> pl.DataFrame:
    """Do members of a known drug class resemble each other more than random pairs?

    For each class, compares mean within-class similarity against the mean similarity of all pairs.
    A measure that captures pharmacology should separate these clearly; one that does not is
    reporting noise, and the clusters below it should not be interpreted.
    """
    classes = classes or VALIDATION_CLASSES
    index = {d: i for i, d in enumerate(drugs)}
    overall = float(sim[np.triu_indices(len(drugs), k=1)].mean()) if len(drugs) > 1 else 0.0

    rows = []
    for name, prefixes in classes.items():
        members = [
            d for d in drugs if any(d.startswith(p) for p in prefixes)
        ]
        if len(members) < 2:
            rows.append(
                {
                    "drug_class": name,
                    "members_found": len(members),
                    "mean_within_class": None,
                    "mean_overall": overall,
                    "ratio": None,
                }
            )
            continue
        idx = [index[m] for m in members]
        sub = sim[np.ix_(idx, idx)]
        within = float(sub[np.triu_indices(len(idx), k=1)].mean())
        rows.append(
            {
                "drug_class": name,
                "members_found": len(members),
                "mean_within_class": within,
                "mean_overall": overall,
                "ratio": within / overall if overall else None,
            }
        )
    return pl.DataFrame(rows)


def write(scored_path: Path, out_dir: Path, summary_path: Path, n_clusters: int = 40) -> dict:
    """Build profiles, compute similarity, cluster, validate, and write everything."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, drugs, reactions = build_profiles(pl.scan_parquet(str(scored_path)))
    if not drugs:
        Path(summary_path).write_text(json.dumps({"drugs": 0}, indent=2) + "\n")
        return {"drugs": 0}

    sim = cosine_similarity(matrix)

    neighbours = nearest_neighbours(sim, drugs)
    neighbours.write_parquet(out_dir / "nearest_neighbours.parquet", compression="zstd")

    clusters = cluster_drugs(sim, drugs, n_clusters=n_clusters)
    clusters.write_parquet(out_dir / "clusters.parquet", compression="zstd")

    validation = validate_against_known_classes(sim, drugs)
    validation.write_parquet(out_dir / "class_validation.parquet", compression="zstd")

    scored_ratios = [r for r in validation["ratio"].to_list() if r is not None]
    summary = {
        "drugs": len(drugs),
        "reactions": len(reactions),
        "profile_density": round(matrix.nnz / (len(drugs) * len(reactions)), 5),
        "clusters": int(clusters["cluster"].n_unique()),
        "largest_cluster": int(clusters.group_by("cluster").len()["len"].max()),
        "validation": {
            r["drug_class"]: {
                "members": r["members_found"],
                "within_class_similarity": (
                    round(r["mean_within_class"], 4)
                    if r["mean_within_class"] is not None
                    else None
                ),
                "vs_overall": round(r["ratio"], 2) if r["ratio"] is not None else None,
            }
            for r in validation.to_dicts()
        },
        "mean_validation_ratio": (
            round(sum(scored_ratios) / len(scored_ratios), 2) if scored_ratios else None
        ),
    }
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
    return summary
