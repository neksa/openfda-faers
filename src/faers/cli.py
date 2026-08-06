"""Command line entry point. Every DVC stage maps to one subcommand."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import typer
import yaml

from . import dedup as dedup_mod
from . import normalize as normalize_mod
from . import signals as signals_mod
from .download import download_all, verify_against_manifest
from .harmonize import harmonize_quarter
from .sources import parse_quarter, quarters

app = typer.Typer(add_completion=False, help=__doc__)

PARAMS = Path("params.yaml")


def load_params() -> dict:
    return yaml.safe_load(PARAMS.read_text())


def _echo(msg: str) -> None:
    typer.echo(msg, err=False)


@app.command()
def download(
    raw_dir: Path = Path("data/raw"),
    manifest: Path = Path("results/source_manifest.json"),
    workers: int = 6,
) -> None:
    """Fetch every quarterly archive and write the checksum manifest."""
    p = load_params()
    qs = quarters(p["coverage"]["start"], p["coverage"]["end"])
    _echo(f"downloading {len(qs)} quarters {qs[0]}..{qs[-1]}")
    m = download_all(qs, raw_dir, manifest, workers=workers)
    _echo(f"{m['n_quarters']} archives, {m['total_bytes']:,} bytes")


@app.command()
def verify(
    raw_dir: Path = Path("data/raw"),
    manifest: Path = Path("results/source_manifest.json"),
) -> None:
    """Check on-disk archives against the committed checksums.

    FDA republishes quarterly extracts as snapshots and can revise them in place, so a drift here
    is expected news rather than a corruption alarm -- but it must be visible, because it changes
    every number downstream.
    """
    drifted = verify_against_manifest(manifest, raw_dir)
    if drifted:
        _echo(f"CHECKSUM DRIFT in {len(drifted)} archive(s): {', '.join(drifted)}")
        raise typer.Exit(1)
    _echo("all archives match the manifest")


def _harmonize_one(args: tuple[str, str, str]) -> tuple[str, dict]:
    """Module-level so it is picklable under the spawn start method."""
    label, raw_dir, out_dir = args
    q = parse_quarter(label)
    return label, harmonize_quarter(Path(raw_dir) / f"{label}.zip", q, Path(out_dir))


@app.command()
def harmonize(
    raw_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/harmonized"),
    stats_path: Path = Path("results/harmonize_stats.json"),
    workers: int = 8,
) -> None:
    """Parse every archive into the canonical two-era schema."""
    p = load_params()
    qs = quarters(p["coverage"]["start"], p["coverage"]["end"])
    jobs = [(q.label, str(raw_dir), str(out_dir)) for q in qs]

    t0 = time.time()
    per_quarter: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_harmonize_one, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            label, st = fut.result()
            per_quarter[label] = st
            if i % 15 == 0 or i == len(jobs):
                _echo(f"  {i}/{len(jobs)} quarters")

    totals: dict[str, int] = {}
    for st in per_quarter.values():
        for k, v in st.items():
            totals[k] = totals.get(k, 0) + v

    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stats_path).write_text(
        json.dumps({"totals": totals, "per_quarter": dict(sorted(per_quarter.items()))}, indent=2)
        + "\n"
    )
    _echo(f"harmonized {len(per_quarter)} quarters in {time.time() - t0:.0f}s")
    _echo(f"row totals: {totals}")


@app.command()
def dedup(
    harmonized_dir: Path = Path("data/harmonized"),
    out_dir: Path = Path("data/curated"),
    stats_path: Path = Path("results/dedup_stats.json"),
) -> None:
    """Collapse the corpus to one record per case and drop FDA-retracted cases."""
    stats = dedup_mod.dedup(harmonized_dir, out_dir, stats_path)
    _echo(json.dumps(stats, indent=2))
    counts = dedup_mod.filter_children(harmonized_dir, out_dir, out_dir)
    _echo(f"child tables: {counts}")


@app.command()
def normalize(
    curated_dir: Path = Path("data/curated"),
    out_path: Path = Path("data/curated/drug_ingredients.parquet"),
    stats_path: Path = Path("results/normalize_stats.json"),
) -> None:
    """Resolve reported drug names to active ingredients."""
    p = load_params()
    stats = normalize_mod.normalize_drugs(
        curated_dir, out_path, stats_path,
        min_support=p["normalize"]["dictionary_min_support"],
    )
    _echo(json.dumps(stats, indent=2))


@app.command()
def describe(
    curated_dir: Path = Path("data/curated"),
    out_dir: Path = Path("results/descriptive"),
) -> None:
    """Answer the corpus-description questions."""
    from .stats import descriptive

    p = load_params()
    counts = descriptive.write_all(curated_dir, out_dir, p["descriptive"]["trend_terms"])
    _echo(json.dumps(counts, indent=2))


@app.command()
def signals(
    curated_dir: Path = Path("data/curated"),
    out_dir: Path = Path("results/signals"),
) -> None:
    """Build drug-event contingency tables and score them with every measure."""
    p = load_params()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roles = tuple(p["signals"]["roles"]) if p["signals"]["roles"] else None
    con = signals_mod.connect(
        memory_limit=p["signals"]["memory_limit"], temp_dir=p["signals"]["temp_dir"]
    )

    drug_p = Path(curated_dir) / "drug_ingredients.parquet"
    reac_p = Path(curated_dir) / "reac.parquet"
    demo_p = Path(curated_dir) / "demo.parquet"

    fan = signals_mod.measure_fanout(con, drug_p, reac_p, roles)
    _echo(f"join shape: {json.dumps(fan, indent=2)}")
    (out_dir / "fanout.json").write_text(json.dumps(fan, indent=2) + "\n")

    counts_p = out_dir / "counts.parquet"
    cohort = signals_mod.build_counts(
        con, drug_p, reac_p, demo_p, counts_p,
        roles=roles, min_count=p["signals"]["min_count"],
    )
    _echo(f"counts: {json.dumps(cohort, indent=2)}")

    summary = signals_mod.score(
        counts_p, out_dir / "scored.parquet", cohort["n_total_reports"],
        params_path=out_dir / "summary.json",
        fit_sample=p["signals"]["gamma_poisson_fit_sample"],
        random_state=p["signals"]["random_state"],
    )
    _echo(json.dumps(summary["signals"], indent=2))


@app.command()
def stratify(
    curated_dir: Path = Path("data/curated"),
    signals_dir: Path = Path("results/signals"),
    out_dir: Path = Path("results/stratified"),
    summary_path: Path = Path("results/stratify_summary.json"),
    max_pairs: int = 20000,
) -> None:
    """Re-test the strongest signals with Mantel-Haenszel adjustment."""
    import polars as pl

    from .stats import stratified as strat

    p = load_params()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curated_dir = Path(curated_dir)

    # Only pairs that signal unadjusted are worth re-testing: adjustment can remove a signal but
    # cannot create one from a pair that showed no disproportionality to begin with.
    #
    # The candidates are a *random sample* of signalling pairs, not the strongest ones. Taking the
    # top pairs by EB05 would estimate a survival rate of essentially 100% by construction -- the
    # most extreme signals are exactly those no adjustment can remove -- which says nothing about
    # how the signal set as a whole behaves under confounder control.
    scored = pl.read_parquet(Path(signals_dir) / "scored.parquet")
    signalling = scored.filter(pl.col("signal_eb05"))
    candidates = signalling.sample(
        n=min(max_pairs, signalling.height), seed=p["signals"]["random_state"]
    ).select("ingredient", "pt")
    _echo(
        f"adjusting a random sample of {candidates.height:,} of "
        f"{signalling.height:,} signalling pairs"
    )

    demo = pl.scan_parquet(str(curated_dir / "demo.parquet"))
    roles = tuple(p["signals"]["roles"]) if p["signals"]["roles"] else None
    con = signals_mod.connect(
        memory_limit=p["signals"]["memory_limit"], temp_dir=p["signals"]["temp_dir"]
    )

    survival_rows = []
    for variable in p["stratify"]["variables"]:
        strata = strat.build_strata(demo, variable, p["stratify"]["calendar_period_years"])
        strata_path = out_dir / f"strata_{variable}.parquet"
        strata.collect().write_parquet(strata_path, compression="zstd")

        counts = strat.stratified_counts(
            con,
            curated_dir / "drug_ingredients.parquet",
            curated_dir / "reac.parquet",
            strata_path,
            variable,
            candidates,
            roles,
        )
        adjusted = strat.adjust(counts, p["stratify"]["min_stratum_count"])
        if adjusted.is_empty():
            _echo(f"  {variable}: no estimable pairs")
            continue
        adjusted.write_parquet(out_dir / f"adjusted_{variable}.parquet", compression="zstd")

        row = {
            "variable": variable,
            "tested": adjusted.height,
            "surviving": int(adjusted["signal_adjusted"].sum()),
            "heterogeneous": int(adjusted["heterogeneous"].sum()),
        }
        row["survival_rate"] = round(row["surviving"] / max(row["tested"], 1), 4)
        survival_rows.append(row)
        _echo(f"  {variable}: {row['surviving']:,}/{row['tested']:,} survive "
              f"({row['survival_rate']:.1%}), {row['heterogeneous']:,} heterogeneous")

    survival = pl.DataFrame(survival_rows)
    survival.write_parquet(out_dir / "survival.parquet", compression="zstd")
    Path(summary_path).write_text(json.dumps({"by_variable": survival_rows}, indent=2) + "\n")


@app.command()
def figures(
    results_dir: Path = Path("results"),
    out_dir: Path = Path("figures"),
) -> None:
    """Render every figure as a Vega-Lite spec plus a PNG."""
    from .plots import render_all

    names = render_all(results_dir, out_dir)
    _echo(f"wrote {len(names)} figures: {', '.join(n.replace('.vl.json', '') for n in names)}")


@app.command()
def report(report_dir: Path = Path("report")) -> None:
    """Render the Quarto report."""
    import subprocess

    proc = subprocess.run(
        ["quarto", "render", "."], cwd=report_dir, capture_output=True, text=True
    )
    if proc.returncode != 0:
        _echo(proc.stdout[-4000:])
        _echo(proc.stderr[-4000:])
        raise typer.Exit(proc.returncode)
    _echo("report rendered")


@app.command()
def publish(dry_run: bool = False) -> None:
    """Upload derived tables to a Hugging Face dataset repo.

    Deliberately excluded from the default `dvc repro`: it is a side effect on an external service
    and must be an explicit act, not something a pipeline rebuild does silently.
    """
    import os

    from huggingface_hub import HfApi

    p = load_params()
    repo = p["publish"]["hf_repo"]
    if not repo:
        _echo("publish.hf_repo is not set in params.yaml")
        raise typer.Exit(1)
    token = os.environ.get("HF_TOKEN")
    if not token and not dry_run:
        _echo("HF_TOKEN is not set in the environment")
        raise typer.Exit(1)

    paths: list[Path] = []
    for pattern in p["publish"]["include"]:
        paths.extend(sorted(Path().glob(pattern)))
    total = sum(f.stat().st_size for f in paths if f.is_file())
    _echo(f"{len(paths)} path(s), {total / 1e6:.1f} MB -> {repo}")
    if dry_run:
        for f in paths:
            _echo(f"  {f}")
        return

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    for f in paths:
        if f.is_dir():
            api.upload_folder(folder_path=str(f), path_in_repo=str(f), repo_id=repo,
                              repo_type="dataset")
        else:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=str(f), repo_id=repo,
                            repo_type="dataset")
    _echo(f"published to https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    app()
