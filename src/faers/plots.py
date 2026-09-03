"""Stage 8: figures, as Vega-Lite specifications.

Every figure is emitted twice: a ``.vl.json`` Vega-Lite spec, which is the actual artifact (data
included, so it renders anywhere and can be re-styled without rerunning the pipeline), and a PNG
for contexts that cannot execute JavaScript. The Quarto report embeds the specs, so its charts stay
interactive.

Altair's default row limit is disabled deliberately: these are aggregated tables of at most a few
thousand rows, and truncating a figure silently would misrepresent the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import polars as pl

alt.data_transformers.disable_max_rows()

#: A single ordinal palette used across every figure so colour means the same thing throughout.
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D"]

WIDTH, HEIGHT = 620, 320


def _save(chart: alt.Chart, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = chart.to_dict()
    (out_dir / f"{name}.vl.json").write_text(json.dumps(spec, indent=1))
    try:
        chart.save(str(out_dir / f"{name}.png"), ppi=144)
    except Exception as exc:  # noqa: BLE001 - PNG is a convenience, the spec is the artifact
        print(f"  ! PNG render failed for {name}: {exc}")


def _base(df: pl.DataFrame, title: str) -> alt.Chart:
    return alt.Chart(df.to_pandas(), title=title).properties(width=WIDTH, height=HEIGHT)


def corpus_growth(df: pl.DataFrame, out_dir: Path) -> None:
    """Reports per year. The notebook's opening figure, rebuilt on deduplicated data."""
    d = df.filter(pl.col("year").is_between(2004, 2026))
    chart = (
        _base(d, "FAERS reports per year, deduplicated (2004-2026)")
        .mark_line(point=True, color=PALETTE[0])
        .encode(
            x=alt.X("year:O", title="FDA receipt year"),
            y=alt.Y("reports:Q", title="Reports", axis=alt.Axis(format="~s")),
            tooltip=["year:O", alt.Tooltip("reports:Q", format=",")],
        )
    )
    _save(chart, out_dir, "corpus_growth")


def sex_over_time(df: pl.DataFrame, out_dir: Path) -> None:
    d = df.filter(pl.col("year").is_between(2004, 2026))
    chart = (
        _base(d, "Reporting by patient sex over time (share of reports)")
        .mark_area()
        .encode(
            x=alt.X("year:O", title="FDA receipt year"),
            y=alt.Y("reports:Q", stack="normalize", title="Share of reports",
                    axis=alt.Axis(format="%")),
            color=alt.Color("sex:N", title="Sex",
                            scale=alt.Scale(range=[PALETTE[0], PALETTE[1], "#BAB0AC"])),
            tooltip=["year:O", "sex:N", alt.Tooltip("reports:Q", format=",")],
        )
    )
    _save(chart, out_dir, "sex_over_time")


def age_distribution(df: pl.DataFrame, out_dir: Path) -> None:
    order = ["Neonate", "Infant", "Child", "Adolescent", "Adult", "Elderly", "Unknown"]
    chart = (
        _base(df, "Reports by patient age band")
        .mark_bar(color=PALETTE[0])
        .encode(
            y=alt.Y("age_band:N", sort=order, title=None),
            x=alt.X("reports:Q", title="Reports", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("reports:Q", format=",")],
        )
        .properties(height=220)
    )
    _save(chart, out_dir, "age_distribution")


def reporters(df: pl.DataFrame, out_dir: Path) -> None:
    d = df.sort("reports", descending=True).head(10)
    chart = (
        _base(d, "Who submits reports")
        .mark_bar(color=PALETTE[2])
        .encode(
            y=alt.Y("reporter:N", sort="-x", title=None),
            x=alt.X("reports:Q", title="Reports", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("reports:Q", format=",")],
        )
        .properties(height=260)
    )
    _save(chart, out_dir, "reporters")


def top_terms(df: pl.DataFrame, column: str, title: str, out_dir: Path, name: str) -> None:
    d = df.head(20)
    chart = (
        _base(d, title)
        .mark_bar(color=PALETTE[0])
        .encode(
            y=alt.Y(f"{column}:N", sort="-x", title=None),
            x=alt.X("reports:Q", title="Reports", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("reports:Q", format=",")],
        )
        .properties(height=440)
    )
    _save(chart, out_dir, name)


def report_structure(drugs: pl.DataFrame, reactions: pl.DataFrame, out_dir: Path) -> None:
    """Drugs and reactions per report -- the question the openFDA API could not answer."""
    d = (
        drugs.filter(pl.col("n_drugs") <= 20)
        .rename({"n_drugs": "count"})
        .with_columns(pl.lit("Drugs per report").alias("kind"))
        .vstack(
            reactions.filter(pl.col("n_reactions") <= 20)
            .rename({"n_reactions": "count"})
            .with_columns(pl.lit("Reactions per report").alias("kind"))
        )
    )
    chart = (
        _base(d, "Structure of a report (truncated at 20)")
        .mark_bar()
        .encode(
            x=alt.X("count:O", title="Number per report"),
            y=alt.Y("reports:Q", title="Reports", axis=alt.Axis(format="~s")),
            color=alt.Color("kind:N", title=None, scale=alt.Scale(range=PALETTE[:2])),
            xOffset="kind:N",
            tooltip=["kind:N", "count:O", alt.Tooltip("reports:Q", format=",")],
        )
    )
    _save(chart, out_dir, "report_structure")


def reaction_trends(df: pl.DataFrame, out_dir: Path) -> None:
    """Reaction term trends as a share of the corpus.

    The notebook plotted raw counts for these same three terms and read the divergence as a
    reporting bias. Raw counts rise with the corpus; the share is what isolates a term becoming
    genuinely more prevalent.
    """
    d = df.filter(pl.col("year").is_between(2004, 2026))
    chart = (
        _base(d, "Reaction terms as a share of yearly reports")
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="FDA receipt year"),
            y=alt.Y("share:Q", title="Share of that year's reports", axis=alt.Axis(format="%")),
            color=alt.Color("pt:N", title="Reaction", scale=alt.Scale(range=PALETTE)),
            tooltip=["year:O", "pt:N", alt.Tooltip("share:Q", format=".2%"),
                     alt.Tooltip("reports:Q", format=",")],
        )
    )
    _save(chart, out_dir, "reaction_trends")


def signal_agreement(scored: pl.DataFrame, out_dir: Path) -> None:
    """How much the four screening rules agree -- none is 'the' answer."""
    flags = ["signal_ror", "signal_prr_mhra", "signal_ic", "signal_eb05"]
    labels = {"signal_ror": "ROR025 > 1", "signal_prr_mhra": "PRR >= 2 (MHRA)",
              "signal_ic": "IC025 > 0", "signal_eb05": "EB05 >= 2"}
    rows = [{"measure": labels[f], "pairs": int(scored[f].sum())} for f in flags]
    d = pl.DataFrame(rows)
    chart = (
        _base(d, "Pairs flagged by each screening rule")
        .mark_bar(color=PALETTE[3])
        .encode(
            y=alt.Y("measure:N", sort="-x", title=None),
            x=alt.X("pairs:Q", title="Drug-event pairs flagged", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("pairs:Q", format=",")],
        )
        .properties(height=180)
    )
    _save(chart, out_dir, "signal_agreement")


def shrinkage_effect(scored: pl.DataFrame, out_dir: Path, sample: int = 6_000) -> None:
    """Why EB05 rather than a raw ratio: shrinkage collapses thinly-evidenced extremes.

    The sample is kept small on purpose. Vega-Lite specs embed their data, so this figure alone
    reached 1.8 MB at 20k points and dominated both the repository and the report's page weight.
    The relationship it shows is a dense band plus a bend at the extremes; 6k points render that
    just as legibly at a twelfth of the size.
    """
    d = (
        scored.select("n_ij", "ror", "ebgm")
        .filter(pl.col("ror").is_finite() & pl.col("ebgm").is_finite() & (pl.col("ror") > 0))
        .sample(n=min(sample, scored.height), seed=0)
    )
    chart = (
        _base(d, "Empirical Bayes shrinkage against the raw reporting odds ratio")
        .mark_circle(opacity=0.25, size=14, color=PALETTE[0])
        .encode(
            x=alt.X("ror:Q", scale=alt.Scale(type="log"), title="ROR (unshrunk)"),
            y=alt.Y("ebgm:Q", scale=alt.Scale(type="log"), title="EBGM (shrunk)"),
            tooltip=[alt.Tooltip("n_ij:Q", title="co-reports")],
        )
    )
    _save(chart, out_dir, "shrinkage_effect")


def stratified_survival(df: pl.DataFrame, out_dir: Path) -> None:
    """How many unadjusted signals survive each Mantel-Haenszel adjustment."""
    chart = (
        _base(df, "Signals surviving confounder adjustment")
        .mark_bar(color=PALETTE[2])
        .encode(
            y=alt.Y("variable:N", sort="-x", title=None),
            x=alt.X("surviving:Q", title="Pairs still signalling after adjustment",
                    axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("tested:Q", format=","),
                     alt.Tooltip("surviving:Q", format=","),
                     alt.Tooltip("heterogeneous:Q", format=",")],
        )
        .properties(height=200)
    )
    _save(chart, out_dir, "stratified_survival")


def render_all(results_dir: Path, out_dir: Path) -> list[str]:
    """Build every figure from the result tables."""
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    desc = results_dir / "descriptive"
    read = lambda n: pl.read_parquet(desc / f"{n}.parquet")  # noqa: E731

    corpus_growth(read("reports_per_year"), out_dir)
    sex_over_time(read("sex_by_year"), out_dir)
    age_distribution(read("age_band"), out_dir)
    reporters(read("reporter_occupation"), out_dir)
    top_terms(read("top_ingredients"), "ingredient", "Most-reported active ingredients",
              out_dir, "top_ingredients")
    top_terms(read("top_reactions"), "pt", "Most-reported reaction terms (MedDRA PT)",
              out_dir, "top_reactions")
    top_terms(read("top_indications"), "indi_pt", "Most-reported drug indications",
              out_dir, "top_indications")
    report_structure(read("drugs_per_report"), read("reactions_per_report"), out_dir)
    reaction_trends(read("reaction_trend"), out_dir)

    scored_path = results_dir / "signals" / "scored.parquet"
    if scored_path.exists():
        scored = pl.read_parquet(scored_path)
        signal_agreement(scored, out_dir)
        shrinkage_effect(scored, out_dir)

    strat_summary = results_dir / "stratified" / "survival.parquet"
    if strat_summary.exists():
        stratified_survival(pl.read_parquet(strat_summary), out_dir)

    return sorted(p.name for p in out_dir.glob("*.vl.json"))
