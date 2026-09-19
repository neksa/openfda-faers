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

#: Categorical slots, in fixed order, from the validated reference palette. The order is the
#: CVD-safety mechanism, not decoration: slots 1-3 clear the all-pairs separation and
#: normal-vision floors in both light and dark. Hues are assigned in this order and never cycled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

#: Emphasis pair: one accent against a de-emphasis grey. Used where the reader's job is to find
#: the few marks that matter rather than to tell many series apart -- colouring everything would
#: bury the point.
ACCENT = "#2a78d6"
MUTED = "#b8b7b0"

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
        .mark_bar(color=PALETTE[0])
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
        .mark_bar(color=PALETTE[0])
        .encode(
            y=alt.Y("measure:N", sort="-x", title=None),
            x=alt.X("pairs:Q", title="Drug-event pairs flagged", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("pairs:Q", format=",")],
        )
        .properties(height=180)
    )
    _save(chart, out_dir, "signal_agreement")


def shrinkage_effect(scored: pl.DataFrame, out_dir: Path, sample: int = 3_000) -> None:
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
        .mark_bar(color=PALETTE[0])
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


def signal_landscape(scored: pl.DataFrame, out_dir: Path, sample: int = 4_000) -> None:
    """Evidence against effect size for every scored pair -- the standard pharmacovigilance view.

    Form is *emphasis*, not categorical: the reader's job is to find the handful of pairs in the
    top-right, so flagged pairs take the accent hue and everything else recedes to grey. Colouring
    five screening rules categorically would bury exactly the points the chart exists to show.

    Both axes are log. Co-report counts span four orders of magnitude and EB05 three; on linear
    axes the entire corpus collapses into the bottom-left corner.

    The sample is small on purpose. A Vega-Lite spec embeds its data, and at 12,000 points carrying
    drug and reaction names this figure alone was 2.5 MB -- it took the report page from 0.7 MB to
    3.4 MB on its own. 4,000 points render the same density and the same outliers.
    """
    d = (
        scored.select("ingredient", "pt", "n_ij", "eb05", "signal_eb05", "ingredient_curated")
        .filter((pl.col("n_ij") > 0) & (pl.col("eb05") > 0))
        .sample(n=min(sample, scored.height), seed=0)
        .with_columns(
            pl.when(pl.col("signal_eb05"))
            .then(pl.lit("Flagged (EB05 ≥ 2)"))
            .otherwise(pl.lit("Not flagged"))
            .alias("status")
        )
    )
    chart = (
        _base(d, "Signal landscape: evidence against effect size")
        .mark_circle(size=18, opacity=0.45)
        .encode(
            x=alt.X("n_ij:Q", scale=alt.Scale(type="log"), title="Co-reports (evidence)"),
            y=alt.Y("eb05:Q", scale=alt.Scale(type="log"), title="EB05 (shrunk effect size)"),
            color=alt.Color(
                "status:N",
                title=None,
                scale=alt.Scale(
                    domain=["Flagged (EB05 ≥ 2)", "Not flagged"],
                    range=[ACCENT, MUTED],
                ),
                legend=alt.Legend(orient="top-left"),
            ),
            tooltip=[
                alt.Tooltip("ingredient:N", title="Drug"),
                alt.Tooltip("pt:N", title="Reaction"),
                alt.Tooltip("n_ij:Q", title="Co-reports", format=","),
                alt.Tooltip("eb05:Q", title="EB05", format=".2f"),
                alt.Tooltip("ingredient_curated:N", title="Curated ingredient"),
            ],
        )
        .properties(height=380)
    )
    _save(chart, out_dir, "signal_landscape")


def class_similarity(matrix: pl.DataFrame, out_dir: Path) -> None:
    """Between-class mean similarity as a heatmap -- the validation, shown rather than asserted.

    Magnitude on a grid, so the colour job is *sequential*: one hue, light to dark. A diverging or
    categorical scheme here would imply a midpoint or an identity that the data does not have.
    If reaction profiles carry pharmacology the diagonal is visibly darkest.
    """
    labels = {
        "statins": "Statins",
        "ace_inhibitors": "ACE inhibitors",
        "tnf_inhibitors": "TNF inhibitors",
        "ssris": "SSRIs",
        "bisphosphonates": "Bisphosphonates",
    }
    d = matrix.filter(pl.col("similarity").is_not_null()).with_columns(
        pl.col("class_a").replace_strict(labels, default=pl.col("class_a")).alias("a"),
        pl.col("class_b").replace_strict(labels, default=pl.col("class_b")).alias("b"),
    )
    order = list(labels.values())

    cells = (
        alt.Chart(d.to_pandas())
        .mark_rect(stroke="#fcfcfb", strokeWidth=2)  # 2px surface gap between fills
        .encode(
            x=alt.X("b:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("a:N", sort=order, title=None),
            color=alt.Color(
                "similarity:Q",
                title="Mean cosine similarity",
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(orient="right", gradientLength=180),
            ),
            tooltip=[
                alt.Tooltip("a:N", title="Class"),
                alt.Tooltip("b:N", title="vs"),
                alt.Tooltip("similarity:Q", format=".3f"),
                alt.Tooltip("n_pairs:Q", title="Drug pairs", format=","),
            ],
        )
    )
    # Direct labels: three of the sequential steps fall below 3:1 on the light surface, and the
    # relief rule requires visible values rather than colour alone.
    text = cells.mark_text(fontSize=11).encode(
        text=alt.Text("similarity:Q", format=".2f"),
        color=alt.condition(
            alt.datum.similarity > 0.35, alt.value("#ffffff"), alt.value("#52514e")
        ),
    )
    chart = (cells + text).properties(
        width=420, height=320, title="Drug classes resemble themselves, not each other"
    )
    _save(chart, out_dir, "class_similarity")


def drift_trajectories(traj: pl.DataFrame, out_dir: Path) -> None:
    """What a drift-suspect term looks like beside a stable one.

    Emphasis again: flagged terms in the accent hue, stable terms grey, so the abrupt step that
    defines vocabulary change is visible as a shape rather than inferred from a count.
    """
    d = traj.with_columns(
        # qidx is year*4 + quarter - 1. Dividing rather than floor-dividing keeps the four
        # quarters of a year at distinct x positions; collapsing them onto the integer year drew
        # a sawtooth that looked like volatility but was four points stacked on one tick.
        (pl.col("qidx") / 4).alias("year"),
        pl.when(pl.col("drift_suspect"))
        .then(pl.lit("Drift-suspect"))
        .otherwise(pl.lit("Stable"))
        .alias("status"),
    )
    chart = (
        _base(d, "Reaction term trajectories: vocabulary change has a shape")
        .mark_line(strokeWidth=2, opacity=0.85)
        .encode(
            x=alt.X("year:Q", title="FDA receipt year", axis=alt.Axis(format="d")),
            y=alt.Y(
                "share:Q",
                title="Share of that quarter's reports",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format="%"),
            ),
            detail="pt:N",
            color=alt.Color(
                "status:N",
                title=None,
                scale=alt.Scale(domain=["Drift-suspect", "Stable"], range=[ACCENT, MUTED]),
                legend=alt.Legend(orient="top-left"),
            ),
            tooltip=[
                alt.Tooltip("pt:N", title="Term"),
                alt.Tooltip("year:Q", title="Year", format=".0f"),
                alt.Tooltip("share:Q", title="Share", format=".3%"),
            ],
        )
        .properties(height=340)
    )
    _save(chart, out_dir, "drift_trajectories")


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
        signal_landscape(scored, out_dir)

    class_matrix = results_dir / "similarity" / "class_matrix.parquet"
    if class_matrix.exists():
        class_similarity(pl.read_parquet(class_matrix), out_dir)

    traj = results_dir / "drift" / "trajectories.parquet"
    if traj.exists():
        drift_trajectories(pl.read_parquet(traj), out_dir)

    strat_summary = results_dir / "stratified" / "survival.parquet"
    if strat_summary.exists():
        stratified_survival(pl.read_parquet(strat_summary), out_dir)

    return sorted(p.name for p in out_dir.glob("*.vl.json"))
