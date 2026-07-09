"""Recreate the phenotype-microarray histogram and heatmap plots from
20250616_test_carbon_source_cp3.ipynb (cells 35-42), reading from the CSV
written by 202607_run_phenotypic_arrays.py instead of live notebook state.

The histogram combines all four PM plates' wells into one shared plot, with
dotted reference lines for each plate's own negative control (PM1 carbon,
PM3 nitrogen, PM4 phosphorus, PM4 sulfur) plus an optional minimal-media
("Basal") reference line computed by actually re-running FBA with no
Add/Remove at all (true minimal media, i.e. normal growth) via
202607_run_phenotypic_arrays.test_NetworkFlowModel. Heatmaps stay per-plate
(an 8x12 grid only makes sense per plate), sharing the same colorscale
boundaries as the combined histogram so a given color means the same
homeostatic_objective range everywhere.

Usage:
    uv run --env-file .env --project . python3 \
        "notebooks/Heena notebooks/Metabolism_New Genes/202607_plot_phenotypic_results.py"
"""

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_CSV = SCRIPT_DIR / "out" / "phenotypic_arrays" / "results.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "phenotypic_arrays" / "plots"
DEFAULT_SIM_FOLDER = "out/phenotypic/basal_min_weights_2500_2026-07-06/"

WELL_RE = re.compile(r"^([A-H])(\d{1,2})$")

# (plate, well) -> reference-line label. Each PM plate's own negative control
# removes a different nutrient category, so these are distinct conditions,
# not all the same as minimal-media "Basal" below.
NEG_CONTROL_WELLS = [
    ("PM1", "A1", "PM1 Carbon Neg Control"),
    ("PM3", "A1", "PM3 Nitrogen Neg Control"),
    ("PM4", "A1", "PM4 Phosphorus Neg Control"),
    ("PM4", "F1", "PM4 Sulfur Neg Control"),
]

# (label, (plate, well) -> bool predicate, output filename, figure title). The
# same predicate is used both to select a df subset and to filter
# NEG_CONTROL_WELLS down to the reference lines relevant to that category.
SOURCE_CATEGORIES = [
    (
        "All Plates Combined",
        lambda plate, well: True,
        "combined_histogram.html",
        "All PM Plates Combined: Homeostatic Objective Distribution",
    ),
    (
        "Carbon",
        lambda plate, well: plate in ("PM1", "PM2"),
        "carbon_histogram.html",
        "Carbon Sources: Homeostatic Objective Distribution",
    ),
    (
        "Nitrogen",
        lambda plate, well: plate == "PM3",
        "nitrogen_histogram.html",
        "Nitrogen Sources: Homeostatic Objective Distribution",
    ),
    (
        "Phosphorus",
        lambda plate, well: plate == "PM4" and parse_well(well)[0] in "ABCDE",
        "phosphorus_histogram.html",
        "Phosphorus Sources: Homeostatic Objective Distribution",
    ),
    (
        "Sulfur",
        lambda plate, well: plate == "PM4" and parse_well(well)[0] in "FGH",
        "sulfur_histogram.html",
        "Sulfur Sources: Homeostatic Objective Distribution",
    ),
]

BIN_COLORS = ["#dda0dd", "#fdf4bf"]
VRECT_COLORS = {
    "Growth": "aqua",
    "No Growth": "yellow",
}


def load_run_phenotypic_arrays():
    """Import 202607_run_phenotypic_arrays.py as a module despite the space in
    its directory path (can't just `import` it normally)."""
    path = SCRIPT_DIR / "202607_run_phenotypic_arrays.py"
    spec = importlib.util.spec_from_file_location("run_phenotypic_arrays", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_basal_homeostatic_objective(rpa, sim_folder, objective_weights):
    """Re-run FBA with no uptake Add/Remove at all -- true minimal media,
    i.e. normal growth -- and return its homeostatic_objective."""
    folder = str(sim_folder)
    if not folder.endswith("/"):
        folder += "/"
    fba, _bulk, metabolism, _output = rpa.load_sim(folder)
    result = rpa.test_NetworkFlowModel(
        objective_weights, metabolism=metabolism, fba=fba
    )
    solution = result[6]
    return float(solution.homeostatic_term)


def parse_well(well):
    match = WELL_RE.match(well)
    if not match:
        raise ValueError(f"Unrecognized well label: {well!r}")
    return match.group(1), int(match.group(2))


def discrete_colorscale(bvals, colors):
    """bvals - boundary values delimiting len(colors) intervals of interest.
    Returns a plotly discrete colorscale."""
    if len(bvals) != len(colors) + 1:
        raise ValueError("len(boundary values) should be equal to len(colors)+1")
    bvals = sorted(bvals)
    nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]
    dcolorscale = []
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
    return dcolorscale


def compute_growth_threshold(df, neg_controls=NEG_CONTROL_WELLS):
    """Growth/No-Growth cutoff = the LOWEST homeostatic_objective among the
    four negative controls (each removes a different nutrient category, so
    they don't all land at the same value). Using the lowest of the four
    means any well scoring above the "healthiest-looking" negative control is
    unambiguously called No Growth -- e.g. if the carbon neg control is 0.05
    and the nitrogen neg control is 0.04, everything above 0.04 is No Growth."""
    by_plate_well = df.set_index(["plate", "well"])["homeostatic_objective"]
    values = []
    for plate, well, _label in neg_controls:
        if (plate, well) in by_plate_well.index:
            value = by_plate_well.loc[(plate, well)]
            if pd.notna(value):
                values.append(float(value))
    if not values:
        raise ValueError("No negative control values found to compute growth threshold")
    return min(values)


def compute_band_boundaries(values, threshold):
    """Binary Growth/No-Growth split of the (now combined, all-plate)
    homeostatic_objective distribution. Lower homeostatic_objective = better
    growth, so everything <= threshold is Growth and everything above it is
    No Growth. Returns three boundary values [min, threshold, max]."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return [float(values.min()), float(threshold), float(values.max())]


def colorbar_ticks(bvals):
    bvals = np.array(bvals)
    tickvals = [np.mean(bvals[k : k + 2]) for k in range(len(bvals) - 1)]
    ticktext = (
        [f"<{bvals[1]:.4g}"]
        + [f"{bvals[k]:.4g}-{bvals[k + 1]:.4g}" for k in range(1, len(bvals) - 2)]
        + [f">{bvals[-2]:.4g}"]
    )
    return tickvals, ticktext


def build_combined_histogram(
    df, boundaries, basal_value, title, neg_controls=NEG_CONTROL_WELLS, nbins=150
):
    """One histogram across every well in the given (sub)set of plates, with
    a dotted reference line per relevant plate's own negative control plus an
    optional minimal-media Basal line. Also annotates the single lowest
    -homeostatic_objective well in this (sub)set, since with many bins it's
    otherwise not obvious which condition a given bar corresponds to."""
    values = df["homeostatic_objective"].dropna().tolist()
    hist = go.Histogram(
        x=values, nbinsx=nbins, marker_color="lightblue", name="Distribution"
    )
    counts, _ = np.histogram(values, bins=nbins)
    max_count = max(counts.max(), 1)

    traces = [hist]
    line_idx = 0
    by_plate_well = df.set_index(["plate", "well"])["homeostatic_objective"]
    for plate, well, label in neg_controls:
        if (plate, well) not in by_plate_well.index:
            continue
        value = by_plate_well.loc[(plate, well)]
        if pd.isna(value):
            continue
        height = max_count * (0.6 + 0.15 * line_idx)
        traces.append(
            go.Scatter(
                x=[value, value],
                y=[0, height],
                mode="lines+text",
                line=dict(color="Navy", width=3, dash="dot"),
                text=[label],
                textposition="bottom center",
                showlegend=False,
            )
        )
        line_idx += 1

    if basal_value is not None:
        height = max_count * (0.6 + 0.15 * line_idx)
        traces.append(
            go.Scatter(
                x=[basal_value, basal_value],
                y=[0, height],
                mode="lines+text",
                line=dict(color="red", width=3, dash="dot"),
                text=["Basal (minimal media, normal growth)"],
                textposition="bottom center",
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)
    band_edges = list(boundaries)
    band_names = ["Growth", "No Growth"]
    for k, name in enumerate(band_names):
        fig.add_vrect(
            x0=band_edges[k],
            x1=band_edges[k + 1],
            line_width=0,
            fillcolor=VRECT_COLORS[name],
            opacity=0.1,
            annotation_text=name,
            annotation_position="top",
            annotation_font_size=16,
            layer="below",
        )

    if not df["homeostatic_objective"].dropna().empty:
        min_idx = df["homeostatic_objective"].idxmin()
        min_row = df.loc[min_idx]
        fig.add_annotation(
            x=min_row["homeostatic_objective"],
            y=0.95,
            xref="x",
            yref="paper",
            text=(
                f"Lowest: {min_row['compound_name']}"
                f"<br>({min_row['plate']} {min_row['well']}, "
                f"{min_row['homeostatic_objective']:.4g})"
            ),
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            bgcolor="white",
            bordercolor="black",
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        title=title,
        xaxis_title="Homeostatic Objective (lower = better growth)",
        yaxis_title="Count",
        bargap=0.05,
    )
    return fig


def build_heatmap(plate, sub, boundaries):
    """Build the plate heatmap."""
    dcolorsc = discrete_colorscale(boundaries, BIN_COLORS)
    tickvals, ticktext = colorbar_ticks(boundaries)

    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    by_well = sub.set_index("well")

    matrix, label = [], []
    for r in rows:
        row_data, row_text = [], []
        for c in cols:
            key = f"{r}{c}"
            if key in by_well.index:
                value = by_well.loc[key, "homeostatic_objective"]
                row_data.append(value)
                compound = by_well.loc[key, "compound_name"]
                text = (
                    "<br>".join(str(compound).split(" ")) if pd.notna(compound) else ""
                )
                row_text.append(text)
            else:
                row_data.append(np.nan)
                row_text.append("")
        matrix.append(row_data)
        label.append(row_text)

    heatmap = go.Heatmap(
        z=matrix,
        x=[str(c) for c in cols],
        y=rows,
        text=label,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=dcolorsc,
        colorbar=dict(thickness=25, tickvals=tickvals, ticktext=ticktext),
    )
    fig = go.Figure(data=[heatmap])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        title=f"{plate}: Plate Reader Heatmap (Homeostatic Objective)",
        xaxis_title="Column",
        yaxis_title="Row",
        yaxis_autorange="reversed",
    )
    return fig


def write_figure(fig, out_dir, stem):
    """Write both an interactive out_dir/stem.html and a static
    out_dir/svg/stem.svg (kaleido) version of a figure."""
    html_path = out_dir / f"{stem}.html"
    fig.write_html(html_path)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg_path = svg_dir / f"{stem}.svg"
    fig.write_image(svg_path)
    return html_path, svg_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Path to results.csv written by 202607_run_phenotypic_arrays.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write the combined histogram and per-plate heatmap HTML files",
    )
    parser.add_argument(
        "--basal-value",
        type=float,
        default=None,
        help="Manually supply the minimal-media homeostatic_objective instead "
        "of recomputing it live (skips loading the simulation checkpoint).",
    )
    parser.add_argument(
        "--skip-basal",
        action="store_true",
        help="Don't compute or draw the Basal (minimal media) reference line at all.",
    )
    parser.add_argument(
        "--sim-folder",
        default=DEFAULT_SIM_FOLDER,
        help="Simulation checkpoint folder used to compute the live Basal "
        "reference line. Must match whatever checkpoint produced "
        "--results-csv for the comparison to be meaningful.",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=150,
        help="Number of histogram bins, default 150 (finer than the original 50).",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open each figure in a browser"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_values = df["homeostatic_objective"].dropna().tolist()
    growth_threshold = compute_growth_threshold(df, NEG_CONTROL_WELLS)
    print(f"Growth/No-Growth threshold (lowest negative control) = {growth_threshold}")
    boundaries = compute_band_boundaries(all_values, growth_threshold)
    print(f"Combined band boundaries (min, threshold, max) = {boundaries}")

    basal_value = args.basal_value
    if basal_value is None and not args.skip_basal:
        # Prefer a "BASAL" row written directly into results.csv by
        # 202607_run_phenotypic_arrays.py (fast: no checkpoint reload) --
        # only fall back to a live re-solve for older results.csv files that
        # predate that row being added.
        basal_rows = df[df["plate"] == "BASAL"]
        if not basal_rows.empty and pd.notna(
            basal_rows.iloc[0]["homeostatic_objective"]
        ):
            basal_value = float(basal_rows.iloc[0]["homeostatic_objective"])
            print(f"Basal homeostatic_objective (from results.csv) = {basal_value}")
        else:
            print(
                f"No BASAL row in results.csv; computing live from {args.sim_folder} ..."
            )
            rpa = load_run_phenotypic_arrays()
            basal_value = compute_basal_homeostatic_objective(
                rpa, args.sim_folder, rpa.DEFAULT_OBJECTIVE_WEIGHTS
            )
            print(f"Basal homeostatic_objective = {basal_value}")

    for label, predicate, filename, title in SOURCE_CATEGORIES:
        mask = df.apply(lambda row: predicate(row["plate"], row["well"]), axis=1)
        sub = df[mask]
        neg_controls = [nc for nc in NEG_CONTROL_WELLS if predicate(nc[0], nc[1])]
        print(f"{label}: {len(sub)} wells")
        hist_fig = build_combined_histogram(
            sub, boundaries, basal_value, title, neg_controls, nbins=args.nbins
        )
        stem = Path(filename).stem
        hist_path, hist_svg_path = write_figure(hist_fig, args.out_dir, stem)
        print(f"wrote {hist_path}")
        print(f"wrote {hist_svg_path}")
        if args.show:
            hist_fig.show()

    for plate in sorted(df["plate"].unique()):
        if plate == "BASAL":
            continue
        sub = df[df["plate"] == plate]
        if sub["homeostatic_objective"].dropna().empty:
            print(
                f"{plate}: no feasible homeostatic_objective values, skipping heatmap"
            )
            continue
        heatmap_fig = build_heatmap(plate, sub, boundaries)
        heatmap_path, heatmap_svg_path = write_figure(
            heatmap_fig, args.out_dir, f"{plate}_heatmap"
        )
        print(f"{plate}: wrote {heatmap_path}")
        print(f"{plate}: wrote {heatmap_svg_path}")
        if args.show:
            heatmap_fig.show()


if __name__ == "__main__":
    main()
