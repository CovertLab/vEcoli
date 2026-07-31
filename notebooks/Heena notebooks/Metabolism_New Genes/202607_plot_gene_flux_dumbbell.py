"""Plot a lollipop chart of new-metabolic-gene usage: for each of the ~307
genes added to the model in the 2022 metabolic-gene expansion, what fraction
of phenotype-microarray conditions gave that gene's enzyme(s) any flux, for a
single model run.

Requires per-well base-reaction flux data captured by
202607_run_phenotypic_arrays.py --capture-fluxes (not present in the
regular results.csv), plus the accompanying reaction_catalysts/
fba_to_base_reactions side-car JSONs it writes alongside the flux parquet.

Gene -> reaction mapping is enzyme-mediated (matches the user's own prior
analysis in 20250616_test_carbon_source_cp3.ipynb cell 24, inherited from
20250307_track_reaction_usage.ipynb): each gene's "Enzyme encoded" value(s)
[new_metabolic_gene_annotation.csv] are looked up in reaction_catalysts
(reaction -> catalyzing enzyme ids) to find every fba-level reaction that
enzyme catalyzes, which is then collapsed to base reaction ids via
fba_reaction_ids_to_base_reaction_ids -- the same base-reaction id space the
flux parquet is indexed by.

A gene's flux is "present" in a well if any of its mapped base-reaction ids
has |flux| > --epsilon there (binary presence, not magnitude) -- this
directly answers "did adding the 2022 reactions get this gene used", the
question motivating the whole analysis; it also matches the truthiness-only
metric already used in the user's prior notebook precedent.

Two figures are produced from the same per-gene fraction data:
  - a lollipop chart (one stem+dot per gene, --orientation horizontal or
    vertical) of the fraction of PM conditions with flux, optionally
    restricted to the top N genes via --top-n;
  - a bar chart (always over ALL genes, regardless of --top-n) counting how
    many genes fall into each of 4 condition-coverage buckets: used in every
    condition, used in exactly one condition, used in no conditions, or
    other.

With ~307 genes, the horizontal lollipop's output SVG/HTML is intentionally
a tall image, and the vertical orientation's is analogously wide -- this
matches the reference chart style, not a layout bug; --top-n is the
mitigation for either orientation. By default every gene is plotted; pass
--top-n N to only plot the top N genes by fraction. The summary CSV and the
bar chart always cover all genes regardless of this flag.

Usage:
    uv run --env-file .env --project . python3 \
        "notebooks/Heena notebooks/Metabolism_New Genes/202607_plot_gene_flux_dumbbell.py"
"""

import argparse
import ast
import json
from pathlib import Path

import altair as alt
import pandas as pd
import plotly.colors as pc

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "phenotypic_arrays"
DEFAULT_PLOTS_DIR = DEFAULT_OUT_DIR / "plots"
DEFAULT_GENE_ANNOTATION_CSV = SCRIPT_DIR / "new_metabolic_gene_annotation.csv"

MARK_COLOR = pc.qualitative.Pastel[2]
LINE_COLOR = "#dddddd"

COVERAGE_CATEGORY_ORDER = [
    "used in no conditions",
    "used in only one condition",
    "other",
    "used in all conditions",
]


def write_altair_figure(chart, out_dir, stem):
    """Write both an interactive out_dir/stem.html and a static
    out_dir/svg/stem.svg (vl-convert) version of an Altair chart, matching
    the convention used by write_figure() in 202607_plot_phenotypic_results.py."""
    html_path = out_dir / f"{stem}.html"
    chart.save(html_path)
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg_path = svg_dir / f"{stem}.svg"
    chart.save(svg_path)
    return html_path, svg_path


def _parse_enzyme_list(value):
    """Parse an "Enzyme encoded" cell: a bracket-list-string (e.g.
    "['PYRUVATEDEH-CPLX', 'E1P-CPLX']") -> list; a bare non-null string
    (e.g. "AAS-MONOMER") -> single-item list; NaN -> empty list."""
    if pd.isna(value):
        return []
    value = str(value).strip()
    if value.startswith("["):
        return ast.literal_eval(value)
    return [value]


def load_gene_to_base_reactions(gene_annotation_csv, reaction_catalysts, fba_to_base):
    """Map each new gene (keyed by "Gene ID (EcoCyc)") to the set of base
    reaction ids its enzyme(s) catalyze: gene -> enzyme(s) ["Enzyme
    encoded"] -> fba-level reactions [reaction_catalysts, already
    tag-stripped by --capture-fluxes] -> base reaction ids [fba_to_base].
    Returns (genes_df, gene_id -> set-of-base-reaction-ids)."""
    genes_df = pd.read_csv(gene_annotation_csv)
    genes_to_enzymes = genes_df.set_index("Gene ID (EcoCyc)")[
        "Enzyme encoded"
    ].to_dict()

    enzyme_to_fba_reactions = {}
    for rxn, enzymes in reaction_catalysts.items():
        for enzyme in enzymes:
            enzyme_to_fba_reactions.setdefault(enzyme, []).append(rxn)

    gene_to_base_reactions = {}
    for gene_id, enzyme_value in genes_to_enzymes.items():
        fba_reactions = set()
        for enzyme in _parse_enzyme_list(enzyme_value):
            fba_reactions.update(enzyme_to_fba_reactions.get(enzyme, []))
        gene_to_base_reactions[gene_id] = {fba_to_base.get(r, r) for r in fba_reactions}
    return genes_df, gene_to_base_reactions


def compute_has_flux_fraction(flux_df, gene_to_base_reactions, epsilon):
    """Per gene, the fraction of wells (rows of flux_df) where at least one
    of its mapped base-reaction columns has |flux| > epsilon. Reaction ids
    not present in this model's flux_df columns are dropped -- treated as
    no-flux by construction, not an error (a handful of base reaction ids
    can differ between checkpoints' networks).

    Returns (gene_id -> fraction, gene_id -> n_mapped_reactions_present,
    gene_id -> n_wells_with_flux)."""
    columns = set(flux_df.columns)
    fractions, coverage, well_counts = {}, {}, {}
    for gene_id, base_reactions in gene_to_base_reactions.items():
        present = base_reactions & columns
        coverage[gene_id] = len(present)
        if not present:
            fractions[gene_id] = 0.0
            well_counts[gene_id] = 0
            continue
        has_flux = (flux_df[list(present)].abs() > epsilon).any(axis=1)
        well_counts[gene_id] = int(has_flux.sum())
        fractions[gene_id] = float(has_flux.mean())
    return fractions, coverage, well_counts


def categorize_condition_coverage(n_wells_with_flux, n_wells):
    """Bucket one gene's raw well-count into one of the 4 coverage
    categories. Branch order matters for the degenerate n_wells<=1 cases:
    checking ==0 and ==n_wells before ==1 means a gene with flux in a
    single-well run's only well is classified as "all conditions" (the
    stronger claim), not "only one condition"."""
    if n_wells_with_flux == 0:
        return "used in no conditions"
    if n_wells_with_flux == n_wells:
        return "used in all conditions"
    if n_wells_with_flux == 1:
        return "used in only one condition"
    return "other"


def compute_condition_coverage_counts(result_df, n_wells):
    """Categorize every gene in result_df (expected to be the FULL,
    unfiltered-by---top-n set) into COVERAGE_CATEGORY_ORDER buckets.
    Returns a DataFrame with columns ["category", "count"], zero-filled for
    any empty bucket."""
    categories = result_df["n_wells_with_flux"].apply(
        categorize_condition_coverage, n_wells=n_wells
    )
    counts = categories.value_counts().reindex(COVERAGE_CATEGORY_ORDER, fill_value=0)
    return counts.rename_axis("category").reset_index(name="count")


def build_lollipop_figure(result_df, orientation, run_name):
    """Lollipop chart: one stem+dot per gene, stem from 0 to fraction, dot
    at the tip, single MARK_COLOR (no legend -- one series). result_df is
    expected pre-sorted by fraction descending; genes are plotted in that
    order.

    orientation="horizontal": genes on y (categorical), fraction on x
    [0, 1], height scales with gene count.
    orientation="vertical": genes on x (categorical, rotated labels),
    fraction on y [0, 1], width scales with gene count instead."""
    labels = result_df["gene_label"].tolist()
    n_genes = len(result_df)
    size = max(600, 22 * n_genes)
    frac_scale = alt.Scale(domain=[-0.02, 1.02])
    plot_df = result_df.assign(zero=0.0)

    if orientation == "horizontal":
        cat_axis = alt.Y("gene_label:N", sort=labels, title="Gene")
        stem_encode = dict(
            y=cat_axis, x=alt.X("zero:Q", scale=frac_scale), x2="fraction:Q"
        )
        dot_encode = dict(
            y=cat_axis,
            x=alt.X(
                "fraction:Q",
                scale=frac_scale,
                title="Fraction of conditions with flux",
            ),
        )
        width, height = 400, size
        label_encode = dict(dot_encode, text=alt.Text("fraction:Q", format=".2f"))
        label_offset = dict(dx=18)
    else:
        cat_axis = alt.X(
            "gene_label:N",
            sort=labels,
            title="Gene",
            axis=alt.Axis(labelAngle=-90),
        )
        stem_encode = dict(
            x=cat_axis, y=alt.Y("zero:Q", scale=frac_scale), y2="fraction:Q"
        )
        dot_encode = dict(
            x=cat_axis,
            y=alt.Y(
                "fraction:Q",
                scale=frac_scale,
                title="Fraction of conditions with flux",
            ),
        )
        width, height = size, 400
        label_encode = dict(dot_encode, text=alt.Text("fraction:Q", format=".2f"))
        label_offset = dict(dy=-12)

    rules = alt.Chart(plot_df).mark_rule(color=LINE_COLOR).encode(**stem_encode)
    points = (
        alt.Chart(plot_df)
        .mark_circle(size=80, color=MARK_COLOR)
        .encode(tooltip=["gene_label:N", "fraction:Q"], **dot_encode)
    )
    value_labels = (
        alt.Chart(plot_df)
        .mark_text(color="black", fontSize=10, **label_offset)
        .encode(**label_encode)
    )

    chart = (
        (rules + points + value_labels)
        .properties(
            width=width,
            height=height,
            title="New Metabolic Gene Usage: Fraction of PM Conditions With "
            f"Flux ({run_name})",
        )
        .configure_axis(labelLimit=200)
    )
    return chart


def build_condition_coverage_bar_figure(category_counts, n_wells, run_name):
    """Bar chart of gene counts per condition-coverage category, computed
    over ALL genes regardless of --top-n. Each bar is labeled with its
    count's share of the total gene count (category_counts is expected to
    cover ALL genes, so the shares sum to 100%)."""
    total_genes = int(category_counts["count"].sum())
    plot_df = category_counts.assign(fraction=lambda d: d["count"] / total_genes)

    bars = (
        alt.Chart(plot_df)
        .mark_bar(color=MARK_COLOR)
        .encode(
            x=alt.X(
                "category:N",
                sort=COVERAGE_CATEGORY_ORDER,
                title=None,
                axis=alt.Axis(labelAngle=-20),
            ),
            y=alt.Y("count:Q", title="Number of genes"),
            tooltip=["category:N", "count:Q", alt.Tooltip("fraction:Q", format=".1%")],
        )
    )
    value_labels = (
        alt.Chart(plot_df)
        .mark_text(color="black", dy=-8)
        .encode(
            x=alt.X("category:N", sort=COVERAGE_CATEGORY_ORDER),
            y=alt.Y("count:Q"),
            text=alt.Text("fraction:Q", format=".1%"),
        )
    )

    chart = (bars + value_labels).properties(
        width=400,
        height=300,
        title="New Metabolic Gene Condition-Coverage Categories "
        f"({run_name}, n={n_wells} wells, all genes)",
    )
    return chart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default="new_reaction_original_weights",
        help="--out-name used when running 202607_run_phenotypic_arrays.py "
        "--capture-fluxes for this model run",
    )
    parser.add_argument(
        "--flux-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory containing the {name}_fluxes.parquet + "
        "{name}_reaction_catalysts.json + {name}_fba_to_base_reactions.json "
        "side-cars written by --capture-fluxes",
    )
    parser.add_argument(
        "--gene-annotation-csv",
        type=Path,
        default=DEFAULT_GENE_ANNOTATION_CSV,
        help="Path to new_metabolic_gene_annotation.csv (the ~307 new genes, "
        "with Gene ID (EcoCyc)/Gene name/Enzyme encoded columns)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-7,
        help="Absolute flux threshold above which a base reaction counts as "
        "'has flux' in a well, default 1e-7",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory to write the lollipop/bar chart HTML/SVG and a summary CSV",
    )
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="horizontal (default): genes on y-axis, fraction on x-axis, "
        "height scales with gene count. vertical: genes on x-axis with "
        "rotated labels, fraction on y-axis, width scales with gene count "
        "instead.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only plot the top N genes by fraction (default: None, plot "
        "every gene). The summary CSV and the condition-coverage bar chart "
        "always cover all genes regardless of this flag.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open the figures in a browser"
    )
    args = parser.parse_args()

    flux_df = pd.read_parquet(args.flux_dir / f"{args.run_name}_fluxes.parquet")
    print(f"flux matrix: {flux_df.shape[0]} wells x {flux_df.shape[1]} reactions")

    with open(args.flux_dir / f"{args.run_name}_reaction_catalysts.json") as handle:
        reaction_catalysts = json.load(handle)
    with open(args.flux_dir / f"{args.run_name}_fba_to_base_reactions.json") as handle:
        fba_to_base = json.load(handle)

    genes_df, gene_to_base = load_gene_to_base_reactions(
        args.gene_annotation_csv, reaction_catalysts, fba_to_base
    )

    fraction, coverage, well_counts = compute_has_flux_fraction(
        flux_df, gene_to_base, args.epsilon
    )
    n_wells = len(flux_df)

    gene_names = genes_df.set_index("Gene ID (EcoCyc)")["Gene name"].to_dict()
    gene_ids = list(gene_to_base.keys())
    result_df = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "gene_name": [gene_names.get(g) for g in gene_ids],
            "fraction": [fraction.get(g, 0.0) for g in gene_ids],
            "n_reactions": [coverage.get(g, 0) for g in gene_ids],
            "n_wells_with_flux": [well_counts.get(g, 0) for g in gene_ids],
        }
    )
    result_df["gene_label"] = result_df["gene_name"].fillna(result_df["gene_id"])

    n_covered = (result_df["n_reactions"] > 0).sum()
    print(
        f"{n_covered}/{len(result_df)} genes have >=1 mapped reaction present "
        "in this run's base_reaction_ids universe"
    )

    result_df = result_df.sort_values("fraction", ascending=False).reset_index(
        drop=True
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "gene_flux_usage.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    if args.top_n is None:
        plot_df = result_df
    else:
        plot_df = result_df.head(args.top_n).reset_index(drop=True)
    print(f"plotting {len(plot_df)}/{len(result_df)} genes (top_n={args.top_n})")

    chart = build_lollipop_figure(plot_df, args.orientation, args.run_name)
    fig_path, svg_path = write_altair_figure(chart, args.out_dir, "gene_flux_lollipop")
    print(f"wrote {fig_path}")
    print(f"wrote {svg_path}")

    category_counts = compute_condition_coverage_counts(result_df, n_wells)
    print(f"condition-coverage categories (all {len(result_df)} genes):")
    print(category_counts)
    bar_chart = build_condition_coverage_bar_figure(
        category_counts, n_wells, args.run_name
    )
    bar_fig_path, bar_svg_path = write_altair_figure(
        bar_chart, args.out_dir, "gene_flux_condition_coverage_bar"
    )
    print(f"wrote {bar_fig_path}")
    print(f"wrote {bar_svg_path}")

    if args.show:
        chart.show()
        bar_chart.show()


if __name__ == "__main__":
    main()
