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
import json
from pathlib import Path

import pandas as pd

from ecoli.analysis.new_gene_flux_usage import (
    build_condition_coverage_bar_figure,
    build_lollipop_figure,
    compute_condition_coverage_counts,
    compute_has_flux_fraction,
    load_gene_to_base_reactions,
    write_altair_figure,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "out" / "phenotypic_arrays"
DEFAULT_PLOTS_DIR = DEFAULT_OUT_DIR / "plots"
DEFAULT_GENE_ANNOTATION_CSV = SCRIPT_DIR / "new_metabolic_gene_annotation.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default="new_reactions_original_weights",
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
    result_df["fraction_label"] = (
        result_df["n_wells_with_flux"].astype(str) + "/" + str(n_wells)
    )

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
