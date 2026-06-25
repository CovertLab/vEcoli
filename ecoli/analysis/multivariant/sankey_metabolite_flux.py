"""
Sankey diagram of production/consumption flux contributions for chosen
metabolites, for multivariant simulations.

Ports ``sankey_metabolite_flux()`` from
``notebooks/Heena notebooks/Metabolism_New Genes/20260224_WC_sankey_for_flux.ipynb``
onto the current parquet/DuckDB pipeline. For each requested metabolite,
finds every FBA reaction with a nonzero stoichiometric coefficient for that
metabolite (via ``sim_data.process.metabolism.reaction_stoich``), weights
each by mean solution flux, and shows the top producers/consumers as a
Sankey diagram (producers -> metabolite -> consumers). One diagram per
(variant, metabolite), laid out in a grid.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ecoli.analysis.multivariant import _variant_label
from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    open_arbitrary_sim_data,
    read_stacked_columns,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

PASTEL = px.colors.qualitative.Pastel
DEFAULT_TOP_N = 12
DEFAULT_SUBPLOT_WIDTH = 600
DEFAULT_SUBPLOT_HEIGHT = 450
DEFAULT_METS = ["PROTON[c]"]


def plot(
    params: dict[str, Any],
    conn: "DuckDBPyConnection",
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
) -> None:
    """
    All options have default values except ``metabolites_of_interest``.

    Args:
        params: Dictionary of parameters given under analysis
            name in configuration JSON. Config options look like this:

            .. code-block:: json

                {
                    // Metabolite IDs to plot (sim_data convention, e.g.
                    // "PROTON[c]"). Required.
                    "metabolites_of_interest": ["PROTON[c]"],

                    // Number of top producers/consumers to show per side.
                    "top_n": 12,

                    // Drop reactions whose |stoich * mean flux| is below
                    // this threshold.
                    "min_abs_contrib": null,

                    // Width/height of each subplot, in pixels.
                    "subplot_width": 450,
                    "subplot_height": 450
                }
    """
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    metabolites_of_interest: list[str] = params.get(
        "metabolites_of_interest", DEFAULT_METS
    )
    top_n = int(params.get("top_n", DEFAULT_TOP_N))
    min_abs_contrib = params.get("min_abs_contrib")
    subplot_width = int(params.get("subplot_width", DEFAULT_SUBPLOT_WIDTH))
    subplot_height = int(params.get("subplot_height", DEFAULT_SUBPLOT_HEIGHT))

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    reaction_stoich = sim_data.process.metabolism.reaction_stoich

    solution_flux_ids = field_metadata(
        conn, config_sql, "listeners__fba_results__solution_fluxes"
    )

    # For each requested metabolite, every reaction with a nonzero
    # stoichiometric coefficient for it (its "participating reactions").
    met_participants: dict[str, dict[str, int]] = {}
    for met in metabolites_of_interest:
        participants = {
            rxn: stoich[met]
            for rxn, stoich in reaction_stoich.items()
            if stoich.get(met, 0) != 0 and rxn in solution_flux_ids
        }
        if not participants:
            print(
                f"sankey_metabolite_flux: no participating reactions found for "
                f"metabolite {met}; skipping it."
            )
            continue
        met_participants[met] = participants

    if not met_participants:
        print(
            "sankey_metabolite_flux: none of the requested metabolites had "
            "participating reactions; skipping."
        )
        return

    all_reactions = sorted({r for p in met_participants.values() for r in p})
    reaction_idx = [solution_flux_ids.index(r) for r in all_reactions]

    columns = [
        named_idx(
            "listeners__fba_results__solution_fluxes",
            all_reactions,
            [reaction_idx],
        ),
    ]
    raw = pl.DataFrame(
        read_stacked_columns(
            history_sql,
            columns,
            conn=conn,
            order_results=True,
            success_sql=success_sql,
            remove_first=True,
        )
    )
    if raw.is_empty():
        print("sankey_metabolite_flux: no rows returned; skipping.")
        return

    unique_variants = sorted(raw["variant"].unique().to_list())
    mets_plotted = list(met_participants.keys())

    subplot_titles = []
    for variant_val in unique_variants:
        label_l = _variant_label(variant_val, per_variant_params)
        variant_label = " ".join(label_l) if isinstance(label_l, list) else label_l
        for met in mets_plotted:
            subplot_titles.append(f"{variant_label}: {met}")

    fig = make_subplots(
        rows=len(unique_variants),
        cols=len(mets_plotted),
        specs=[[{"type": "domain"}] * len(mets_plotted) for _ in unique_variants],
        subplot_titles=subplot_titles,
    )

    for row, variant_val in enumerate(unique_variants, start=1):
        variant_pdf = raw.filter(pl.col("variant") == variant_val)
        mean_flux = {r: float(variant_pdf[r].to_numpy().mean()) for r in all_reactions}

        for col, met in enumerate(mets_plotted, start=1):
            participants = met_participants[met]
            contrib = {
                rxn: coeff * mean_flux[rxn] for rxn, coeff in participants.items()
            }
            abs_contrib = {rxn: abs(v) for rxn, v in contrib.items()}

            if min_abs_contrib is not None:
                contrib = {
                    r: v
                    for r, v in contrib.items()
                    if abs_contrib[r] >= min_abs_contrib
                }
                abs_contrib = {r: abs_contrib[r] for r in contrib}

            producers = sorted(
                ((r, v) for r, v in contrib.items() if v > 0),
                key=lambda rv: abs_contrib[rv[0]],
                reverse=True,
            )[:top_n]
            consumers = sorted(
                ((r, v) for r, v in contrib.items() if v < 0),
                key=lambda rv: abs_contrib[rv[0]],
                reverse=True,
            )[:top_n]

            if not producers and not consumers:
                print(
                    f"sankey_metabolite_flux: no nonzero contributions for {met} "
                    f"in variant {variant_val}; leaving that panel empty."
                )
                continue

            prod_names = [r for r, _ in producers]
            cons_names = [r for r, _ in consumers]
            nodes = prod_names + [met] + cons_names
            idx = {name: i for i, name in enumerate(nodes)}
            met_idx = idx[met]

            sources, targets, values, link_labels = [], [], [], []
            for r, v in producers:
                sources.append(idx[r])
                targets.append(met_idx)
                values.append(abs_contrib[r])
                link_labels.append(f"{r} produces {met}: |S*v|={abs_contrib[r]:.3g}")
            for r, v in consumers:
                sources.append(met_idx)
                targets.append(idx[r])
                values.append(abs_contrib[r])
                link_labels.append(f"{r} consumes {met}: |S*v|={abs_contrib[r]:.3g}")

            node_colors = [PASTEL[i % len(PASTEL)] for i in range(len(nodes))]
            link_colors = [node_colors[s] for s in sources]

            fig.add_trace(
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=12,
                        thickness=16,
                        line=dict(width=0.5),
                        label=nodes,
                        color=node_colors,
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        label=link_labels,
                        color=link_colors,
                    ),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title="Metabolite Production/Consumption Flux Contributions by Variant",
        template="plotly_white",
        width=subplot_width * len(mets_plotted),
        height=subplot_height * len(unique_variants),
    )

    out_path = os.path.join(outdir, "sankey_metabolite_flux.html")
    fig.write_html(out_path)
    print(f"Saved sankey metabolite flux to {out_path}")
