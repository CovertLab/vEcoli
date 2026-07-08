"""
Multiexperiment FBA flux comparison (plotly scatters)

Produces two HTML plots (mirrors cistron_count_comparison_plotly.py /
protein_count_comparison_plotly.py):
  1. fba_flux_comparison_plotly_highlighted_<exp1>_vs_<exp2>.html: every
     reaction in one color except a user-specified highlight list (red).
  2. fba_flux_comparison_plotly_metabolite_<exp1>_vs_<exp2>.html: reactions
     colored by whether they produce/consume a params-driven metabolite of
     interest.

# TODOs:
# TODO 1: write a long and descriptive docstring
# TODO 2: edit the hover data and finalize
# TODO: remove default variables
# TODO: decide if the reaction list cap should be expanded based on how many this affects
# TODO: determine what is causing the odd warning to emit
# TODO: resolve all other TODOs in the script
"""

import os
import pickle
from typing import Any, cast

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import plotly.graph_objects as go

from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
    read_stacked_columns,
)

# USER-DEFINED REACTIONS OF INTEREST (to be highlighted in red).
# Default = murein polymerization (RXN0-5405) and cross-linking (RXN-11302):
PLOT_REACTIONS_OF_INTEREST = ["RXN0-5405", "RXN-11302"]

# Default metabolites of interest for the by-metabolite scheme (overridable via
# params["metabolites_of_interest"]; Bare or tagged ids both work):
DEFAULT_METABOLITES_OF_INTEREST = ["ATP", "Pi"]

FLUX_COLUMN = "listeners__fba_results__reaction_fluxes"

# Cap for long id lists shown in hover text (show first N, then "(+M more)"):
MAX_HOVER_IDS = 5

# (key, color, marker_size, opacity) for the remaining two-way categorization
# scheme:
METABOLITE_CATEGORY_SPECS = [
    ("Involves metabolite of interest", "crimson", 8, 0.85),
    ("Other reactions", "lightseagreen", 6, 0.5),
]


def flux_units(flux_column: str) -> str:
    """Return the flux unit string for a given FBA flux listener column.

    reaction_fluxes / solution_fluxes are the FBA solver velocities on a
    concentration basis (mM/s, i.e. mmol/L/s); estimated_fluxes /
    base_reaction_fluxes are the same converted to molecule counts per
    timestep. # Todo: double check this again
    """
    leaf = flux_column.rsplit("__", 1)[-1]
    if leaf in ("estimated_fluxes", "base_reaction_fluxes"):
        return "counts/timestep"
    return "mM/s"


def _strip_compartment(mol_id: str) -> str:
    """Removes a trailing compartment tag from a molecule id."""
    if mol_id.endswith("]") and len(mol_id) >= 3 and mol_id[-3] == "[":
        return mol_id[:-3]
    return mol_id


def get_reaction_categories(sim_data, flux_column):
    """Builds the per-reaction category sets and hover-detail maps from sim_data.

    Returns a dict with:
      - kinetic_set: reaction ids carrying a kinetic (kcat) constraint
        (metabolism.kinetic_constraint_reactions).
      - catalyst_map: reaction id -> list of catalyst ids
        (metabolism.reaction_catalysts; only non-empty entries).
      - stoich: reaction id -> {metabolite id: coeff}
        (metabolism.reaction_stoich).
    """
    metabolism = sim_data.process.metabolism
    kinetic_set = set(metabolism.kinetic_constraint_reactions)
    catalyst_map = {
        rxn: list(cats) for rxn, cats in metabolism.reaction_catalysts.items() if cats
    }
    stoich = metabolism.reaction_stoich
    return {
        "kinetic_set": kinetic_set,
        "catalyst_map": catalyst_map,
        "stoich": stoich,
    }


def reaction_reactants(rxn_id, stoich):
    """Return the list of reactant (consumed, i.e. negative-coefficient)
    participant ids for a reaction, in stoich dict order."""
    return [met_id for met_id, coeff in stoich.get(rxn_id, {}).items() if coeff < 0]


def reaction_products(rxn_id, stoich):
    """Return the list of product (produced, i.e. positive-coefficient)
    participant ids for a reaction, in stoich dict order."""
    return [met_id for met_id, coeff in stoich.get(rxn_id, {}).items() if coeff > 0]


def reaction_involves_metabolite(rxn_id, stoich, metabolite_bases):
    """Return the signed involvement of a reaction with the metabolites of
    interest: +1 if it (net) produces one, -1 if it (net) consumes one, 0 if it
    does not involve any. metabolite_bases is a set of compartment-stripped
    metabolite ids (a reaction matches if any participant's base id is in it).

    # TODO: figure out a better way to notate both consumption and production
    in the hover text
    NOTE #1: When a reaction both consumes and produces metabolites of interest the
    production sign wins (arbitrary tie-break).
    NOTE #2: for the plot we only use whether the result is nonzero, so the
    exact sign only affects the hover text.
    """
    rxn_stoich = stoich.get(rxn_id, {})
    sign = 0
    for met_id, coeff in rxn_stoich.items():
        if _strip_compartment(met_id) in metabolite_bases:
            if coeff > 0:
                return 1
            sign = -1
    return sign


def _fmt_id_list(ids):
    """Format a list of ids for hover, one per line (via '<br>'), capping at
    MAX_HOVER_IDS with a trailing '(+N more)' line."""
    ids = list(ids)
    if not ids:
        return ""
    shown = "<br>".join(f"&nbsp;&nbsp;- {x}" for x in ids[:MAX_HOVER_IDS])
    if len(ids) > MAX_HOVER_IDS:
        shown += f"<br>&nbsp;&nbsp;- (+{len(ids) - MAX_HOVER_IDS} more)"
    return shown


def _fmt_id_list_with_counts(ids, counts_1, counts_2):
    """Format a list of ids for hover, each annotated with its per-sim average
    bulk count (when resolvable), one per line (via '<br>') on its own
    indented '-' bullet, capping at MAX_HOVER_IDS like _fmt_id_list. An id
    missing from a sim's count map (unresolved bulk index, or absent from
    that sim's bulk container) is shown as 'n/a' for that sim rather than
    silently dropped, so a gap is visible.
    #TODO: find a set of sims to test this functionality on
    """
    ids = list(ids)
    if not ids:
        return ""

    def _one(mol_id):
        c1 = counts_1.get(mol_id)
        c2 = counts_2.get(mol_id)
        c1_str = f"{c1:.3g}" if c1 is not None else "n/a"
        c2_str = f"{c2:.3g}" if c2 is not None else "n/a"
        return f"&nbsp;&nbsp;- {mol_id} (sim1: {c1_str}, sim2: {c2_str})"

    shown = "<br>".join(_one(mid) for mid in ids[:MAX_HOVER_IDS])
    if len(ids) > MAX_HOVER_IDS:
        shown += f"<br>&nbsp;&nbsp;- (+{len(ids) - MAX_HOVER_IDS} more)"
    return shown


def bulk_ids_to_indices(mol_ids, bulk_id_to_idx):
    """Resolve a list of molecule ids to bulk indices for one sim's own bulk
    ordering. Returns (id_to_idx, missing_ids): id_to_idx only contains ids
    that were found; missing_ids lists the ones that were not (e.g. a
    compartment-tag mismatch, or a reaction definition not tracked in bulk
    directly). Missing ids are not an error here -- the caller just omits
    their count from the hover text.
    # TODO: determine if this is the best way to fall back here, or if
    something should be done like what the cistron plot does.
    """
    id_to_idx = {}
    missing = []
    for mol_id in mol_ids:
        idx = bulk_id_to_idx.get(mol_id)
        if idx is None:
            # Fall back to a compartment-stripped match against the bulk
            # container's own (stripped) ids, in case of a tag mismatch:
            idx = bulk_id_to_idx.get(_strip_compartment(mol_id))
        if idx is None:
            missing.append(mol_id)
        else:
            id_to_idx[mol_id] = idx
    return id_to_idx, missing


def read_bulk_means_for_experiment(
    conn: DuckDBPyConnection,
    history_sql: str,
    experiment_id: str,
    bulk_indices: list[int],
    gen_floor: int,
) -> dict[int, float]:
    """Mean-of-per-cell-mean bulk counts for one experiment's bulk indices.

    Mirrors read_bulk_rna_means_for_experiment in cistron_count_comparison_plotly.py:
    reads experiment_id's rows only, averages per cell (generation >= gen_floor),
    then averages across cells. Returns {bulk_idx: mean_count}.

    NOTE: bulk_indices must be indices into this experiment's own sim_data bulk-
    molecule ordering (integer bulk indices are not interchangeable between
    experiments, so this must be called once per experiment with that
    experiment's own index list (see the sim_data_1 / sim_data_2 loading below)).
    """
    if not bulk_indices:
        return {}
    bulk_indices = [int(i) for i in bulk_indices]
    names = [f"bulkrxn_{i}" for i in bulk_indices]
    bulk_expr = named_idx("bulk", names, [bulk_indices])
    subquery = cast(
        str, read_stacked_columns(history_sql, [bulk_expr], order_results=False)
    )
    avg_exprs = ", ".join(f'avg("{n}") AS "{n}"' for n in names)
    exp_literal = experiment_id.replace("'", "''")
    per_cell = conn.sql(
        f"""
        SELECT experiment_id, {avg_exprs}
        FROM ({subquery})
        WHERE generation >= {gen_floor}
            AND experiment_id = '{exp_literal}'
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).pl()
    if per_cell.height == 0:
        return {idx: 0.0 for idx in bulk_indices}
    means = per_cell.drop("experiment_id").mean()
    row = means.row(0, named=True)
    return {idx: row[f"bulkrxn_{idx}"] for idx in bulk_indices}


def make_hover_texts(
    rxn_ids,
    f1_signed,
    f2_signed,
    categories,
    metabolite_bases,
    catalyst_counts_1=None,
    catalyst_counts_2=None,
    reactant_counts_1=None,
    reactant_counts_2=None,
    product_counts_1=None,
    product_counts_2=None,
):
    """Generate Per-reaction hover text.

    Shows average flux for both sims (labeled
    "Sim 1"/"Sim 2", see the plot subtitle for which experiment_id each one
    is) plus the catalyzing enzyme(s), reactant(s), and product(s), each with
    both sims' average bulk count when resolvable, whether kinetically
    constrained, and the key metabolites of interest the reaction involves.

    The count maps (catalyst_counts_1/2, reactant_counts_1/2, product_counts_1/2)
    are keyed by molecule id to get the average bulk count for that sim.
    Note: they are looked up per id here rather than assumed present, since
    some ids may not resolve to a bulk index in one or both sims (see bulk_ids_to_indices).
    # TODO: create a sim comparison with different orderings to test this out
    """
    kinetic_set = categories["kinetic_set"]
    catalyst_map = categories["catalyst_map"]
    stoich = categories["stoich"]
    catalyst_counts_1 = catalyst_counts_1 or {}
    catalyst_counts_2 = catalyst_counts_2 or {}
    reactant_counts_1 = reactant_counts_1 or {}
    reactant_counts_2 = reactant_counts_2 or {}
    product_counts_1 = product_counts_1 or {}
    product_counts_2 = product_counts_2 or {}

    texts = []
    for i, rxn_id in enumerate(rxn_ids):
        a, b = f1_signed[i], f2_signed[i]
        catalysts = catalyst_map.get(rxn_id, [])
        reactants = reaction_reactants(rxn_id, stoich)
        products = reaction_products(rxn_id, stoich)
        constrained = "yes" if rxn_id in kinetic_set else "no"
        key_mets = [
            met_id
            for met_id in stoich.get(rxn_id, {})
            if _strip_compartment(met_id) in metabolite_bases
        ]
        lines = [
            f"<b>{rxn_id}</b><br>",
            f"Sim 1 avg flux: {a:.3e}<br>",
            f"Sim 2 avg flux: {b:.3e}<br>",
            f"Kinetically constrained: {constrained}<br>",
            f"Catalyst avg count(s):<br>"
            f"{_fmt_id_list_with_counts(catalysts, catalyst_counts_1, catalyst_counts_2) or 'none'}<br>",
            f"Reactant avg count(s):<br>"
            f"{_fmt_id_list_with_counts(reactants, reactant_counts_1, reactant_counts_2) or 'none'}<br>",
            f"Product avg count(s):<br>"
            f"{_fmt_id_list_with_counts(products, product_counts_1, product_counts_2) or 'none'}",
        ]
        if key_mets:
            lines.append(f"<br>Metabolite(s) of interest:<br>{_fmt_id_list(key_mets)}")
        texts.append("".join(lines))
    return texts


def _add_parity_line(fig: go.Figure, sim1_log: np.ndarray, sim2_log: np.ndarray):
    """Add a dashed y = x reference line spanning the data range."""
    max_val = max(float(sim1_log.max()), float(sim2_log.max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="black", dash="dash", width=0.5),
            name="y=x",
            showlegend=True,
            hoverinfo="skip",
        )
    )


def _add_active_count_legend(fig: go.Figure, active_legend: str):
    """Add a marker-less legend entry reporting how many reactions are plotted
    (active in >=1 sim out of the shared total)."""
    if not active_legend:
        return
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)"),
            name=active_legend,
            showlegend=True,
            hoverinfo="skip",
        )
    )


def _add_stats_annotation(
    fig: go.Figure, r_value: float, pearson_r2: float, cod_r2: float
):
    """Add the Pearson / coefficient of determination (COD) stats box
    (computed over all plotted reactions)."""
    fig.add_annotation(
        x=0.95,
        y=0.05,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Statistics (plotted reactions):</b><br>"
            f"Pearson r = {r_value:.3f}<br>"
            f"Pearson R² = {pearson_r2:.3f}<br>"
            f"COD R² = {cod_r2:.3f}"
        ),
        showarrow=False,
        align="right",
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        borderpad=10,
        font=dict(size=11, family="monospace"),
    )


def _apply_square_layout(fig: go.Figure, title: str, xlabel: str, ylabel: str):
    """Square log-axis layout shared by all figures."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=900,
        height=900,
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(constrain="domain")


def build_highlighted_figure(
    highlight_mask,
    sim1_log,
    sim2_log,
    hover_texts,
    stats,
    title,
    xlabel,
    ylabel,
    active_legend="",
):
    """Scatter with all reactions one color except a highlighted set (red)."""
    fig = go.Figure()

    bg = ~highlight_mask
    if bg.sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=sim1_log[bg],
                y=sim2_log[bg],
                mode="markers",
                marker=dict(
                    color="lightseagreen", size=6, opacity=0.5, line=dict(width=0)
                ),
                name=f"All reactions ({int(bg.sum())})",
                text=[hover_texts[i] for i in np.where(bg)[0]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )
    if highlight_mask.sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=sim1_log[highlight_mask],
                y=sim2_log[highlight_mask],
                mode="markers",
                marker=dict(
                    color="red",
                    size=11,
                    opacity=0.95,
                    line=dict(width=1, color="black"),
                ),
                name=f"Highlighted ({int(highlight_mask.sum())})",
                text=[hover_texts[i] for i in np.where(highlight_mask)[0]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )
    _add_parity_line(fig, sim1_log, sim2_log)
    _add_active_count_legend(fig, active_legend)
    _add_stats_annotation(fig, *stats)
    _apply_square_layout(fig, title, xlabel, ylabel)
    return fig


def build_categorized_figure(
    category_labels,
    specs,
    sim1_log,
    sim2_log,
    hover_texts,
    stats,
    title,
    xlabel,
    ylabel,
    active_legend="",
):
    """Scatter colored by a two-way category, one trace per category for a
    discrete legend. category_labels is a per-reaction array of labels and
    specs is the matching list of (label, color, size, opacity)."""
    labels_arr = np.array(category_labels, dtype=object)

    fig = go.Figure()
    for label, color, size, opacity in specs:
        mask = labels_arr == label
        if mask.sum() == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=sim1_log[mask],
                y=sim2_log[mask],
                mode="markers",
                marker=dict(
                    color=color, size=size, opacity=opacity, line=dict(width=0)
                ),
                name=f"{label} ({int(mask.sum())})",
                text=[hover_texts[i] for i in np.where(mask)[0]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )
    _add_parity_line(fig, sim1_log, sim2_log)
    _add_active_count_legend(fig, active_legend)
    _add_stats_annotation(fig, *stats)
    _apply_square_layout(fig, title, xlabel, ylabel)
    return fig


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """
    # TODO: change comments to say Sim1 and Sim2 instead of full names to make the output less crowded
    Plot average FBA reaction fluxes from two simulations against each other,
    with reactions of interest highlighted in red and one additional
    categorization scheme (metabolite_of_interest).

    Args:
        params: Dictionary containing parameters of the format::

            {
                # Number of initial generations worth of data to skip
                "skip_n_gens": int (default: 2),
                # Optional override of PLOT_REACTIONS_OF_INTEREST
                "reactions_of_interest": list[str],
                # Metabolites whose producing/consuming reactions are colored in
                # the by-metabolite scheme (bare or tagged ids)
                "metabolites_of_interest": list[str]
                    (default: DEFAULT_METABOLITES_OF_INTEREST),
                # Flux listener column to read (default reaction_fluxes)
                "flux_column": str,
            }
    """
    skip_gens = params.get("skip_n_gens", 2)
    reactions_of_interest = params.get(
        "reactions_of_interest", PLOT_REACTIONS_OF_INTEREST
    )
    metabolites_of_interest = params.get(
        "metabolites_of_interest", DEFAULT_METABOLITES_OF_INTEREST
    )
    flux_column = params.get("flux_column", FLUX_COLUMN)
    flux_unit = flux_units(flux_column)

    # "skip the first skip_gens generations of each seed" -> derive the floor
    # from the data's actual minimum generation (generations are 1-indexed), so
    # skip_gens=2 drops exactly the first two generations per seed:
    min_gen = int(
        conn.sql(f"SELECT min(generation) AS g FROM ({history_sql})").pl()["g"][0]
    )
    gen_floor = min_gen + skip_gens

    # Per-reaction average flux (one averaged value per cell, skipping early
    # generations), grouped so each row is one cell's per-reaction average list:
    # TODO: determine if overall time average would be better
    subquery = cast(
        str, read_stacked_columns(history_sql, [flux_column], order_results=False)
    )
    all_fluxes = conn.sql(
        f"""
        WITH unnested_fluxes AS (
            SELECT unnest({flux_column}) AS flux,
                generate_subscripts({flux_column}, 1) AS idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({subquery})
            WHERE generation >= {gen_floor}
        ),
        avg_fluxes AS (
            SELECT avg(flux) AS avgFlux,
                experiment_id, variant, lineage_seed,
                generation, agent_id, idx
            FROM unnested_fluxes
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, idx
        )
        SELECT list(avgFlux ORDER BY idx) AS avgFlux,
               experiment_id
        FROM avg_fluxes
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).pl()

    unique_exp_ids = all_fluxes["experiment_id"].unique().to_list()
    if len(unique_exp_ids) < 2:
        raise ValueError(
            f"Expected 2 experiments but found {len(unique_exp_ids)}: "
            f"{unique_exp_ids}. Make sure both experiment_ids are in the config."
        )

    exp_id_1, exp_id_2 = unique_exp_ids[0], unique_exp_ids[1]

    print(f"Comparing {exp_id_1} (Sim1; x-axis) vs {exp_id_2} (Sim2; y-axis)")

    fluxes_exp1 = all_fluxes.filter(pl.col("experiment_id") == exp_id_1)
    fluxes_exp2 = all_fluxes.filter(pl.col("experiment_id") == exp_id_2)

    sim1_fluxes = ndlist_to_ndarray(fluxes_exp1["avgFlux"])
    sim2_fluxes = ndlist_to_ndarray(fluxes_exp2["avgFlux"])
    sim1_avg_raw = sim1_fluxes.mean(axis=0)
    sim2_avg_raw = sim2_fluxes.mean(axis=0)
    print(f"Sim1 has {len(sim1_fluxes)} cells")
    print(f"Sim2 has {len(sim2_fluxes)} cells")

    # Align reactions by ID, not by list position, using each sim's sim_data:
    exp1_config = f"SELECT * FROM ({config_sql}) WHERE experiment_id = '{exp_id_1}'"
    exp2_config = f"SELECT * FROM ({config_sql}) WHERE experiment_id = '{exp_id_2}'"
    rxn_ids_1 = field_metadata(conn, exp1_config, flux_column)
    rxn_ids_2 = field_metadata(conn, exp2_config, flux_column)
    if len(rxn_ids_1) != len(sim1_avg_raw):
        raise ValueError(
            f"{exp_id_1}: reaction id count ({len(rxn_ids_1)}) does not match "
            f"averaged flux length ({len(sim1_avg_raw)})."
        )
    if len(rxn_ids_2) != len(sim2_avg_raw):
        raise ValueError(
            f"{exp_id_2}: reaction id count ({len(rxn_ids_2)}) does not match "
            f"averaged flux length ({len(sim2_avg_raw)})."
        )

    pos1 = {r: i for i, r in enumerate(rxn_ids_1)}
    pos2 = {r: i for i, r in enumerate(rxn_ids_2)}

    # Sorted intersection is a deterministic, order-independent point list:
    rxn_ids = sorted(pos1.keys() & pos2.keys())
    if not rxn_ids:
        raise ValueError(
            f"No reaction ids shared between {exp_id_1} ({len(rxn_ids_1)} rxns) "
            f"and {exp_id_2} ({len(rxn_ids_2)} rxns); cannot compare."
        )
    sim1_avg = sim1_avg_raw[[pos1[r] for r in rxn_ids]]
    sim2_avg = sim2_avg_raw[[pos2[r] for r in rxn_ids]]

    only_in_1 = sorted(pos1.keys() - pos2.keys())
    only_in_2 = sorted(pos2.keys() - pos1.keys())
    print(
        f"Reactions: {len(rxn_ids_1)} in {exp_id_1}, {len(rxn_ids_2)} in "
        f"{exp_id_2}; {len(rxn_ids)} shared (plotted)."
    )
    if only_in_1:
        preview = ", ".join(only_in_1[:10]) + (" ..." if len(only_in_1) > 10 else "")
        print(f"  {len(only_in_1)} only in {exp_id_1} (dropped): {preview}")
    if only_in_2:
        preview = ", ".join(only_in_2[:10]) + (" ..." if len(only_in_2) > 10 else "")
        print(f"  {len(only_in_2)} only in {exp_id_2} (dropped): {preview}")

    # Keep only reactions that actually carry flux: |avg flux| > 0 in at least one
    # sim. NOTE: Most of the ~9k FBA reactions are inactive (exactly zero) and
    # would pile up at the origin, so they are dropped  from the plot. The
    # legend reports how many of the shared reactions remained active:
    n_shared = len(rxn_ids)
    shared_id_set = set(rxn_ids)  # all shared ids, before the active filter
    active = (np.abs(sim1_avg) > 0) | (np.abs(sim2_avg) > 0)
    n_active = int(active.sum())
    print(f"Active reactions (|avg flux| > 0 in >=1 sim): {n_active} of {n_shared}")
    rxn_ids = [r for r, keep in zip(rxn_ids, active) if keep]
    sim1_avg = sim1_avg[active]
    sim2_avg = sim2_avg[active]
    active_legend = f"{n_active} of {n_shared} reactions active in ≥1 sim"

    # Log-transform the magnitude of the signed average flux (+1 pseudocount), so
    # reactions off in one sim still appear; the hover keeps the signed values.
    # TODO: determine if there is a better way to handle the negative magnitude
    #  fluxes so we dont net them, look into how these are analyzed usually in
    #  wcEcoli scripts
    sim1_log = np.log10(np.abs(sim1_avg) + 1)
    sim2_log = np.log10(np.abs(sim2_avg) + 1)

    # Stats (computed over all plotted reactions):
    r_value = pearsonr(sim1_log, sim2_log)[0]
    pearson_r2 = r_value**2
    cod_r2 = r2_score(sim2_log, sim1_log)
    stats = (r_value, pearson_r2, cod_r2)

    # Category sets/maps from sim_data (ids align with the flux listener ids).
    # This one shared sim_data load is safe for reaction *definitions*
    # (reaction_stoich / reaction_catalysts / kinetic_constraint_reactions), which
    # are assumed consistent between the two sims being compared.
    # TODO: determine if it is just as fast to do each separately, and if so,
    #  what to do if there are mismatches
    with open_arbitrary_sim_data(sim_data_paths) as f:
        sim_data = pickle.load(f)
    categories = get_reaction_categories(sim_data, flux_column)
    kinetic_set = categories["kinetic_set"]
    catalyst_map = categories["catalyst_map"]
    stoich = categories["stoich"]

    metabolite_bases = {_strip_compartment(m) for m in metabolites_of_interest}

    n_kinetic = sum(1 for r in rxn_ids if r in kinetic_set)
    n_catalyzed = sum(1 for r in rxn_ids if r in catalyst_map)
    involvement = np.array(
        [reaction_involves_metabolite(r, stoich, metabolite_bases) for r in rxn_ids]
    )
    n_involved = int((involvement != 0).sum())
    print(f"Kinetically constrained reactions: {n_kinetic}")
    print(f"Catalyzed reactions: {n_catalyzed}")
    print(
        f"Reactions involving metabolites of interest "
        f"{sorted(metabolite_bases)}: {n_involved}"
    )

    # Catalyst/reactant average bulk counts per sim for hover text. Only
    # computed for the reactions actually being plotted (the active/filtered
    # rxn_ids above):
    with open_arbitrary_sim_data({exp_id_1: sim_data_paths[exp_id_1]}) as f:
        sim_data_1 = pickle.load(f)
    with open_arbitrary_sim_data({exp_id_2: sim_data_paths[exp_id_2]}) as f:
        sim_data_2 = pickle.load(f)
    bulk_id_to_idx_1 = {
        bid: i
        for i, bid in enumerate(
            sim_data_1.internal_state.bulk_molecules.bulk_data["id"].tolist()
        )
    }
    bulk_id_to_idx_2 = {
        bid: i
        for i, bid in enumerate(
            sim_data_2.internal_state.bulk_molecules.bulk_data["id"].tolist()
        )
    }

    # Distinct catalyst + reactant + product ids across only the plotted
    # reactions:
    hover_molecule_ids: set[str] = set()
    for r in rxn_ids:
        hover_molecule_ids.update(catalyst_map.get(r, []))
        hover_molecule_ids.update(reaction_reactants(r, stoich))
        hover_molecule_ids.update(reaction_products(r, stoich))

    idx_map_1, missing_1 = bulk_ids_to_indices(hover_molecule_ids, bulk_id_to_idx_1)
    idx_map_2, missing_2 = bulk_ids_to_indices(hover_molecule_ids, bulk_id_to_idx_2)
    if missing_1:
        preview = ", ".join(sorted(missing_1)[:10]) + (
            " ..." if len(missing_1) > 10 else ""
        )
        print(
            f"  WARNING: {len(missing_1)} catalyst/reactant/product id(s) not "
            f"found in {exp_id_1}'s bulk container (hover count omitted): {preview}"
        )
    if missing_2:
        preview = ", ".join(sorted(missing_2)[:10]) + (
            " ..." if len(missing_2) > 10 else ""
        )
        print(
            f"  WARNING: {len(missing_2)} catalyst/reactant/product id(s) not "
            f"found in {exp_id_2}'s bulk container (hover count omitted): {preview}"
        )

    means_1 = read_bulk_means_for_experiment(
        conn, history_sql, exp_id_1, list(idx_map_1.values()), gen_floor
    )
    means_2 = read_bulk_means_for_experiment(
        conn, history_sql, exp_id_2, list(idx_map_2.values()), gen_floor
    )
    # mol id to average count mapping, for whichever ids resolved to a bulk index in
    # each sim (ids missing from a sim's bulk container are simply absent from
    # that sim's map, and make_hover_texts renders them as 'n/a'):
    # TODO: determine if this is the best way to handle this going forward
    counts_1 = {mol_id: means_1[idx] for mol_id, idx in idx_map_1.items()}
    counts_2 = {mol_id: means_2[idx] for mol_id, idx in idx_map_2.items()}

    hover_texts = make_hover_texts(
        rxn_ids,
        sim1_avg,
        sim2_avg,
        categories,
        metabolite_bases,
        catalyst_counts_1=counts_1,
        catalyst_counts_2=counts_2,
        reactant_counts_1=counts_1,
        reactant_counts_2=counts_2,
        product_counts_1=counts_1,
        product_counts_2=counts_2,
    )

    def extract_short_id(exp_id):
        """Extract a short identifier from the full experiment ID."""
        parts = exp_id.split("_")
        for i, part in enumerate(parts):
            if "-" in part and len(part) == 15:
                # timestamp format YYYYMMDD-HHMMSS
                return "_".join(parts[:i])
        return exp_id

    sim1_short = extract_short_id(exp_id_1)
    sim2_short = extract_short_id(exp_id_2)

    comparison_outdir = outdir + f"_{exp_id_1}_vs_{exp_id_2}"
    os.makedirs(comparison_outdir, exist_ok=True)

    subtitle = (
        f"<sub>|avg flux| log-log; flux units = {flux_unit}; "
        f"{skip_gens} generations skipped. "
        f"Sim 1 (x): {exp_id_1} (avg over {len(sim1_fluxes)} cells) vs. "
        f"<br>Sim 2 (y): {exp_id_2} (avg over {len(sim2_fluxes)} cells)</sub>"
    )
    xlabel = f"log10(|Sim 1 avg flux| + 1) ({flux_unit})"
    ylabel = f"log10(|Sim 2 avg flux| + 1) ({flux_unit})"

    # Plot 1: highlighted reactions of interest.
    highlight_set = set(reactions_of_interest)
    active_id_set = set(rxn_ids)  # active (plotted) shared ids
    print("\nHIGHLIGHT REACTIONS:")
    for r in reactions_of_interest:
        if r not in shared_id_set:
            print(f"  WARNING: '{r}' not found in '{flux_column}' for these sims.")
        elif r not in active_id_set:
            print(
                f"  NOTE: '{r}' has zero avg flux in BOTH sims (inactive); "
                f"it is filtered out and will not appear on the plot."
            )
        else:
            print(f"  + '{r}' highlighted.")
    highlight_mask = np.array([r in highlight_set for r in rxn_ids])
    fig_hi = build_highlighted_figure(
        highlight_mask,
        sim1_log,
        sim2_log,
        hover_texts,
        stats,
        f"FBA Reaction Flux Comparison (highlighted)<br>{subtitle}",
        xlabel,
        ylabel,
        active_legend,
    )
    hi_filename = os.path.join(
        comparison_outdir,
        f"fba_flux_comparison_plotly_highlighted_{sim1_short}_vs_{sim2_short}.html",
    )
    fig_hi.write_html(hi_filename)
    print(
        f"Highlighted reactions: {int(highlight_mask.sum())} of "
        f"{len(reactions_of_interest)} requested"
    )

    # Highlight labels plot with metabolite of interest (any reaction producing/consuming one):
    metabolite_labels = [
        METABOLITE_CATEGORY_SPECS[0][0] if inv != 0 else METABOLITE_CATEGORY_SPECS[1][0]
        for inv in involvement
    ]
    fig_met = build_categorized_figure(
        metabolite_labels,
        METABOLITE_CATEGORY_SPECS,
        sim1_log,
        sim2_log,
        hover_texts,
        stats,
        (
            f"FBA Flux Comparison by Metabolite of Interest "
            f"({', '.join(sorted(metabolite_bases))})<br>{subtitle}"
        ),
        xlabel,
        ylabel,
        active_legend,
    )
    met_filename = os.path.join(
        comparison_outdir,
        f"fba_flux_comparison_plotly_metabolite_{sim1_short}_vs_{sim2_short}.html",
    )
    fig_met.write_html(met_filename)

    # Override the default metadata saving file path:
    return {"metadata_path": comparison_outdir}
