"""
Plot selected metabolic fluxes across an entire lineage (one seed, all
generations) as a single continuous trace versus cumulative time.

This is the multigeneration analog of ecoli.analysis.single.selected_fluxes.
Because this is a multigeneration analysis, history_sql spans every
generation of a single (experiment, variant, lineage_seed), so each reaction's
flux is drawn as across all generations (with some initial time steps in each
cell omitted, see the _drop_second_time_step() function for more info on this).
"""

import os
import pickle
import textwrap
import warnings
import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from typing import Any, TYPE_CHECKING, cast
from collections import defaultdict
from ecoli.library.parquet_emitter import (
    read_stacked_columns,
    field_metadata,
    named_idx,
    open_arbitrary_sim_data,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def esc_latex(s):
    """Helper function to escape a string for LaTeX."""

    return s.replace("_", r"\_")


def _drop_second_time_step(flux_data: pl.DataFrame) -> pl.DataFrame:
    """
    Remove the second timestep (first remaining timestep after
    "remove_first=True" is used) of each cell.

    The first two timesteps of each cell are not representative of the cell's
    own metabolism and are trimmed before plotting/averaging. The reasoning
    below is for sims run with the metabolism.py process
    (ecoli/processes/metabolism.py). Note: the listeners in the
    metabolism_redux processes behave the same way.

    First timestep: removed upstream by "remove_first=True" in the
    read_stacked_columns call:
      Metabolism computes values at a single instant from the current cell's
      state, so it only solves when next_update_time <= global_time
      (in metabolism.py update_condition()), and listeners are emitted at the
      start of a step, before that step's solve (each row is emitted before
      that increment's solve, so the first row of a cell is never
      computed by that cell, it is a carried over value from previous parent
      cell's last solve (or is an uninitialized default for generation 1)).

      The base_reaction_fluxes listener is declared with listener_schema:
      updater "set" with no divider, so at division vivarium's default "set"
      divider copies the parent's value to each daughter unchanged. So a
      daughter's first flux equals the parent's flux at the division instant,
      not the parent's last recorded row, which was emitted one solve earlier
      (or the schema default (~0) for generation 1, which has no parent). The
      parent's post-solve value is never emitted on the parent side (it only
      shows up as the daughter's first row). Because the first timestep value
      represents the parent's metabolism (or an uninitialized default for
      generation 1) and not the daughter's, it is filtered out from the plot
      and the average calculation.

    Second timestep -- removed by this function:
      The first actual solve after division tends to overshoot. The homeostatic
      driving target is (target_conc-current_conc)/timestep (in metabolism.py).
      At cell birth, counts were halved at division and other processes have
      already advanced a step or two without metabolism replenishing pools, so
      the difference between the current_conc and target_conc might be larger
      than usual. This can drive an unusually large flux near the start of the
      cell, and thus can cause a large-magnitude spike at the start of a cell
      that is uncommon throughout the rest of its lifetime, so it is dropped
      here from the plot too. Since remove_first already dropped the first
      timestep, the earliest row remaining per cell is this second timestep.
      Note: this will not remove spikes that occur later in a cell's lifetime.

    Args:
        flux_data: Polars DataFrame (already filtered with remove_first=True)
            including the cell identity columns (experiment_id, variant,
            lineage_seed, generation, agent_id) and time.

    Returns:
        flux_data with the minimum-time row of each cell removed (the
        second timestep of each cell).
    """
    cell_id_cols = [
        "experiment_id",
        "variant",
        "lineage_seed",
        "generation",
        "agent_id",
    ]
    return flux_data.filter(pl.col("time") > pl.col("time").min().over(cell_id_cols))


def _extract_average_fluxes(
    flux_data: pl.DataFrame, reaction_set_ids: list[str]
) -> dict[str, tuple[float, float]]:
    """Compute the mean and standard deviation of flux for each reaction.

    Statistics are computed over whatever rows are present in ``flux_data``;
    callers that want to exclude the unreliable first two timesteps of each cell
    should pass data already filtered with ``remove_first=True`` and
    :func:`_drop_second_time_step`.

    Args:
        flux_data: Polars DataFrame with one column per reaction.
        reaction_set_ids: List of reaction IDs to compute averages for.

    Returns:
        Dictionary mapping each reaction ID to a ``(mean, std)`` tuple.
    """
    average_fluxes: dict[str, tuple[float, float]] = {}
    for reaction_id in reaction_set_ids:
        if reaction_id not in flux_data.columns:
            raise KeyError(f"Reaction ID {reaction_id} not found in flux data.")
        col = flux_data[reaction_id]
        mean = float(col.cast(float).mean()) if col.len() > 0 else float("nan")
        std = float(col.cast(float).std()) if col.len() > 1 else 0.0
        average_fluxes[reaction_id] = (mean, std)
    return average_fluxes


def _mark_generations(ax, gen_bounds: pl.DataFrame) -> None:
    """Draw dotted vertical lines at generation boundaries (division events).

    Args:
        ax: Matplotlib axis to annotate.
        gen_bounds: Polars DataFrame with "generation" and "t_start"
            columns, sorted by generation.
    """
    starts = gen_bounds["t_start"].to_list()

    # Dotted line at the start of each generation after the first (i.e. at each
    # division/generation boundary). Note: The first generation starts at the
    # left edge of the plot, so "1" is used as the starting index to avoid
    # drawing a line at the very left edge:
    for t_start in starts[1:]:
        ax.axvline(
            t_start, linestyle=":", color="0.5", linewidth=1.0, alpha=0.8, zorder=0
        )


def plot(
    params: dict[str, Any],
    conn: "DuckDBPyConnection",
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
    All options have default values (do not need to be explicitly provided).

    Args:
        params: Dictionary of parameters given under analysis
            name in configuration JSON. Config options look like this:

            .. code-block:: json

                {

                    // show_enzyme_counts: whether to show enzyme counts along with fluxes
                    "show_enzyme_counts": true,

                    // show_flux_averages: whether to show flux averages along with fluxes
                    "show_flux_averages": false,

                    // Height of each row and width of each column in inches.
                    // Defaults to 3 and 4 inches, respectively.
                    // Overidden if "figsize" is provided!
                    "row_height": 3,
                    "col_width": 4,

                    // figsize: custom figure size
                    "figsize": [12, 9],

                    // mark_generations: whether to draw dotted vertical lines at
                    // generation boundaries and label each generation (default true).
                    "mark_generations": true,

                    // plot_reactions: list of reaction-sets to plot in each row.
                    "plot_reactions" : [
                        // "Reaction-sets" can be:
                        // (1) single strings, for the ID of a single reaction
                        "PGLUCISOM-RXN",

                        // (2) lists of string IDs
                        ["3PGAREARR-RXN", "RXN-15513"],

                        // (3) dictionaries of string IDs paired with human-readable labels
                        {"PEPDEPHOS-RXN" : "PEP kinase", "PEPSYNTH-RXN": "PEP synthase"}
                    ]
                }
    """
    # Marker symbols to loop through for enzyme counts
    MARKER_SYMBOLS = "ovs+*DX"

    # Overwrite default parameters with provided values
    defaults: dict[str, Any] = {
        "show_enzyme_counts": False,
        "show_flux_averages": False,
        "figsize": None,
        "row_height": 3,
        "col_width": 4,
        "mark_generations": True,
        "plot_reactions": [[]],
    }
    params = {**defaults, **params}

    # Set up subplots according to parameters
    n_rows = len(params["plot_reactions"])
    n_cols = 1 if not params["show_enzyme_counts"] else 3
    fig, axs = plt.subplots(n_rows, n_cols, dpi=300)

    # Ensure axs is expected shape
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    if n_rows == 1:
        axs = np.array([axs])

    # Group axes by what type of data we'll plot into them
    flux_axes = axs[:, 0]
    count_axes = [None] * n_rows if not params["show_enzyme_counts"] else axs[:, 1]
    legend_axes = [None] * n_rows if not params["show_enzyme_counts"] else axs[:, 2]

    # Retrieve reaction IDs from config metadata
    reaction_ids = np.array(
        field_metadata(conn, config_sql, "listeners__fba_results__base_reaction_fluxes")
    )

    # If selected, pull necessary information to plot enzyme counts
    if params["show_enzyme_counts"]:
        # Get mappings from sim_data: reactions to catalysts, and fba reaction ids to base reaction ids.
        #
        # NOTE: Will NOT work if the mapping from reactions to catalysts changes between variants.
        # Since this is a multigeneration analysis (a single variant/lineage), using
        # open_arbitrary_sim_data is fine as it will just open the only possible option:
        # the sim data corresponding to this lineage's variant.
        with open_arbitrary_sim_data(sim_data_paths) as f:
            sim_data = pickle.load(f)
        reaction_catalyst_mapping = sim_data.process.metabolism.reaction_catalysts
        reaction_id_to_base_reaction_id = (
            sim_data.process.metabolism.reaction_id_to_base_reaction_id
        )

        # Invert reaction_id_to_base_reaction_id
        base_reaction_id_to_reaction_ids = defaultdict(list)
        for rxnid, baseid in reaction_id_to_base_reaction_id.items():
            base_reaction_id_to_reaction_ids[baseid].append(rxnid)

        # Pull catalyst IDs
        catalyst_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__catalyst_counts"
        )

    # Captured from the data on the first reaction set for the figure title
    # (identical across reaction sets since they share the same lineage/cells):
    experiment_id_val: str | None = None
    lineage_seed_val: int | None = None
    generations: list[int] = []

    # Plot each reaction set in a separate axis
    for reaction_set, ax, ax2, lax in zip(
        params["plot_reactions"], flux_axes, count_axes, legend_axes
    ):
        if isinstance(reaction_set, str):
            reaction_set = {reaction_set: reaction_set}
        elif isinstance(reaction_set, list):
            reaction_set = dict(zip(reaction_set, reaction_set))

        # Complain if any reaction ids don't exist in the data, and drop them
        filtered_set = {}
        for rxnid, label in reaction_set.items():
            if rxnid in reaction_ids:
                filtered_set[rxnid] = label
            else:
                warnings.warn(
                    f"Reaction with ID {rxnid} was not found in the set of metabolic reactions!"
                )
        reaction_set = filtered_set

        # Skip if all reactions removed
        if len(reaction_set) == 0:
            continue

        # Get reactions indices and read flux values (because this is a
        # multigeneration analysis, history_sql spans all generations of the
        # lineage):
        reaction_set_ids = list(reaction_set.keys())
        reaction_idx = np.nonzero(
            np.array(reaction_set_ids)[:, np.newaxis] == reaction_ids
        )[1]
        flux_data = cast(
            pl.DataFrame,
            read_stacked_columns(
                history_sql,
                [
                    named_idx(
                        "listeners__fba_results__base_reaction_fluxes",
                        reaction_set_ids,
                        [list(reaction_idx)],
                    )
                ],
                remove_first=True,
                order_results=True,
                conn=conn,
            ),
        )

        # Compute the start time of each generation for the dotted division
        # lines (note that this is done after remove_first=True, so the first
        # timestep of each cell is already dropped and these are off by one
        # timestep, however, this is almost invisible on the plot anyway):
        gen_bounds = (
            flux_data.group_by("generation")
            .agg(pl.col("time").min().alias("t_start"))
            .sort("generation")
        )

        # Record experiment / seed / generations for the figure title (once)
        if lineage_seed_val is None and not flux_data.is_empty():
            experiment_id_val = str(flux_data["experiment_id"][0])
            lineage_seed_val = int(flux_data["lineage_seed"][0])
            generations = [int(g) for g in gen_bounds["generation"].to_list()]

        # Drop the second timestep of each cell (note: the first was already
        # dropped by remove_first) so both the plotted trace and the averages
        # exclude the first two timesteps of each cell):
        flux_data = _drop_second_time_step(flux_data)

        # Optionally compute per-reaction average flux (over the trimmed data,
        # i.e. excluding each cell's first two timesteps) to annotate the legend.
        flux_averages = (
            _extract_average_fluxes(flux_data, reaction_set_ids)
            if params["show_flux_averages"]
            else None
        )

        # Plot flux traces (continuous across generations). When flux averages
        # are requested, append "(mean ± std)" to each reaction's legend label.
        for reaction_id, label in reaction_set.items():
            trace_label = label
            if flux_averages is not None:
                mean, std = flux_averages[reaction_id]
                trace_label = f"{label} ({mean:.2e} ± {std:.2e})"
            ax.plot(flux_data["time"], flux_data[reaction_id], label=trace_label)

        # Axis aesthetics
        ax.set_xlabel("Time (cumulative across generations)")
        ax.set_ylabel("Flux")
        ax.legend()

        # Mark generation boundaries with dotted vertical lines
        if params["mark_generations"]:
            _mark_generations(ax, gen_bounds)

        # Plot enzyme counts, if requested
        if params["show_enzyme_counts"]:
            # Axis aesthetics
            ax2.set_ylabel("Enzyme count")
            ax2.set_xlabel("Time (cumulative across generations)")
            lax.axis("off")

            # Get a tree mapping from catalysts in this subplot (root)
            # to base reactions (level 2) to fba reactions (level 3)
            catalysts_to_reactions_mapping: defaultdict[
                str, defaultdict[str, list[str]]
            ] = defaultdict(lambda: defaultdict(list))
            for baseid in reaction_set:
                # Get fba reactions of this base reaction
                fba_reactions = base_reaction_id_to_reaction_ids[baseid]

                # Collect catalysts of the fba reactions and store
                for rxnid in fba_reactions:
                    catalysts = reaction_catalyst_mapping.get(rxnid, [])
                    if len(catalysts) > 0:
                        for catalyst in catalysts:
                            catalysts_to_reactions_mapping[catalyst][baseid].append(
                                rxnid
                            )

            if len(catalysts_to_reactions_mapping) == 0:
                continue

            # Get catalyst indices and read counts
            catalyst_set_ids = list(catalysts_to_reactions_mapping.keys())
            catalyst_idx = np.nonzero(
                np.array(catalyst_set_ids)[:, np.newaxis] == catalyst_ids
            )[1]
            catalyst_counts = cast(
                pl.DataFrame,
                read_stacked_columns(
                    history_sql,
                    [
                        named_idx(
                            "listeners__fba_results__catalyst_counts",
                            catalyst_set_ids,
                            [list(catalyst_idx)],
                        )
                    ],
                    remove_first=True,
                    order_results=True,
                    conn=conn,
                ),
            )

            # Plot counts for each catalyst
            legend_handles = []
            for i, (catalyst, base_to_fba_reaction_mapping) in enumerate(
                catalysts_to_reactions_mapping.items()
            ):
                # Build label based on base reactions and fba reactions of this catalyst
                label = f"$\\bf{{{esc_latex(catalyst)}}}$"
                for base_rxn, fba_rxns in base_to_fba_reaction_mapping.items():
                    base_rxn_label = reaction_set[base_rxn]
                    label += f"\n  {base_rxn}" + (
                        f" ({base_rxn_label})" if base_rxn_label != base_rxn else ""
                    )

                    for fba_rxn in fba_rxns:
                        # Get suffix distinguishing fba reaction from base reaction
                        suffix = fba_rxn[len(base_rxn) :]
                        if len(suffix) > 0:
                            label += "\n    + " + textwrap.fill(
                                suffix, width=40, subsequent_indent="      "
                            )

                # Plot zero line
                ax2.hlines(
                    0,
                    xmin=catalyst_counts["time"].min(),
                    xmax=catalyst_counts["time"].max(),
                    color="0.8",
                    linewidth=1.0,
                    zorder=0,
                )

                # Plot catalyst counts vs time
                # (Store color so markers plotted separately share color)
                lines = ax2.plot(
                    catalyst_counts["time"],
                    catalyst_counts[catalyst],
                    linestyle="--",
                )
                color = lines[0].get_color()

                # Plot markers separately to sub-sample timepoints
                N_POINTS = 20

                len_t = len(catalyst_counts["time"])
                step = max(1, len_t // N_POINTS)
                subsampled_times = catalyst_counts["time"][::step]
                subsampled_counts = catalyst_counts.filter(
                    pl.col("time").is_in(subsampled_times)
                )

                legend_handles.append(
                    ax2.scatter(
                        subsampled_counts["time"],
                        subsampled_counts[catalyst],
                        color=color,
                        marker=MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)],
                        label=label,
                    )
                )

            # Mark generation boundaries on the enzyme-count axis too
            if params["mark_generations"]:
                _mark_generations(ax2, gen_bounds)

            # Legend
            lax.legend(handles=legend_handles)

    if params["figsize"] is None:
        fig.set_size_inches(params["col_width"] * n_cols, params["row_height"] * n_rows)
    else:
        fig.set_size_inches(*params["figsize"])
    # Leave headroom at the top for the two-line title
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    # Figure title: experiment ID on the top line, with the seed and generation
    # range on a second line below it.
    if lineage_seed_val is not None and generations:
        fig.suptitle(
            f"Selected fluxes | Simulation ID: {experiment_id_val}",
            fontsize="large",
            y=0.99,
        )
        fig.text(
            0.5,
            0.955,
            f"seed {lineage_seed_val} · generations {min(generations)}–{max(generations)}",
            ha="center",
            va="top",
            fontsize="small",
            color="0.3",
        )

    fig.savefig(os.path.join(outdir, "selected_fluxes.svg"))
    fig.savefig(os.path.join(outdir, "selected_fluxes.png"))
