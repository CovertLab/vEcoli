import os
import pickle
import textwrap
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
    Args:
        params: Dictionary of parameters given under analysis
            name in configuration JSON.

        Config options look like this:

        ```{json}
        "selected_fluxes": {

            // show_enzyme_counts: whether to show enzyme counts along with fluxes
            "show_enzyme_counts": true,

            // Height of each row and width of each column in inches.
            // Defaults to 3 and 4 inches, respectively.
            // Overidden if "figsize" is provided!
            "row_height": 3,
            "col_width": 4,

            // figsize: custom figure size
            "figsize": [12, 9],

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
        ```

        All options have default values (do not need to be explicitly provided).
    """
    # Marker symbols to loop through for enzyme counts
    MARKER_SYMBOLS = "ovs+*DX"

    # Overwrite default parameters with provided values
    defaults: dict[str, Any] = {
        "show_enzyme_counts": False,
        "figsize": None,
        "row_height": 3,
        "col_width": 4,
        "plot_reactions": [[]],
    }
    params = {**defaults, **params}

    # Set up subplots according to parameters
    n_rows = len(params["plot_reactions"])
    n_cols = 1 if not params["show_enzyme_counts"] else 3
    fig, axs = plt.subplots(n_rows, n_cols)

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
        # Get mappings from sim_data: reactions to catalysts, and fba reaction ids to base reaction ids
        # TODO: Is this the best way to access this information from a multi-variant sim?
        # I think this information is more directly available if using metabolism_redux_classic as well
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

    # Plot each reaction set in a separate axis
    for reaction_set, ax, ax2, lax in zip(
        params["plot_reactions"], flux_axes, count_axes, legend_axes
    ):
        if isinstance(reaction_set, str):
            reaction_set = {reaction_set: reaction_set}
        elif isinstance(reaction_set, list):
            reaction_set = dict(zip(reaction_set, reaction_set))

        # Complain if any reaction ids don't exist in the data
        for rxnid in reaction_set.keys():
            if rxnid not in reaction_ids:
                raise KeyError(
                    f"Reaction with ID {rxnid} was not found in the set of metabolic reactions!"
                )

        # Get reactions indices and read flux values
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
                conn=conn,
            ),
        )

        # Plot flux traces
        for reaction_id, label in reaction_set.items():
            ax.plot(flux_data["time"], flux_data[reaction_id], label=label)

        # Plot enzyme counts, if requested
        if params["show_enzyme_counts"]:
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

            # Axis aesthetics
            ax2.set_ylabel("Enzyme count")
            lax.legend(handles=legend_handles)
            lax.axis("off")

        # Axis aesthetics
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")
        ax.legend()

    if params["figsize"] is None:
        fig.set_size_inches(params["col_width"] * n_cols, params["row_height"] * n_rows)
    else:
        fig.set_size_inches(*params["figsize"])
    fig.tight_layout()

    fig.savefig(os.path.join(outdir, "selected_fluxes.svg"))
    fig.savefig(os.path.join(outdir, "selected_fluxes.png"))
