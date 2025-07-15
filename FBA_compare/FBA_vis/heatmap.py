import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def find_reaction_columns(
    reaction_id: str, flux_columns: List[str]
) -> Dict[str, List[str]]:
    """
    Find all related columns for a given reaction ID including reverse and expanded forms.

    Parameters
    ----------
    reaction_id : str
        Base reaction ID to search for
    flux_columns : List[str]
        List of all column names in the flux data

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'forward' and 'reverse' keys containing lists of matching column names
    """
    forward_cols = []
    reverse_cols = []

    # Base reaction name variations
    base_patterns = [reaction_id]

    # Add expanded forms with different separators
    separators = ["/", "[", "(", "-", "__"]
    for sep in separators:
        for col in flux_columns:
            if col.startswith(reaction_id + sep):
                base_patterns.append(col)

    # Remove duplicates while preserving order
    base_patterns = list(dict.fromkeys(base_patterns))

    # For each base pattern, find forward and reverse forms
    for pattern in base_patterns:
        # Check if exact match exists
        if pattern in flux_columns:
            forward_cols.append(pattern)

        # Check for reverse form
        reverse_pattern = pattern + " (reverse)"
        if reverse_pattern in flux_columns:
            reverse_cols.append(reverse_pattern)

    return {"forward": forward_cols, "reverse": reverse_cols}


def calculate_net_flux(reaction_id: str, flux_df: pd.DataFrame) -> pd.Series:
    """
    Calculate net flux for a reaction considering all forward and reverse components.

    Parameters
    ----------
    reaction_id : str
        Base reaction ID
    flux_df : pd.DataFrame
        DataFrame containing flux data with time as index

    Returns
    -------
    pd.Series
        Net flux values over time
    """
    flux_columns = flux_df.columns.tolist()
    related_cols = find_reaction_columns(reaction_id, flux_columns)

    # Calculate forward flux (sum of all forward components)
    forward_flux = pd.Series(0, index=flux_df.index)
    for col in related_cols["forward"]:
        forward_flux += flux_df[col].fillna(0)

    # Calculate reverse flux (sum of all reverse components)
    reverse_flux = pd.Series(0, index=flux_df.index)
    for col in related_cols["reverse"]:
        reverse_flux += flux_df[col].fillna(0)

    # Net flux = forward - reverse
    net_flux = forward_flux - reverse_flux

    return net_flux


def load_and_process_data(
    flux_csv: str, core_csv: str, drop_time0: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process flux and core reaction data.

    Parameters
    ----------
    flux_csv : str
        Path to the flux CSV file
    core_csv : str
        Path to the core reactions CSV file
    drop_time0 : bool
        Whether to drop time=0 row

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Processed flux data and core reactions data
    """
    # Load flux data
    flux_df = pd.read_csv(flux_csv, header=0)

    # Set first column as time if it's unnamed
    if flux_df.columns[0] == "Unnamed: 0" or "time" not in flux_df.columns[0].lower():
        flux_df.columns = ["time"] + flux_df.columns[1:].tolist()

    flux_df["time"] = pd.to_numeric(flux_df["time"], errors="coerce")
    flux_df = flux_df.set_index("time")

    if drop_time0:
        flux_df = flux_df[flux_df.index != 0]

    # Load core reactions data
    core_df = pd.read_csv(core_csv)

    return flux_df, core_df


def calculate_all_net_fluxes(
    flux_df: pd.DataFrame, core_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate net fluxes for all core reactions.

    Parameters
    ----------
    flux_df : pd.DataFrame
        Flux data with time as index
    core_df : pd.DataFrame
        Core reactions data

    Returns
    -------
    pd.DataFrame
        DataFrame with reactions as columns and time as index
    """
    net_fluxes = pd.DataFrame(index=flux_df.index)

    for _, row in core_df.iterrows():
        reaction_id = row["BioCyc_id"]
        net_flux = calculate_net_flux(reaction_id, flux_df)
        net_fluxes[reaction_id] = net_flux

    return net_fluxes


def normalize_fluxes(
    net_fluxes: pd.DataFrame, reference_rxn: str = "2.7.3.9-RXN"
) -> pd.DataFrame:
    """
    Normalize all fluxes by the reference reaction at each time point.

    Parameters
    ----------
    net_fluxes : pd.DataFrame
        Net flux data
    reference_rxn : str
        Reference reaction for normalization

    Returns
    -------
    pd.DataFrame
        Normalized flux data
    """
    if reference_rxn not in net_fluxes.columns:
        raise ValueError(f"Reference reaction '{reference_rxn}' not found in flux data")

    reference_flux = net_fluxes[reference_rxn]
    normalized_fluxes = net_fluxes.div(reference_flux, axis=0) * 100

    return normalized_fluxes


def plot_grouped_heatmaps(
    normalized_fluxes: pd.DataFrame,
    core_df: pd.DataFrame,
    reference_rxn: str = "2.7.3.9-RXN",
):
    """
    Create grouped heatmaps by metabolic pathway type.

    Parameters
    ----------
    normalized_fluxes : pd.DataFrame
        Normalized flux data
    core_df : pd.DataFrame
        Core reactions data with type information
    reference_rxn : str
        Reference reaction name for plot title
    """
    # Handle duplicate BioCyc_id values by keeping the first occurrence
    core_df_unique = core_df.drop_duplicates(subset=["BioCyc_id"], keep="first")

    # Create mapping from type_num to type_name
    type_name_map = (
        core_df_unique.drop_duplicates(subset=["type_num"])
        .set_index("type_num")["type_name"]
        .to_dict()
    )

    # Create mapping from reaction to type information
    reaction_type_map = core_df_unique.set_index("BioCyc_id")["type"].to_dict()

    # Group reactions by type
    type_groups = {}
    for reaction in normalized_fluxes.columns:
        if reaction in reaction_type_map:
            type_num = reaction_type_map[reaction]

            # Skip if type_num is NaN or 0
            if pd.isna(type_num) or type_num == 0:
                continue

            type_name = type_name_map.get(type_num, f"Type {type_num}")

            if type_num not in type_groups:
                type_groups[type_num] = {"name": type_name, "reactions": []}
            type_groups[type_num]["reactions"].append(reaction)

    # Sort types by number
    sorted_types = sorted(type_groups.keys())

    # Calculate subplot layout
    n_types = len(sorted_types)
    n_cols = min(3, n_types)  # Maximum 3 columns
    n_rows = (n_types + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_types == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes for easy indexing
    axes_flat = axes.flatten() if n_types > 1 else axes

    for i, type_num in enumerate(sorted_types):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]
        type_info = type_groups[type_num]
        reactions = type_info["reactions"]
        type_name = type_info["name"]

        # Get data for this type
        type_data = normalized_fluxes[reactions]

        # Create heatmap
        im = ax.imshow(
            type_data.T, aspect="auto", cmap="RdBu_r", interpolation="nearest"
        )

        # Set labels
        ax.set_title(f"Type {type_num}: {type_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time Points")
        ax.set_ylabel("Reactions")

        # Set y-axis ticks and labels
        ax.set_yticks(range(len(reactions)))
        ax.set_yticklabels(reactions, fontsize=8)

        # Set x-axis ticks (show every few time points to avoid crowding)
        time_points = normalized_fluxes.index
        n_ticks = min(10, len(time_points))
        tick_indices = np.linspace(0, len(time_points) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(
            [f"{time_points[i]:.1f}" for i in tick_indices], rotation=45, fontsize=8
        )

        # Add colorbar
        plt.colorbar(im, ax=ax, label=f"Normalized Flux\n(% of {reference_rxn})")

    # Hide unused subplots
    for i in range(n_types, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig("grouped_heatmaps_by_type.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_flux_heatmaps_enhanced(
    flux_csv: str,
    core_csv: str,
    reference_rxn: str = "2.7.3.9-RXN",
    drop_time0: bool = True,
):
    """
    Enhanced version of the flux heatmap plotting function.

    Parameters
    ----------
    flux_csv : str
        Path to the flux CSV file (agent_0_csv_analysis_solution_fluxes.csv)
    core_csv : str
        Path to the core reactions CSV file (core_rnx_FBA.csv)
    reference_rxn : str
        Reference reaction for normalization
    drop_time0 : bool
        Whether to drop time=0 row
    """
    flux_df, core_df = load_and_process_data(flux_csv, core_csv, drop_time0)

    net_fluxes = calculate_all_net_fluxes(flux_df, core_df)

    if reference_rxn not in net_fluxes.columns:
        raise ValueError(
            f"Reference reaction {reference_rxn} not found! Available reactions: {net_fluxes.columns.tolist()}"
        )

    normalized_fluxes = normalize_fluxes(net_fluxes, reference_rxn)

    plot_grouped_heatmaps(normalized_fluxes, core_df, reference_rxn)

    # Save processed data
    normalized_fluxes.to_csv("normalized_fluxes.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate grouped heatmaps for FBA flux data."
    )
    parser.add_argument(
        "flux_csv",
        type=str,
        help="Path to the flux CSV file (agent_0_csv_analysis_solution_fluxes.csv)",
    )
    parser.add_argument(
        "core_csv",
        type=str,
        help="Path to the core reactions CSV file (core_rnx_FBA.csv)",
    )
    parser.add_argument(
        "--reference_rxn",
        type=str,
        default="2.7.3.9-RXN",
        help="Reference reaction for normalization (default: 2.7.3.9-RXN)",
    )
    parser.add_argument(
        "--keep_time0", action="store_true", help="Keep time=0 row (default: drop it)"
    )

    args = parser.parse_args()

    plot_flux_heatmaps_enhanced(
        flux_csv=args.flux_csv,
        core_csv=args.core_csv,
        reference_rxn=args.reference_rxn,
        drop_time0=not args.keep_time0,
    )
