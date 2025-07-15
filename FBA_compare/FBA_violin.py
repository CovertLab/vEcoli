import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


def find_reaction_columns(reaction_id, flux_df):
    """
    Find all possible columns for a given reaction ID including:
    1. Original reaction
    2. Reverse reaction (with " (reverse)" suffix)
    3. Expanded reactions (with "/", "[", "(", "-", "__" separators)
    4. Reverse of expanded reactions

    Returns: (found_columns, search_summary)
    """
    found_columns = []
    search_summary = {
        "reaction_id": reaction_id,
        "original_found": False,
        "reverse_found": False,
        "expanded_found": [],
        "expanded_reverse_found": [],
        "total_columns": 0,
    }

    # 1. Try original reaction
    if reaction_id in flux_df.columns:
        found_columns.append(reaction_id)
        search_summary["original_found"] = True

    # 2. Try reverse of original reaction
    reverse_name = reaction_id + " (reverse)"
    if reverse_name in flux_df.columns:
        found_columns.append(reverse_name)
        search_summary["reverse_found"] = True

    # 3. Try expanded reactions with different separators
    separators = ["/", "[", "(", "-", "__"]
    for sep in separators:
        # Find all columns that start with reaction_id followed by separator
        expanded_cols = [
            col for col in flux_df.columns if col.startswith(reaction_id + sep)
        ]
        if expanded_cols:
            found_columns.extend(expanded_cols)
            search_summary["expanded_found"].extend(expanded_cols)

            # Also try reverse of expanded reactions
            for col in expanded_cols:
                reverse_expanded = col + " (reverse)"
                if reverse_expanded in flux_df.columns:
                    found_columns.append(reverse_expanded)
                    search_summary["expanded_reverse_found"].append(reverse_expanded)

    # Remove duplicates and update summary
    found_columns = list(set(found_columns))
    search_summary["total_columns"] = len(found_columns)
    search_summary["all_columns"] = found_columns

    return found_columns, search_summary


def calculate_net_flux(reaction_id, flux_df):
    """
    Calculate net flux for a reaction:
    Net flux = Sum of forward fluxes - Sum of reverse fluxes

    Returns: (net_flux_series, search_summary)
    """
    columns, search_summary = find_reaction_columns(reaction_id, flux_df)

    if not columns:
        print(f"Warning: No columns found for reaction {reaction_id}")
        return pd.Series(0, index=flux_df.index), search_summary

    forward_flux = pd.Series(0, index=flux_df.index)
    reverse_flux = pd.Series(0, index=flux_df.index)

    for col in columns:
        if " (reverse)" in col:
            reverse_flux += flux_df[col]
        else:
            forward_flux += flux_df[col]

    net_flux = forward_flux - reverse_flux

    # Add flux statistics to search summary
    search_summary["forward_flux_mean"] = forward_flux.mean()
    search_summary["reverse_flux_mean"] = reverse_flux.mean()
    search_summary["net_flux_mean"] = net_flux.mean()
    search_summary["net_flux_std"] = net_flux.std()
    search_summary["net_flux_min"] = net_flux.min()
    search_summary["net_flux_max"] = net_flux.max()

    return net_flux, search_summary


def save_search_summary(
    search_summaries, reaction_info_df, filename="reaction_search_summary.csv"
):
    """
    Save search summary to CSV file with type information
    """
    # Create reaction to types mapping
    reaction_types_map = defaultdict(list)
    for _, row in reaction_info_df.iterrows():
        reaction_id = row["BioCyc_id"]
        reaction_type = row["type"]
        type_name = row["type_name"]
        reaction_types_map[reaction_id].append(
            {
                "type": reaction_type,
                "type_name": type_name
                if pd.notna(type_name)
                else f"Type_{reaction_type}",
            }
        )

    summary_data = []

    for summary in search_summaries:
        reaction_id = summary["reaction_id"]
        types_info = reaction_types_map.get(
            reaction_id, [{"type": "Unknown", "type_name": "Unknown"}]
        )

        # Format type information
        types_str = "; ".join(
            [f"{info['type']}:{info['type_name']}" for info in types_info]
        )
        type_numbers = [str(info["type"]) for info in types_info]

        row = {
            "Reaction_ID": reaction_id,
            "Types": types_str,
            "Type_Numbers": "; ".join(type_numbers),
            "Multi_Type": len(types_info) > 1,
            "Original_Found": summary["original_found"],
            "Reverse_Found": summary["reverse_found"],
            "Expanded_Count": len(summary["expanded_found"]),
            "Expanded_Reverse_Count": len(summary["expanded_reverse_found"]),
            "Total_Columns_Found": summary["total_columns"],
            "All_Columns": "; ".join(summary["all_columns"]),
            "Expanded_Columns": "; ".join(summary["expanded_found"]),
            "Expanded_Reverse_Columns": "; ".join(summary["expanded_reverse_found"]),
            "Forward_Flux_Mean": summary.get("forward_flux_mean", 0),
            "Reverse_Flux_Mean": summary.get("reverse_flux_mean", 0),
            "Net_Flux_Mean": summary.get("net_flux_mean", 0),
            "Net_Flux_Std": summary.get("net_flux_std", 0),
            "Net_Flux_Min": summary.get("net_flux_min", 0),
            "Net_Flux_Max": summary.get("net_flux_max", 0),
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(filename, index=False)
    print(f"Search summary saved to {filename}")

    return summary_df


def normalize_flux_data(flux_data, reference_flux):
    """
    Normalize flux data using reference flux
    Normalized = (flux / reference_flux) * 100
    """
    normalized_data = {}

    for reaction_id, flux_series in flux_data.items():
        # Avoid division by zero
        normalized_series = []
        for i, (flux_val, ref_val) in enumerate(zip(flux_series, reference_flux)):
            if ref_val != 0:
                normalized_series.append((flux_val / ref_val) * 100)
            else:
                normalized_series.append(0)  # or np.nan if you prefer

        normalized_data[reaction_id] = normalized_series

    return normalized_data


def create_violin_plots(normalized_data, reaction_info_df):
    """
    Create grouped violin plots by reaction type
    Note: A reaction can belong to multiple types
    """
    # Create type_num to type_name mapping - handle duplicates and NaN values
    type_mapping = {}
    for _, row in reaction_info_df.iterrows():
        type_num = row["type_num"]
        type_name = row["type_name"]
        # Skip if type_name is NaN or empty
        if pd.notna(type_name) and str(type_name).strip():
            type_mapping[type_num] = str(type_name).strip()

    print(f"Type mapping: {type_mapping}")

    # Group reactions by type - handle multiple type assignments
    type_groups = defaultdict(list)
    reaction_type_assignments = defaultdict(
        set
    )  # Track which types each reaction belongs to

    # First pass: collect all type assignments for each reaction
    for _, row in reaction_info_df.iterrows():
        reaction_id = row["BioCyc_id"]
        reaction_type = row["type"]
        reaction_type_assignments[reaction_id].add(reaction_type)

    # Second pass: add reactions to type groups
    for reaction_id, flux_values in normalized_data.items():
        # Find all types this reaction belongs to
        if reaction_id in reaction_type_assignments:
            for reaction_type in reaction_type_assignments[reaction_id]:
                type_groups[reaction_type].append(
                    {
                        "reaction_id": reaction_id,
                        "flux_values": flux_values,
                        "all_types": sorted(
                            list(reaction_type_assignments[reaction_id])
                        ),
                    }
                )

    print("\nReaction type assignments:")
    for reaction_id, types in reaction_type_assignments.items():
        if len(types) > 1:
            print(f"  {reaction_id}: belongs to types {sorted(list(types))}")

    # Create subplots
    n_types = len(type_groups)
    if n_types == 0:
        print("No data to plot")
        return

    # Calculate subplot layout
    cols = min(3, n_types)
    rows = (n_types + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_types == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()

    # Plot each type group
    for idx, (reaction_type, reactions) in enumerate(sorted(type_groups.items())):
        ax = axes[idx]

        # Prepare data for violin plot
        plot_data = []
        labels = []

        for reaction_info in reactions:
            reaction_id = reaction_info["reaction_id"]
            flux_values = reaction_info["flux_values"]
            all_types = reaction_info["all_types"]

            # Add annotation if reaction belongs to multiple types
            if len(all_types) > 1:
                label = f"{reaction_id}*"  # Add asterisk for multi-type reactions
            else:
                label = reaction_id

            # Add data for this reaction
            plot_data.extend(flux_values)
            labels.extend([label] * len(flux_values))

        # Create DataFrame for seaborn
        if plot_data:
            df_plot = pd.DataFrame({"Normalized Flux": plot_data, "Reaction": labels})

            # Create violin plot
            sns.violinplot(
                data=df_plot, x="Reaction", y="Normalized Flux", ax=ax, scale="width"
            )

            # Set title using type mapping
            type_name = type_mapping.get(reaction_type, f"Unknown Type {reaction_type}")
            ax.set_title(f"Type {reaction_type}: {type_name}", fontsize=12, pad=10)

            # Improve x-axis labels
            ax.set_xlabel("Reaction ID (* = multi-type)", fontsize=10)
            ax.set_ylabel("Normalized Flux (%)", fontsize=10)

            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

        else:
            type_name = type_mapping.get(reaction_type, f"Unknown Type {reaction_type}")
            ax.set_title(
                f"Type {reaction_type}: {type_name} (No data)", fontsize=12, pad=10
            )
            ax.set_xlabel("Reaction ID", fontsize=10)
            ax.set_ylabel("Normalized Flux (%)", fontsize=10)

    # Hide unused subplots
    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title with note about multi-type reactions
    fig.suptitle(
        "Flux Distribution by Reaction Type\n(* indicates reactions belonging to multiple types)",
        fontsize=14,
        y=0.98,
    )

    # Adjust layout to prevent label overlap
    plt.tight_layout(pad=2.0)
    plt.savefig("flux_violin_plots.png", dpi=300)
    plt.show()


def create_violin_plots_separete(normalized_data, reaction_info_df):
    """
    Create one violin plot per reaction type.
    Each reaction type gets its own matplotlib figure.
    Reactions belonging to multiple types are marked with '*'.
    """
    type_mapping = {}
    for _, row in reaction_info_df.iterrows():
        type_num = row["type_num"]
        type_name = row["type_name"]
        if pd.notna(type_name) and str(type_name).strip():
            type_mapping[type_num] = str(type_name).strip()

    # Create a mapping of reaction IDs to their types
    reaction_type_assignments = defaultdict(set)
    for _, row in reaction_info_df.iterrows():
        reaction_type_assignments[row["BioCyc_id"]].add(row["type"])

    type_groups = defaultdict(list)
    for rxn_id, flux_values in normalized_data.items():
        if rxn_id in reaction_type_assignments:
            all_types = sorted(reaction_type_assignments[rxn_id])
            for t in all_types:
                type_groups[t].append(
                    {
                        "reaction_id": rxn_id,
                        "flux_values": flux_values,
                        "all_types": all_types,
                    }
                )

    for reaction_type in sorted(type_groups):
        reactions = type_groups[reaction_type]
        plot_data = []
        labels = []
        for info in reactions:
            rxn = info["reaction_id"]
            values = info["flux_values"]
            # multi-type reactions are marked with '*'
            label = rxn + ("*" if len(info["all_types"]) > 1 else "")
            plot_data.extend(values)
            labels.extend([label] * len(values))

        if not plot_data:
            continue

        df_plot = pd.DataFrame({"Normalized Flux (%)": plot_data, "Reaction": labels})

        # plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=df_plot, x="Reaction", y="Normalized Flux (%)", scale="width"
        )
        type_name = type_mapping.get(reaction_type, f"Unknown Type {reaction_type}")
        plt.title(f"Type {reaction_type}: {type_name}", pad=12)
        plt.xlabel("Reaction ID (* = multi‑type)")
        plt.ylabel("Normalized Flux (%)")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to execute the flux analysis workflow
    """
    # Read input files
    print("Loading data files...")
    try:
        # Load reaction information
        reaction_info_df = pd.read_csv("core_rnx_FBA.csv")
        print(f"Loaded {len(reaction_info_df)} reactions from core_rnx_FBA.csv")

        # Load flux data
        flux_df = pd.read_csv("agent_0_csv_analysis_solution_fluxes_1080.csv")
        print(f"Loaded flux data with shape: {flux_df.shape}")

        # Set first column as index (time column)
        flux_df.set_index(flux_df.columns[0], inplace=True)
        print(f"Time points: {len(flux_df.index)}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Find reference reaction (2.7.3.9-RXN with type 0)
    print("\nFinding reference reaction...")
    reference_reaction = "2.7.3.9-RXN"
    reference_info = reaction_info_df[
        reaction_info_df["BioCyc_id"] == reference_reaction
    ]

    if reference_info.empty:
        print(
            f"Error: Reference reaction {reference_reaction} not found in core_rnx_FBA.csv"
        )
        return

    if reference_info["type"].iloc[0] != 0:
        print(
            f"Warning: Reference reaction {reference_reaction} has type {reference_info['type'].iloc[0]}, not 0"
        )

    # Calculate reference flux
    reference_flux, ref_summary = calculate_net_flux(reference_reaction, flux_df)
    print(f"Reference flux calculated for {reference_reaction}")

    # Store all search summaries
    all_search_summaries = [ref_summary]

    # Calculate flux for all reactions
    print("\nCalculating flux for all reactions...")
    flux_data = {}

    for _, row in reaction_info_df.iterrows():
        reaction_id = row["BioCyc_id"]
        net_flux, search_summary = calculate_net_flux(reaction_id, flux_df)
        flux_data[reaction_id] = net_flux
        all_search_summaries.append(search_summary)
        print(f"Processed {reaction_id}")

    # Save search summary
    print("\nSaving search summary...")
    summary_df = save_search_summary(all_search_summaries, reaction_info_df)

    # Print summary statistics
    print("\n" + "=" * 50)
    print("REACTION SEARCH SUMMARY")
    print("=" * 50)

    total_reactions = len(all_search_summaries)
    found_original = sum(1 for s in all_search_summaries if s["original_found"])
    found_reverse = sum(1 for s in all_search_summaries if s["reverse_found"])
    found_expanded = sum(1 for s in all_search_summaries if s["expanded_found"])
    found_expanded_reverse = sum(
        1 for s in all_search_summaries if s["expanded_reverse_found"]
    )
    found_any = sum(1 for s in all_search_summaries if s["total_columns"] > 0)

    # Multi-type reaction statistics
    multi_type_count = sum(1 for row in summary_df.iterrows() if row[1]["Multi_Type"])

    print(f"Total reactions analyzed: {total_reactions}")
    print(f"Reactions with original columns found: {found_original}")
    print(f"Reactions with reverse columns found: {found_reverse}")
    print(f"Reactions with expanded columns found: {found_expanded}")
    print(f"Reactions with expanded reverse columns found: {found_expanded_reverse}")
    print(f"Reactions with any columns found: {found_any}")
    print(f"Reactions with no columns found: {total_reactions - found_any}")
    print(f"Reactions belonging to multiple types: {multi_type_count}")

    # Show top 10 reactions with most columns found
    sorted_summaries = sorted(
        all_search_summaries, key=lambda x: x["total_columns"], reverse=True
    )
    print("\nTop 10 reactions with most columns found:")
    for i, summary in enumerate(sorted_summaries[:10]):
        print(
            f"{i + 1:2d}. {summary['reaction_id']}: {summary['total_columns']} columns"
        )

    # Show multi-type reactions
    multi_type_reactions = summary_df[summary_df["Multi_Type"]]
    if not multi_type_reactions.empty:
        print("\nReactions belonging to multiple types:")
        for _, row in multi_type_reactions.iterrows():
            print(f"  {row['Reaction_ID']}: {row['Types']}")

    print("=" * 50)

    # Normalize flux data
    print("\nNormalizing flux data...")
    normalized_data = normalize_flux_data(flux_data, reference_flux)

    # Create violin plots
    print("\nCreating violin plots...")
    create_violin_plots(normalized_data, reaction_info_df)

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
