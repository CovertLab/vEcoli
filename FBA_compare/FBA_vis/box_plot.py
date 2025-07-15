import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore")

# Read the CSV data
df = pd.read_csv("merged_flux_data_complete.csv")

# Read FBA data and preference data
fba_data = pd.read_csv("agent_0_csv_analysis_solution_fluxes_1080.csv")
prefer_data = pd.read_csv("prefer_FBA_rnx.csv")

# Remove time = 0 from FBA data (first column is time)
fba_data = fba_data[fba_data.iloc[:, 0] != 0]

# Normalize FBA data using reference reaction - ONLY for preferred reactions
reference_rxn = "2.7.3.9-RXN"


def normalize_preferred_fba_data(fba_data, prefer_data, reference_rxn):
    """
    Normalize FBA data using reference reaction values - only for preferred reactions
    """
    # Get list of preferred reactions from BioCyc_id column
    preferred_reactions = prefer_data["BioCyc_id"].tolist()

    # Check if reference reaction exists in FBA data
    if reference_rxn not in fba_data.columns:
        print(f"Warning: Reference reaction {reference_rxn} not found in FBA data")
        print(f"Available columns: {list(fba_data.columns)[:10]}...")
        return fba_data

    # Create a copy of FBA data
    fba_normalized = fba_data.copy()

    # Get reference values (skip time column)
    ref_values = fba_data[reference_rxn]

    # Normalize only the preferred reactions
    for reaction in preferred_reactions:
        if reaction in fba_data.columns:
            for i in range(len(fba_data)):
                if ref_values.iloc[i] != 0:  # Avoid division by zero
                    fba_normalized.iloc[i, fba_data.columns.get_loc(reaction)] = (
                        fba_data.iloc[i, fba_data.columns.get_loc(reaction)]
                        / ref_values.iloc[i]
                    )
                else:
                    fba_normalized.iloc[i, fba_data.columns.get_loc(reaction)] = 0

    print(f"FBA data normalized using reference reaction: {reference_rxn}")
    print(
        f"Normalized {len([r for r in preferred_reactions if r in fba_data.columns])} preferred reactions"
    )

    # Scale to percentage for better visualization
    fba_normalized = fba_normalized * 100
    return fba_normalized


# Apply normalization only to preferred reactions
fba_data = normalize_preferred_fba_data(fba_data, prefer_data, reference_rxn)


# Automatically detect the number of data groups
def detect_data_groups(df):
    """
    Automatically detect the number of data groups based on column names
    Returns a list of data group prefixes (e.g., ['data1', 'data2', 'data3'])
    """
    data_groups = []
    col_names = df.columns.tolist()

    # Look for patterns like 'data1_best_fit', 'data2_best_fit', etc.
    for col in col_names:
        if "_best_fit" in col:
            prefix = col.replace("_best_fit", "")
            if prefix not in data_groups:
                data_groups.append(prefix)

    return sorted(data_groups)


# Detect data groups dynamically
data_groups = detect_data_groups(df)
n_groups = len(data_groups)

print(f"Detected {n_groups} data groups: {data_groups}")

# Convert numeric columns to proper data types
numeric_cols = []
for group in data_groups:
    numeric_cols.extend([f"{group}_best_fit", f"{group}_LB95", f"{group}_UB95"])

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Filter only rows with type values (non-null)
df_filtered = df[df["type"].notna()].copy()

# Create a proper mapping for type names
type_mapping = {
    1: "Glycolysis",
    2: "Pentose phosphate pathway",
    3: "ED",
    4: "TCA Cycle",
    5: "Glyox",
    6: "Amphibolic",
}

# Fill missing type_name values using the mapping
df_filtered["type_name"] = df_filtered["type"].map(type_mapping)

print(f"Original data shape: {df.shape}")
print(f"Filtered data shape (with type): {df_filtered.shape}")
print(f"Unique types: {df_filtered['type'].unique()}")
print(f"Type names: {df_filtered['type_name'].unique()}")


# Function to format flux names
def format_flux_name(flux_name):
    """
    Format flux names according to the rules:
    - If contains '(net)', remove 'net' and use '<->' to connect reactants and products
    - If no '(net)', use '->' to connect reactants and products
    """
    if pd.isna(flux_name):
        return flux_name

    # Remove (net) and (exch) for cleaner display
    if "(net)" in flux_name:
        flux_name = flux_name.replace("(net)", "").strip()
        # Use bidirectional arrow for net fluxes
        if "<->" not in flux_name and "->" in flux_name:
            flux_name = flux_name.replace("->", "<->")

    return flux_name


# Apply flux name formatting
df_filtered["flux_formatted"] = df_filtered["flux"].apply(format_flux_name)


# Create improved BioCyc mapping
def create_biocyc_mapping(df_filtered, prefer_data):
    """
    Create mapping from BioCyc_id to flux data
    """
    biocyc_to_flux_mapping = {}

    # Create a mapping from BioCyc_id to flux information
    for _, row in df_filtered.iterrows():
        biocyc_id = row["BioCyc_id"]
        if pd.notna(biocyc_id):
            biocyc_to_flux_mapping[biocyc_id] = {
                "flux_formatted": row["flux_formatted"],
                "flux_original": row["flux"],
            }

    print(
        f"BioCyc mapping created from main data: {len(biocyc_to_flux_mapping)} mappings"
    )
    return biocyc_to_flux_mapping


biocyc_to_flux_mapping = create_biocyc_mapping(df_filtered, prefer_data)


# Function to get FBA data for a specific flux - optimized for preferred reactions
def get_fba_data_for_flux(flux_formatted, biocyc_to_flux_mapping, fba_data):
    """
    Get FBA data for a specific flux name
    Returns list of values or None if not found
    """
    # Find the BioCyc_id for this flux
    biocyc_id = None
    for biocyc, flux_info in biocyc_to_flux_mapping.items():
        if flux_info["flux_formatted"] == flux_formatted:
            biocyc_id = biocyc
            break

    if biocyc_id is None:
        return None

    # Check if this BioCyc_id exists in FBA data columns
    if biocyc_id in fba_data.columns:
        fba_values = fba_data[biocyc_id].dropna()
        return fba_values.tolist() if len(fba_values) > 0 else None

    return None


# Debug: Print available FBA columns and mappings
print("\n=== FBA Data Debug ===")
print(f"FBA data shape: {fba_data.shape}")
print(f"FBA data columns (first 10): {list(fba_data.columns)[:10]}")
print(f"Total FBA columns: {len(fba_data.columns)}")

# Check BioCyc_id overlap with preferred reactions
fba_columns = set(fba_data.columns[1:])  # Skip time column
preferred_biocyc = set(prefer_data["BioCyc_id"].dropna())
df_biocyc = set(df_filtered["BioCyc_id"].dropna())

overlap_preferred = fba_columns.intersection(preferred_biocyc)
overlap_df = fba_columns.intersection(df_biocyc)

print(
    f"Preferred BioCyc_id overlap: {len(overlap_preferred)} out of {len(preferred_biocyc)} preferred reactions"
)
print(
    f"Main data BioCyc_id overlap: {len(overlap_df)} out of {len(df_biocyc)} in main data"
)
print(f"Sample overlapping preferred BioCyc_ids: {list(overlap_preferred)[:5]}")

# Sort by type for better visualization
df_filtered = df_filtered.sort_values(["type", "flux_formatted"])

# Prepare data for box plots
flux_names = df_filtered["flux_formatted"].tolist()
type_names = df_filtered["type_name"].tolist()
types = df_filtered["type"].tolist()

# Get unique type names and create subplots
unique_types = df_filtered["type_name"].unique()
unique_types = [t for t in unique_types if pd.notna(t)]  # Remove NaN values
n_types = len(unique_types)

# Calculate subplot layout (prefer wider layout)
if n_types <= 3:
    ncols = n_types
    nrows = 1
elif n_types <= 6:
    ncols = 3
    nrows = 2
else:
    ncols = 3
    nrows = (n_types + 2) // 3

# Create figure with subplots
fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
if nrows == 1 and ncols == 1:
    axes = [axes]
elif nrows == 1:
    axes = axes
elif ncols == 1:
    axes = axes
else:
    axes = axes.flatten()


# Generate dynamic colors for different numbers of data groups
def generate_colors(n):
    """Generate n distinct colors"""
    if n <= 10:
        # Use predefined colors for small numbers
        base_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        return base_colors[:n]
    else:
        # Use colormap for larger numbers
        cmap = plt.cm.Set3
        return [cmap(i / n) for i in range(n)]


colors = generate_colors(n_groups)
labels = [f"{group.replace('data', 'Dataset ')}" for group in data_groups]

# FBA color (distinct from dataset colors)
fba_color = "#FF6B6B"  # Red color for FBA

# Create handles for legend
legend_handles = []
legend_labels = []
fba_legend_added = False  # Track if FBA legend has been added

# Plot each type in a separate subplot
for type_idx, type_name in enumerate(unique_types):
    if type_idx >= len(axes):
        break

    ax = axes[type_idx]

    # Filter data for this type
    type_data = df_filtered[df_filtered["type_name"] == type_name].copy()

    x_pos = 0
    x_ticks = []
    x_labels = []

    for idx, (_, row) in enumerate(type_data.iterrows()):
        # Calculate the width needed for this group (datasets + FBA)
        group_width = max(2, n_groups * 0.6 + 0.8)  # Extra space for FBA

        x_ticks.append(x_pos + group_width / 2)  # Center position for the group
        x_labels.append(row["flux_formatted"])

        # Plot data for each dataset dynamically
        for i, data_group in enumerate(data_groups):
            best_fit_col = f"{data_group}_best_fit"
            lb_col = f"{data_group}_LB95"
            ub_col = f"{data_group}_UB95"

            # Check if columns exist
            if best_fit_col not in df_filtered.columns:
                continue

            best_fit = row[best_fit_col]
            lb = row[lb_col] if lb_col in df_filtered.columns else np.nan
            ub = row[ub_col] if ub_col in df_filtered.columns else np.nan

            # Skip if no best_fit value or if it's not a valid number
            if pd.isna(best_fit) or not isinstance(best_fit, (int, float)):
                continue

            # Position for this data group
            pos = x_pos + i * 0.6

            # If missing UB or LB, draw as horizontal line
            if pd.isna(lb) or pd.isna(ub):
                line = ax.plot(
                    [pos - 0.2, pos + 0.2],
                    [best_fit, best_fit],
                    color=colors[i],
                    linewidth=3,
                )[0]
                # Add to legend only once
                if type_idx == 0 and idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(labels[i])
            else:
                # Create box plot data - ensure all values are numeric
                try:
                    lb_val = float(lb)
                    ub_val = float(ub)
                    best_fit_val = float(best_fit)

                    # Create a simple box representation
                    box_width = 0.4

                    # Box from LB to UB
                    rect = Rectangle(
                        (pos - box_width / 2, lb_val),
                        box_width,
                        ub_val - lb_val,
                        facecolor=colors[i],
                        alpha=0.7,
                        edgecolor="black",
                    )
                    ax.add_patch(rect)

                    # Median line
                    ax.plot(
                        [pos - box_width / 2, pos + box_width / 2],
                        [best_fit_val, best_fit_val],
                        color="black",
                        linewidth=2,
                    )

                    # Add to legend only once
                    if type_idx == 0 and idx == 0:
                        legend_handles.append(rect)
                        legend_labels.append(labels[i])

                except (ValueError, TypeError):
                    # If conversion fails, draw as horizontal line
                    line = ax.plot(
                        [pos - 0.2, pos + 0.2],
                        [best_fit, best_fit],
                        color=colors[i],
                        linewidth=3,
                    )[0]
                    # Add to legend only once
                    if type_idx == 0 and idx == 0:
                        legend_handles.append(line)
                        legend_labels.append(labels[i])

        # Add FBA box plot if data exists
        fba_values = get_fba_data_for_flux(
            row["flux_formatted"], biocyc_to_flux_mapping, fba_data
        )
        if fba_values is not None and len(fba_values) > 0:
            # Position for FBA box plot (after all datasets)
            fba_pos = x_pos + n_groups * 0.6 + 0.3

            # Create box plot for FBA data
            box_plot = ax.boxplot(
                fba_values,
                positions=[fba_pos],
                widths=0.4,
                patch_artist=True,
                showfliers=False,
            )

            # Color the box plot
            for patch in box_plot["boxes"]:
                patch.set_facecolor(fba_color)
                patch.set_alpha(0.7)
                # Add to legend only once globally
                if not fba_legend_added:
                    legend_handles.append(patch)
                    legend_labels.append("FBA")
                    fba_legend_added = True

            # Add individual points
            for value in fba_values:
                ax.plot(fba_pos, value, "o", color="darkred", markersize=4, alpha=0.6)

        x_pos += group_width + 0.5  # Spacing between flux groups

    # Set x-axis for this subplot
    ax.set_xlim(-0.5, x_pos - 0.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)

    # Set labels and title for this subplot
    ax.set_ylabel("Flux Value", fontsize=11)
    ax.set_xlabel("Metabolic Reactions", fontsize=11)
    ax.set_title(f"{type_name}", fontsize=12, fontweight="bold")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

# Add legend to the figure (outside the subplots)
if legend_handles:
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=10,
    )

# Hide unused subplots
for i in range(len(unique_types), len(axes)):
    axes[i].set_visible(False)

# Add overall title
fig.suptitle(
    f"E.coli Metabolic Flux Analysis: Comparison of {n_groups} Datasets + FBA by Pathway",
    fontsize=16,
    fontweight="bold",
    y=0.95,
)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.90, right=0.85)

# Show the plot
plt.savefig("flux_analysis_comparison.png", dpi=300)
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
for i, data_group in enumerate(data_groups):
    print(f"\n{labels[i]} Statistics:")
    best_fit_col = f"{data_group}_best_fit"

    if best_fit_col in df_filtered.columns:
        valid_data = df_filtered[
            df_filtered[best_fit_col].notna()
            & (df_filtered[best_fit_col].apply(lambda x: isinstance(x, (int, float))))
        ]
        print(f"  Number of valid flux measurements: {len(valid_data)}")
        if len(valid_data) > 0:
            print(f"  Mean flux value: {valid_data[best_fit_col].mean():.2f}")
            print(f"  Median flux value: {valid_data[best_fit_col].median():.2f}")
            print(f"  Min flux value: {valid_data[best_fit_col].min():.2f}")
            print(f"  Max flux value: {valid_data[best_fit_col].max():.2f}")
        else:
            print("  No valid data found")
    else:
        print(f"  Column {best_fit_col} not found in data")

# Print FBA statistics
print("\n=== FBA Statistics ===")
fba_flux_count = 0
total_flux_count = len(df_filtered)
total_fba_data_points = 0

for _, row in df_filtered.iterrows():
    fba_values = get_fba_data_for_flux(
        row["flux_formatted"], biocyc_to_flux_mapping, fba_data
    )
    if fba_values is not None and len(fba_values) > 0:
        fba_flux_count += 1
        total_fba_data_points += len(fba_values)

print(f"Total fluxes: {total_flux_count}")
print(f"Fluxes with FBA data: {fba_flux_count}")
print(f"Total FBA data points: {total_fba_data_points}")
print(f"FBA coverage: {fba_flux_count / total_flux_count * 100:.1f}%")

# Check which fluxes have FBA data
fluxes_with_fba = []
fluxes_without_fba = []

for _, row in df_filtered.iterrows():
    fba_values = get_fba_data_for_flux(
        row["flux_formatted"], biocyc_to_flux_mapping, fba_data
    )
    if fba_values is not None and len(fba_values) > 0:
        fluxes_with_fba.append(
            (row["flux_formatted"], row["BioCyc_id"], len(fba_values))
        )
    else:
        fluxes_without_fba.append((row["flux_formatted"], row["BioCyc_id"]))

print(f"\nFluxes with FBA data ({len(fluxes_with_fba)}):")
for flux_name, biocyc_id, n_points in fluxes_with_fba[:10]:  # Show first 10
    print(f"  - {flux_name} ({biocyc_id}) - {n_points} points")
if len(fluxes_with_fba) > 10:
    print(f"  ... and {len(fluxes_with_fba) - 10} more")

print(f"\nFluxes without FBA data ({len(fluxes_without_fba)}):")
for flux_name, biocyc_id in fluxes_without_fba[:10]:  # Show first 10
    print(f"  - {flux_name} ({biocyc_id})")
if len(fluxes_without_fba) > 10:
    print(f"  ... and {len(fluxes_without_fba) - 10} more")

# Create a second plot showing flux distributions by type (including FBA)
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

type_names_unique = df_filtered["type_name"].unique()
for i, type_name in enumerate(type_names_unique):
    if i >= 6:  # Only show first 6 types
        break

    type_data = df_filtered[df_filtered["type_name"] == type_name]

    # Prepare data for violin plot
    plot_data_type = []
    for _, row in type_data.iterrows():
        # Add dataset data
        for j, data_group in enumerate(data_groups):
            best_fit_col = f"{data_group}_best_fit"
            if best_fit_col in df_filtered.columns:
                best_fit_val = row[best_fit_col]
                if not pd.isna(best_fit_val) and isinstance(best_fit_val, (int, float)):
                    plot_data_type.append(
                        {
                            "dataset": labels[j],
                            "flux_value": float(best_fit_val),
                            "flux_name": row["flux_formatted"],
                        }
                    )

        # Add FBA data
        fba_values = get_fba_data_for_flux(
            row["flux_formatted"], biocyc_to_flux_mapping, fba_data
        )
        if fba_values is not None and len(fba_values) > 0:
            for fba_val in fba_values:
                plot_data_type.append(
                    {
                        "dataset": "FBA",
                        "flux_value": float(fba_val),
                        "flux_name": row["flux_formatted"],
                    }
                )

    if plot_data_type:
        plot_df = pd.DataFrame(plot_data_type)

        # Create violin plot
        sns.violinplot(data=plot_df, x="dataset", y="flux_value", ax=axes[i])
        axes[i].set_title(f"{type_name}", fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Flux Value")
        axes[i].set_xlabel("Dataset")
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis="x", rotation=45)

# Hide unused subplots
for i in range(len(type_names_unique), 6):
    axes[i].set_visible(False)

plt.suptitle(
    "Flux Distribution by Metabolic Pathway Type (Including FBA)",
    fontsize=16,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("flux_distribution_by_type.png", dpi=300)
plt.show()

# Print detailed flux information (including FBA)
print("\n=== Detailed Flux Information (Including FBA) ===")
for idx, row in df_filtered.iterrows():
    print(f"\nFlux: {row['flux_formatted']}")
    print(f"BioCyc_id: {row['BioCyc_id']}")
    print(f"Type: {row['type_name']}")

    # Print dataset information
    for i, data_group in enumerate(data_groups):
        best_fit_col = f"{data_group}_best_fit"
        lb_col = f"{data_group}_LB95"
        ub_col = f"{data_group}_UB95"

        if best_fit_col in df_filtered.columns:
            best_fit = row[best_fit_col]
            lb = row[lb_col] if lb_col in df_filtered.columns else np.nan
            ub = row[ub_col] if ub_col in df_filtered.columns else np.nan

            if not pd.isna(best_fit) and isinstance(best_fit, (int, float)):
                if pd.isna(lb) or pd.isna(ub):
                    print(f"  {labels[i]}: {best_fit:.2f} (point estimate)")
                else:
                    print(f"  {labels[i]}: {best_fit:.2f} [{lb:.2f}, {ub:.2f}]")
            else:
                print(f"  {labels[i]}: No data")
        else:
            print(f"  {labels[i]}: Column not found")

    # Print FBA information
    fba_values = get_fba_data_for_flux(
        row["flux_formatted"], biocyc_to_flux_mapping, fba_data
    )
    if fba_values is not None and len(fba_values) > 0:
        fba_array = np.array(fba_values)
        print(
            f"  FBA: Mean={fba_array.mean():.2f}, Median={np.median(fba_array):.2f}, "
            f"Min={fba_array.min():.2f}, Max={fba_array.max():.2f}, N={len(fba_values)}"
        )
    else:
        print("  FBA: No data available")

    if idx >= 10:  # Limit output for readability
        print(f"\n... and {len(df_filtered) - 11} more fluxes")
        break
