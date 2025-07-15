import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore")

# Read the CSV data
df = pd.read_csv("merged_flux_data_complete.csv")


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
    # elif '(exch)' in flux_name:
    #     flux_name = flux_name.replace('(exch)', '').strip()

    return flux_name


# Apply flux name formatting
df_filtered["flux_formatted"] = df_filtered["flux"].apply(format_flux_name)

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
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
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
    legend_added = False

    for idx, (_, row) in enumerate(type_data.iterrows()):
        x_ticks.append(
            x_pos + (n_groups - 1) * 0.6 / 2
        )  # Center position for the group
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
                ax.plot(
                    [pos - 0.2, pos + 0.2],
                    [best_fit, best_fit],
                    color=colors[i],
                    linewidth=3,
                    label=labels[i] if not legend_added else "",
                )
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

                    # Add label only for the first occurrence
                    if not legend_added:
                        rect.set_label(labels[i])

                except (ValueError, TypeError):
                    # If conversion fails, draw as horizontal line
                    ax.plot(
                        [pos - 0.2, pos + 0.2],
                        [best_fit, best_fit],
                        color=colors[i],
                        linewidth=3,
                        label=labels[i] if not legend_added else "",
                    )

        # Mark legend as added after first flux
        if idx == 0:
            legend_added = True

        x_pos += max(
            2, n_groups * 0.6 + 0.5
        )  # Adjust spacing based on number of groups

    # Set x-axis for this subplot
    ax.set_xlim(-0.5, x_pos - 0.5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)

    # Set labels and title for this subplot
    ax.set_ylabel("Flux Value", fontsize=11)
    ax.set_xlabel("Metabolic Reactions", fontsize=11)
    ax.set_title(f"{type_name}", fontsize=12, fontweight="bold")

    # Add legend only to the first subplot
    if type_idx == 0:
        ax.legend(loc="upper right", fontsize=10)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for i in range(len(unique_types), len(axes)):
    axes[i].set_visible(False)

# Add overall title
fig.suptitle(
    f"E.coli Metabolic Flux Analysis: Comparison of {n_groups} Datasets by Pathway",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.93)

plt.savefig("flux_box_plot.png", dpi=300, bbox_inches="tight")
# Show the plot
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

# Create a second plot showing flux distributions by type
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

    if plot_data_type:
        plot_df = pd.DataFrame(plot_data_type)

        # Create violin plot
        sns.violinplot(data=plot_df, x="dataset", y="flux_value", ax=axes[i])
        axes[i].set_title(f"{type_name}", fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Flux Value")
        axes[i].set_xlabel("Dataset")
        axes[i].grid(True, alpha=0.3)

# Hide unused subplots
for i in range(len(type_names_unique), 6):
    axes[i].set_visible(False)

plt.suptitle(
    "Flux Distribution by Metabolic Pathway Type", fontsize=16, fontweight="bold"
)
plt.tight_layout()
plt.show()

# Print detailed flux information
print("\n=== Detailed Flux Information ===")
for idx, row in df_filtered.iterrows():
    print(f"\nFlux: {row['flux_formatted']}")
    print(f"Type: {row['type_name']}")
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
