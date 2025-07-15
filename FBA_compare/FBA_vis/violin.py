import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
fba_data_normalized = normalize_preferred_fba_data(fba_data, prefer_data, reference_rxn)

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

# Filter for only preferred reactions
preferred_biocyc_ids = set(prefer_data["BioCyc_id"].dropna())
df_preferred = df_filtered[df_filtered["BioCyc_id"].isin(preferred_biocyc_ids)].copy()

print(f"Total fluxes: {len(df_filtered)}")
print(f"Preferred fluxes: {len(df_preferred)}")
print(f"Unique types in preferred fluxes: {df_preferred['type_name'].unique()}")

# Create mapping from BioCyc_id to flux information
biocyc_to_flux_mapping = {}
for _, row in df_preferred.iterrows():
    biocyc_id = row["BioCyc_id"]
    if pd.notna(biocyc_id):
        biocyc_to_flux_mapping[biocyc_id] = {
            "flux_formatted": row["flux_formatted"],
            "flux_original": row["flux"],
            "type_name": row["type_name"],
        }


# Function to get normalized FBA data for preferred fluxes
def get_normalized_fba_data_for_flux(biocyc_id, fba_data_normalized):
    """
    Get normalized FBA data for a specific BioCyc_id
    Returns list of values or None if not found
    """
    if biocyc_id in fba_data_normalized.columns:
        fba_values = fba_data_normalized[biocyc_id].dropna()
        return fba_values.tolist() if len(fba_values) > 0 else None
    return None


# Prepare data for violin plots
violin_data = []
for biocyc_id, flux_info in biocyc_to_flux_mapping.items():
    fba_values = get_normalized_fba_data_for_flux(biocyc_id, fba_data_normalized)
    if fba_values is not None and len(fba_values) > 0:
        for value in fba_values:
            violin_data.append(
                {
                    "BioCyc_id": biocyc_id,
                    "flux_name": flux_info["flux_formatted"],
                    "type_name": flux_info["type_name"],
                    "normalized_flux_value": value,
                }
            )

# Convert to DataFrame
violin_df = pd.DataFrame(violin_data)

print("\nViolin plot data summary:")
print(f"Total data points: {len(violin_df)}")
print(f"Unique fluxes: {violin_df['flux_name'].nunique()}")
print("Data points by type:")
for type_name in violin_df["type_name"].unique():
    count = len(violin_df[violin_df["type_name"] == type_name])
    unique_fluxes = violin_df[violin_df["type_name"] == type_name][
        "flux_name"
    ].nunique()
    print(f"  {type_name}: {count} data points, {unique_fluxes} unique fluxes")

# Create violin plots grouped by type
unique_types = violin_df["type_name"].unique()
n_types = len(unique_types)

# Calculate subplot layout
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
elif nrows == 1 or ncols == 1:
    axes = axes.flatten()
else:
    axes = axes.flatten()

# Color palette for better visualization
colors = plt.cm.Set3(np.linspace(0, 1, 12))

# Create violin plots for each type
for type_idx, type_name in enumerate(unique_types):
    if type_idx >= len(axes):
        break

    ax = axes[type_idx]

    # Filter data for this type
    type_data = violin_df[violin_df["type_name"] == type_name]

    # Get unique flux names for this type
    flux_names_in_type = type_data["flux_name"].unique()

    if len(flux_names_in_type) > 0:
        # Create violin plot
        sns.violinplot(
            data=type_data,
            x="flux_name",
            y="normalized_flux_value",
            ax=ax,
            palette=colors[: len(flux_names_in_type)],
            inner="box",
        )

        # Customize the plot
        ax.set_title(f"{type_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Normalized Flux Value (%)", fontsize=12)
        ax.set_xlabel("Metabolic Reactions", fontsize=12)

        # Rotate x-axis labels for better readability
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # Add statistics text
        stats_text = f"N fluxes: {len(flux_names_in_type)}\nN points: {len(type_data)}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=9,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(f"{type_name}", fontsize=14, fontweight="bold")

# Hide unused subplots
for i in range(len(unique_types), len(axes)):
    axes[i].set_visible(False)

# Add overall title
fig.suptitle(
    f"Normalized Preferred Flux Distribution by Metabolic Pathway\n(Normalized by {reference_rxn})",
    fontsize=16,
    fontweight="bold",
    y=0.95,
)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.88)

# Save and show the plot
plt.savefig("normalized_preferred_flux_violin_plots.png", dpi=300, bbox_inches="tight")
plt.show()

# Print detailed statistics
print("\n=== Detailed Statistics for Normalized Preferred Fluxes ===")
for type_name in unique_types:
    type_data = violin_df[violin_df["type_name"] == type_name]
    print(f"\n{type_name}:")
    print(f"  Number of unique fluxes: {type_data['flux_name'].nunique()}")
    print(f"  Total data points: {len(type_data)}")

    if len(type_data) > 0:
        print(
            f"  Mean normalized flux: {type_data['normalized_flux_value'].mean():.2f}%"
        )
        print(
            f"  Median normalized flux: {type_data['normalized_flux_value'].median():.2f}%"
        )
        print(f"  Std normalized flux: {type_data['normalized_flux_value'].std():.2f}%")
        print(f"  Min normalized flux: {type_data['normalized_flux_value'].min():.2f}%")
        print(f"  Max normalized flux: {type_data['normalized_flux_value'].max():.2f}%")

        # Show individual flux statistics
        print("  Individual flux statistics:")
        for flux_name in type_data["flux_name"].unique():
            flux_data = type_data[type_data["flux_name"] == flux_name][
                "normalized_flux_value"
            ]
            print(
                f"    {flux_name}: Mean={flux_data.mean():.2f}%, "
                f"Median={flux_data.median():.2f}%, N={len(flux_data)}"
            )

# Create a summary plot showing distribution across all types
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

# Create a combined violin plot
sns.violinplot(
    data=violin_df,
    x="type_name",
    y="normalized_flux_value",
    ax=ax2,
    palette="Set2",
    inner="box",
)

ax2.set_title(
    "Overall Distribution of Normalized Preferred Fluxes by Pathway Type",
    fontsize=16,
    fontweight="bold",
)
ax2.set_ylabel("Normalized Flux Value (%)", fontsize=12)
ax2.set_xlabel("Metabolic Pathway Type", fontsize=12)
ax2.tick_params(axis="x", rotation=45, labelsize=11)
ax2.tick_params(axis="y", labelsize=11)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)

# Add overall statistics
total_stats = f"Total fluxes: {violin_df['flux_name'].nunique()}\n"
total_stats += f"Total data points: {len(violin_df)}\n"
total_stats += f"Overall mean: {violin_df['normalized_flux_value'].mean():.2f}%\n"
total_stats += f"Overall median: {violin_df['normalized_flux_value'].median():.2f}%"

ax2.text(
    0.02,
    0.98,
    total_stats,
    transform=ax2.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    fontsize=10,
)

plt.tight_layout()
plt.savefig("normalized_preferred_flux_summary.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== Summary ===")
print(f"Successfully created violin plots for {len(unique_types)} pathway types")
print(f"Total preferred fluxes visualized: {violin_df['flux_name'].nunique()}")
print(f"Total data points: {len(violin_df)}")
print(f"Reference reaction used for normalization: {reference_rxn}")
