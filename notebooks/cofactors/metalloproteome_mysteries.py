# ruff: noqa
# %%
import matplotlib as mpl
import sys
import os

sys.path.append(os.path.abspath("../.."))  # Adds the repo root directory to Python path


mpl.rcParams["figure.dpi"] = 150

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import warnings
from scipy.sparse import csr_matrix
from scipy import stats
import scipy.ndimage as ndimage


sns.set(style="darkgrid", palette="Set2", context="paper")
warnings.filterwarnings(action="ignore")

RELOAD_ACCESSION_DATA = False


# %%
def read_names(file_path):
    return np.genfromtxt(file_path, dtype=str, delimiter="\n")


complex_ids = read_names("data/complex_ids.txt")
monomer_ids = read_names("data/protein_ids.txt")
cofactor_ids = read_names("data/cofactor_ids.txt")
element_ids = read_names("data/element_ids.txt")
protein_ids = complex_ids.tolist() + monomer_ids.tolist()


# Reading the matrix CSV files
def read_matrix(file_path, sparse=False):
    # add sparse matrix reading
    if sparse:
        return csr_matrix(pl.read_csv(file_path, has_header=False).to_numpy())
    else:
        return pl.read_csv(file_path, has_header=False).to_numpy()


#
# C = read_matrix("data/C_matrix.csv", sparse=True)
# P = read_matrix("data/P_matrix.csv", sparse=True)
# E = read_matrix("data/E_matrix.csv", sparse=True)
Tree = read_matrix("data/tree_matrix.csv", sparse=True)
tree_ids = complex_ids.tolist() + monomer_ids.tolist() + cofactor_ids.tolist()

total_counts_min = read_matrix("data/counts.csv")

# %% [markdown]
# ## import conversion table

# %%
conversion_df = pl.read_csv("external_data/metalloproteome_exp_conversion.csv")

# get all unique values of End and Index
end_values = conversion_df["End"].unique().to_list()
index_values = conversion_df["Index"].unique().to_list()

# give each unique value a unique number
end_dict = {end: i for i, end in enumerate(end_values)}
index_dict = {index: i for i, index in enumerate(index_values)}

# create a new column for the unique value
conversion_df = conversion_df.with_columns(x=pl.col("End").replace(end_dict))
conversion_df = conversion_df.with_columns(y=pl.col("Index").replace(index_dict))

# drop start end index
conversion_df = conversion_df.drop(["Start", "End", "Index"])

conversion_df

# %% [markdown]
#
# ## Import protein data

# %%
import requests
import xml.etree.ElementTree as ET
import time

if RELOAD_ACCESSION_DATA:
    for accession_id in prot_df["Accession Number"]:
        if accession_id not in gene_dict:
            # add a 0.1 second delay between each request
            time.sleep(1)

            # Step 1: Use elink to find the Gene ID associated with the protein accession
            elink_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=protein&db=gene&id={accession_id}&retmode=xml"
            elink_response = requests.get(elink_url)
            elink_tree = ET.fromstring(elink_response.content)

            # Extract the Gene ID from the elink response
            gene_id = None
            for linkset in elink_tree.findall(".//LinkSetDb"):
                if (
                    linkset.find("LinkName").text == "protein_gene"
                    and linkset.find(".//Id") is not None
                ):
                    gene_id = linkset.find(".//Id").text
                    break

            # Step 2: Use esummary to get the gene information based on the Gene ID
            if gene_id:
                esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id={gene_id}&retmode=xml"
                esummary_response = requests.get(esummary_url, headers=headers)
                esummary_tree = ET.fromstring(esummary_response.content)

                # Extract the gene name
                gene_name = esummary_tree.find(".//Name").text
                print(f"Gene name: {gene_name}")
            else:
                print("Gene ID not found.")

            gene_dict[accession_id] = gene_name

# %%
# load gene df
gene_df = pl.read_csv("data/accession_gene_dict.csv")
gene_df

# convert to dict
gene_dict = dict(zip(gene_df["Accession Number"], gene_df["Gene Name"]))

gene_df

# %% [markdown]
# ## Change column to gene name

# %%
# Load the data
prot_df_raw = pl.read_csv("external_data/metalloproteome_proteins.csv")

# drop "#", "Visible?", "Starred?", "Molecular Weight"
prot_df = prot_df_raw.drop(
    ["#", "Visible?", "Starred?", "Molecular Weight", "Taxonomy"]
)

prot_df.head(5)

# change column to gene name
prot_df = prot_df.with_columns(Gene=pl.col("Accession Number").replace(gene_dict))

# drop Identified Proteins (1235), Accession Number
prot_df = prot_df.drop(["Identified Proteins (1235)", "Accession Number"])

# drop duplicate rows with the same gene name
prot_df = prot_df.unique(subset=["Gene"], maintain_order=True)

# transpose and keep gene as column names
prot_exp_names = list(prot_df.columns)[:-1]
gene_names = prot_df["Gene"].to_list()

# %%
# transpose and keep gene as column names
prot_df = prot_df.drop(["Gene"]).transpose()
prot_df.columns = gene_names

# add a new column, "Experiment"
prot_df = prot_df.with_columns(pl.Series(name="Experiment", values=prot_exp_names))

# join with the conversion table
prot_df = prot_df.join(conversion_df, on="Experiment")

prot_df

# %% [markdown]
# ## Load the metal data

# %%
# load metal data
metal_df = pl.read_csv("external_data/metalloproteome_metals.csv")

# drop columns that started with "_dupli"
metal_df = metal_df.drop([col for col in metal_df.columns if col.startswith("_dupli")])

# drop columns without names
metal_df = metal_df.drop([col for col in metal_df.columns if len(col) < 2])

# for all float cols, subtract minimum value
# for col in metal_df.columns:
#     if metal_df[col].dtype == pl.Float64:
#         metal_df = metal_df.with_columns(pl.Series(name=col, values=metal_df[col] - metal_df[col].min()))

metal_df

# %%
metal_column_name = "60Ni"
metal_name = "".join([char for char in metal_column_name if not char.isdigit()])
gene_name = "metE"

# %%
n_samples = metal_df.shape[0]
linspace = np.linspace(0, 1, n_samples)

sns.lineplot(x=linspace, y=metal_column_name, data=metal_df)

# plot rpmE col
sns.lineplot(x=linspace, y=gene_name, data=prot_df, alpha=0.3)

# %% [markdown]
# # Join the metalloproteome data with the conversion table

# %%
# join the data
metal_join_df = metal_df.join(conversion_df, on="Experiment")

# filter out edges
# if x or y are 0, or x=17 or y=31, remove them.
metal_join_df = metal_join_df.filter(
    (pl.col("x") > 1) & (pl.col("y") > 0) & (pl.col("x") < 17) & (pl.col("y") < 29)
)
prot_df = prot_df.filter(
    (pl.col("x") > 1) & (pl.col("y") > 0) & (pl.col("x") < 17) & (pl.col("y") < 29)
)

metal_join_df

# %%
n_samples = metal_join_df.shape[0]
linspace = np.linspace(0, 1, n_samples)

sns.lineplot(x=linspace, y=metal_column_name, data=metal_join_df)

# plot rpmE col
sns.lineplot(x=linspace, y=gene_name, data=prot_df, alpha=0.3)

# %% [markdown]
# # Import simulation data and conversion tables

# %%
# make protein-element mapping
elements = ["FE", "ZN", "MN", "CU", "MO", "NI"]
element_indices = [np.where(element_ids == element)[0][0] for element in elements]


# get name conversion table
gene_name_table = pl.read_csv(
    "external_data/ecocyc_name_conversion.txt", separator="\t"
)

# rename cols to "Protein ID", "EcoCyc ID", "Accession Number", "Gene name"
gene_name_table.columns = ["Protein ID", "EcoCyc ID", "Accession Number", "Gene name"]

# get corresponding gene name for each protein
gene_name_table = gene_name_table.filter(pl.col("Protein ID").is_in(monomer_ids))

# make a mapping
gene_to_prot_dict = dict(
    zip(gene_name_table["Gene name"], gene_name_table["Protein ID"])
)
prot_to_gene_dict = dict(
    zip(gene_name_table["Protein ID"], gene_name_table["Gene name"])
)

# convert gene_names list to protein id list
detected_protein_ids = [
    gene_to_prot_dict[gene_name]
    for gene_name in gene_names
    if gene_name in gene_to_prot_dict
]
# get indices of genes in gene dict
gene_indices = [
    gene_names.index(gene_name)
    for gene_name in gene_names
    if gene_name in gene_to_prot_dict
]

# for each protein id, index it in the monomer_ids
detected_protein_idx = [
    monomer_ids.tolist().index(protein_id) for protein_id in detected_protein_ids
]


# %% [markdown]
# # Plot unprocessed data

# %%
from matplotlib.colors import LogNorm

plt.figure(figsize=(10, 5))

# heatmap of 56Fe data against x and y
sns.heatmap(
    metal_join_df.pivot(
        values=metal_column_name,
        index="x",
        on="y",
    )[:, 1:],
    cmap="Grays",
    square=True,
    cbar=True,
    norm=LogNorm(),
    cbar_kws={"label": f"{metal_name} (part per billion)"},
)

# change x and y axis labels
plt.ylabel("Size-exclusion chromatography fraction")
plt.xlabel("Ion chromatography fraction")


plt.xlim([0, 27])
plt.ylim([15, 0])

plt.tight_layout()

# save as png and svg
# plt.savefig("figures/zn_bare.png")
# plt.savefig("figures/zn_bare.svg")


# %%
from matplotlib.colors import LogNorm

plt.figure(figsize=(10, 5))

# heatmap of 56Fe data against x and y
sns.heatmap(
    metal_join_df.pivot(
        values=metal_column_name,
        index="x",
        on="y",
    )[:, 1:],
    cmap="Grays",
    square=True,
    cbar=True,
    norm=LogNorm(),
    cbar_kws={"label": f"{metal_name} (part per billion)"},
)
sns.heatmap(
    prot_df.pivot(
        values=gene_name,
        index="x",
        on="y",
    )[:, 1:],
    mask=(
        prot_df.pivot(
            values=gene_name,
            index="x",
            on="y",
        )
        < 50
    ).to_numpy()[:, 1:],
    cmap="Blues",
    square=True,
    cbar=False,
    vmin=0,
    cbar_kws={"label": f"{gene_name} (spectral count)"},
)

# change x and y axis labels
plt.ylabel("Size-exclusion chromatography fraction")
plt.xlabel("Ion chromatography fraction")

plt.xlim([0, 27])
plt.ylim([15, 0])

plt.tight_layout()

# save as png and svg
# plt.savefig("figures/zn_mete.png")
# plt.savefig("figures/zn_mete.svg")


# %% [markdown]
# # How many proteins in each well?

# %%
print(prot_df.shape)
print(prot_df.head())
# 1230 proteins

# %%
threshold = 0

protein_counts = []
for x in range(2, 17):  # x: 2-16
    for y in range(1, 29):  # y: 1-28
        well_data = prot_df.filter((pl.col("x") == x) & (pl.col("y") == y))
        if well_data.shape[0] > 0:
            count = (
                well_data.select(pl.exclude(["Experiment", "x", "y"])) > threshold
            ).sum_horizontal()[0]
            protein_counts.append({"x": x, "y": y, "count": count})

counts_df = pl.DataFrame(protein_counts)
counts_pivot = counts_df.pivot(
    values="count",
    index="x",
    on="y",
)

plt.figure(figsize=(10, 5))

sns.heatmap(
    counts_pivot[:, 1:],
    cmap="viridis",
    square=True,
    cbar=True,
    annot=True,
    fmt="g",
    cbar_kws={"label": f"Number of proteins (intensity > {threshold})"},
)

plt.ylabel("Size-exclusion chromatography fraction")
plt.xlabel("Ion chromatography fraction")

plt.xlim([0, 27])
plt.ylim([15, 0])

plt.tight_layout()

# %% [markdown]
# # Determing gene with highest overlap

# %%
# metal_column_name = "66Zn"
# metal_column_name = "182W"
# metal_column_name = "95Mo"
metal_column_name = "60Ni"

metal_name = "".join([char for char in metal_column_name if not char.isdigit()])


# %%
def find_intensity_cutoff(protein_values, gene, method="percentile"):
    nonzero_values = protein_values[protein_values > 0]
    if len(nonzero_values) == 0:
        return 0

    if method == "percentile":
        gene_matrix = prot_df.pivot(
            values=gene,
            index="x",
            on="y",
        )[:, 1:].to_numpy()
        gene_nonzero = gene_matrix.flatten()[gene_matrix.flatten() > 0]
        return np.percentile(gene_nonzero, 80)

    elif method == "valley":
        # Find the valley between first two peaks using KDE
        if len(nonzero_values) < 4:  # Need minimum points for KDE
            return np.median(nonzero_values)
        kde = stats.gaussian_kde(nonzero_values)
        x_range = np.linspace(nonzero_values.min(), nonzero_values.max(), 100)
        kde_values = kde(x_range)
        # Find first valley after first peak
        peaks = (
            np.where(
                (kde_values[1:-1] > kde_values[:-2])
                & (kde_values[1:-1] > kde_values[2:])
            )[0]
            + 1
        )
        if len(peaks) > 1:
            return x_range[peaks[0]]
        return np.median(nonzero_values)

    elif method == "std":
        # Use mean + 1 std of log values to handle skewed distributions
        log_values = np.log1p(nonzero_values)  # log1p handles zeros
        cutoff = np.exp(log_values.mean() + log_values.std()) - 1
        return cutoff


masked_df = prot_df.clone()
metal_matrix = metal_join_df.pivot(
    values=metal_column_name,
    index="x",
    on="y",
)[:, 1:].to_numpy()
metal_nonzero = metal_matrix.flatten()[metal_matrix.flatten() > 0]
metal_threshold = np.percentile(metal_nonzero, 75)

all_results = []
metal_proteins = []
overlap_threshold = 0.75
for gene in gene_names:
    if gene not in ["Experiment", "x", "y"]:
        protein_matrix = prot_df.pivot(
            values=gene,
            index="x",
            on="y",
        )[:, 1:].to_numpy()
        protein_values = protein_matrix.flatten()

        # Get cutoff specific to this gene
        cutoff = find_intensity_cutoff(protein_values, gene, method="percentile")

        # Create mask: keep only values above cutoff
        mask = protein_matrix <= cutoff
        protein_matrix[mask] = 0

        # Update the dataframe with masked values
        masked_df = masked_df.with_columns(
            pl.Series(name=gene, values=protein_matrix.flatten())
        )
        if protein_matrix.sum() > 0:  # Only analyze if protein has signal
            # Create boolean masks
            protein_mask = protein_matrix > 0
            high_metal_mask = metal_matrix > metal_threshold

            # Calculate overlap using boolean masks
            overlap_ratio = (protein_mask & high_metal_mask).sum() / protein_mask.sum()

            all_results.append(
                {
                    "gene": gene,
                    "overlap_ratio": overlap_ratio,
                    "protein_wells": protein_mask.sum(),
                    "overlap_wells": (protein_mask & high_metal_mask).sum(),
                }
            )

            if overlap_ratio > overlap_threshold:
                metal_proteins.append(gene)

print(f"{metal_name} associated proteins: {metal_proteins}")

# %%
score_used = "good_to_bad_ratio"
normalize = False


def calculate_overlap_score(protein_data, metal_data):
    # Focus on highest intensity regions
    metal_threshold = np.percentile(metal_data, 95)
    protein_threshold = np.percentile(protein_data, 95)

    # Get masks for high/low intensity regions
    high_metal = metal_data > metal_threshold
    high_protein = protein_data > protein_threshold

    # Penalize regions with high protein but low metal
    penalty_regions = high_protein & (metal_data < np.percentile(metal_data, 50))

    # Original scoring approaches with penalties
    peak_overlap = high_metal & high_protein

    # Approach 1: Peak product with penalty
    score1 = np.sum(protein_data[peak_overlap] * metal_data[peak_overlap])
    penalty1 = np.sum(protein_data[penalty_regions])
    score1 = score1 - penalty1  # Adjust 0.5 to control penalty strength

    # Approach 2: Max combined signal with penalty
    combined_signal = protein_data * metal_data
    score2 = np.max(combined_signal)
    score2 = score2 - np.sum(protein_data[penalty_regions])

    # Approach 3: Ratio of good overlap to bad overlap
    good_signal = np.sum(protein_data[peak_overlap] * metal_data[peak_overlap])
    bad_signal = np.sum(protein_data[penalty_regions])
    score3 = good_signal / (bad_signal + 1)  # Add 1 to avoid division by zero

    return {
        "penalized_peak_product": score1,
        "penalized_max_signal": score2,
        "good_to_bad_ratio": score3,
    }


def find_metal_associated_proteins(metal_column_name, score_type, normalize):
    metal_matrix = metal_join_df.pivot(
        values=metal_column_name,
        index="x",
        on="y",
    )[:, 1:].to_numpy()

    if normalize:
        threshold_protein = 50
        protein_counts = []
        for x in range(2, 17):
            for y in range(1, 29):
                well_data = prot_df.filter((pl.col("x") == x) & (pl.col("y") == y))
                if well_data.shape[0] > 0:
                    count = (
                        well_data.select(pl.exclude(["Experiment", "x", "y"]))
                        > threshold_protein
                    ).sum(axis=1)[0]
                    protein_counts.append({"x": x, "y": y, "count": max(count, 1)})

        counts_df = pl.DataFrame(protein_counts)
        counts_pivot = counts_df.pivot(
            values="count",
            index="x",
            on="y",
        )[:, 1:].to_numpy()
        normalized_metal = metal_matrix / counts_pivot
        metal_matrix = normalized_metal

    results = []
    for gene in gene_names:
        protein_matrix = prot_df.pivot(
            values=gene,
            index="x",
            on="y",
        )[:, 1:].to_numpy()

        if protein_matrix.size == 0 or protein_matrix.max() == 0:
            continue

        scores = calculate_overlap_score(protein_matrix, metal_matrix)
        score = scores[score_type]

        if score > 0:
            results.append(
                {
                    "gene": gene,
                    "score": score,
                    "max_signal": protein_matrix.max(),
                    "total_points": (protein_matrix > 2).sum(),
                    "all_scores": scores,  # Optional: keep all scores for reference
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)

    # Print results with visualization for top hits
    print(f"\nTop {top_n} proteins associated with {metal_column_name}:")
    for result in [r for r in results if r["gene"].startswith("y")][:top_n]:
        # for result in results[:top_n]:
        print(f"\nGene: {result['gene']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Max signal: {result['max_signal']:.1f}")
        print(f"Total signal points: {result['total_points']}")

        if result["max_signal"] > 0:  # Additional check
            plt.figure(figsize=(5, 3))

            sns.heatmap(
                metal_join_df.pivot(
                    values=metal_column_name,
                    index="x",
                    on="y",
                )[:, 1:],
                cmap="Grays",
                square=True,
                cbar=True,
                norm=LogNorm(),
                cbar_kws={"label": f"{metal_column_name} (ppb)"},
            )

            protein_matrix = prot_df.pivot(
                values=result["gene"],
                index="x",
                on="y",
            )[:, 1:].to_numpy()
            sns.heatmap(
                protein_matrix,
                mask=(protein_matrix < 2),  # Only show strong signals
                cmap="Blues",
                square=True,
                cbar=True,
                alpha=0.5,
                cbar_kws={"label": "Protein Signal"},
            )

            plt.title(f"{result['gene']} (Score: {result['score']:.3f})")
            plt.show()

    return results


top_n = 10
results = find_metal_associated_proteins(
    metal_column_name, score_type=score_used, normalize=normalize
)
print(results[:top_n])


def get_gene_score(gene_name, metal_column_name, score_type, normalize):
    metal_matrix = metal_join_df.pivot(
        values=metal_column_name,
        index="x",
        on="y",
    )[:, 1:].to_numpy()
    protein_matrix = prot_df.pivot(
        values=gene_name,
        index="x",
        on="y",
    )[:, 1:].to_numpy()

    if normalize:
        threshold_protein = 50
        protein_counts = []
        for x in range(2, 17):
            for y in range(1, 29):
                well_data = prot_df.filter((pl.col("x") == x) & (pl.col("y") == y))
                if well_data.shape[0] > 0:
                    count = (
                        well_data.select(pl.exclude(["Experiment", "x", "y"]))
                        > threshold_protein
                    ).sum(axis=1)[0]
                    protein_counts.append({"x": x, "y": y, "count": max(count, 1)})

        counts_df = pl.DataFrame(protein_counts)
        counts_pivot = counts_df.pivot("count", "x", "y")[:, 1:].to_numpy()
        normalized_metal = metal_matrix / counts_pivot
        metal_matrix = normalized_metal

    score = calculate_overlap_score(protein_matrix, metal_matrix)
    score = score[score_type]

    print(f"\nResults for {gene_name}:")
    print(f"Score: {score:.3f}")
    print(f"Max signal: {protein_matrix.max():.1f}")
    print(f"Total signal points: {(protein_matrix > 2).sum()}")

    # Also show the visualization
    plt.figure(figsize=(5, 3))
    sns.heatmap(
        metal_join_df.pivot(
            values=metal_column_name,
            index="x",
            on="y",
        )[:, 1:],
        cmap="Grays",
        square=True,
        cbar=True,
        norm=LogNorm(),
    )
    sns.heatmap(
        protein_matrix,
        mask=(protein_matrix < 2),
        cmap="Blues",
        square=True,
        cbar=True,
        alpha=0.5,
    )
    plt.title(f"{gene_name} (Score: {score:.3f})")
    plt.show()

    return score


hya_genes = [col for col in prot_df.columns if col.startswith("mog")]
print(hya_genes)

score = get_gene_score(
    "yggX", metal_column_name, score_type=score_used, normalize=normalize
)
print(score)

# %%
from matplotlib.colors import LogNorm

plt.figure(figsize=(15, 4))

# heatmap of 56Fe data against x and y
sns.heatmap(
    metal_join_df.pivot(
        values="66Zn",
        index="x",
        on="y",
    )[:, 1:],
    cmap="Grays",
    square=True,
    cbar=True,
    norm=LogNorm(),
)
sns.heatmap(
    prot_df.pivot(
        values="metE",
        index="x",
        on="y",
    )[:, 1:],
    mask=(
        prot_df.pivot(
            values="metE",
            index="x",
            on="y",
        )
        < 50
    ).to_numpy()[:, 1:],
    cmap="Blues",
    square=True,
    cbar=False,
    vmin=0,
)
sns.heatmap(
    prot_df.pivot(
        values="rpmE",
        index="x",
        on="y",
    )[:, 1:],
    mask=(
        prot_df.pivot(
            values="rpmE",
            index="x",
            on="y",
        )
        < 1
    ).to_numpy()[:, 1:],
    cmap="Greens",
    square=True,
    vmin=0,
    cbar=False,
)
sns.heatmap(
    prot_df.pivot(
        values="rpoC",
        index="x",
        on="y",
    )[:, 1:],
    mask=(
        prot_df.pivot(
            values="rpoC",
            index="x",
            on="y",
        )
        < 100
    ).to_numpy()[:, 1:],
    cmap="Reds",
    square=True,
    vmin=0,
    cbar=False,
)
sns.heatmap(
    prot_df.pivot(
        values="pyrI",
        index="x",
        on="y",
    )[:, 1:],
    mask=(
        prot_df.pivot(
            values="pyrI",
            index="x",
            on="y",
        )
        < 80
    ).to_numpy()[:, 1:],
    cmap="Purples",
    square=True,
    vmin=0,
    cbar=False,
)

# change x and y axis labels
plt.ylabel("Size-exclusion chromatography fraction (")
plt.xlabel("Ion chromatography fraction")

# make xticklabels integers 2-18
# plt.xticks(np.arange(1, 14, 2))
# plt.xlim([0, 28])
# plt.ylim([0, 14])

# save as png and svg
plt.savefig("figures/metal_splashes.png")
plt.savefig("figures/metal_splashes.svg")


# %% [markdown]
# # Plot grid of heatmaps

# %%
# for 9 random proteins, plot a grid of small heatmaps
random_proteins = np.random.choice(gene_names, 9)

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
for i, ax in enumerate(axs.flatten()):
    sns.heatmap(
        prot_df.pivot(
            random_proteins[i],
            "x",
            "y",
        )[:, 1:],
        cmap="viridis",
        square=True,
        cbar=True,
        ax=ax,
    )
    ax.set_title(random_proteins[i])
    ax.set_xticks([])
    ax.set_yticks([])

# %% [markdown]
#     ## Filter out edges

# %%
# For each protein, pivot and do some filtering

for protein in gene_names:
    prot_pivot = prot_df.pivot(
        protein,
        "x",
        "y",
    ).to_numpy()[:, 1:]

    # create boolean array
    prot_mask = prot_pivot > 0

    # find number of neighbours per well
    footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    f = lambda x: x.sum()

    results = ndimage.generic_filter(
        prot_mask, f, footprint=footprint, mode="constant", cval=0
    )

    # get AND of the two
    prot_pivot_true_mask = prot_mask & results

    # dont zero if it would remove all the data
    if prot_pivot_true_mask.sum() > 0:
        prot_pivot[~prot_pivot_true_mask] = 0

    # set column as flattened pivot
    prot_df = prot_df.with_columns(pl.Series(name=protein, values=prot_pivot.flatten()))


# %%
from matplotlib.colors import LogNorm

plt.figure(figsize=(10, 5))

# heatmap of 56Fe data against x and y
sns.heatmap(
    metal_join_df.pivot(
        "66Zn",
        "x",
        "y",
    )[:, 1:],
    cmap="Grays",
    square=True,
    cbar=True,
    norm=LogNorm(),
    cbar_kws={"label": "Zn (part per billion)"},
)
sns.heatmap(
    prot_df.pivot(
        "rpmE",
        "x",
        "y",
    )[:, 1:],
    mask=(
        prot_df.pivot(
            "rpmE",
            "x",
            "y",
        )
        < 1
    ).to_numpy()[:, 1:],
    cmap="Greens",
    square=True,
    cbar=False,
    vmin=0,
    cbar_kws={"label": "RpmE (spectral count)"},
)

# change x and y axis labels
plt.ylabel("Size-exclusion chromatography fraction")
plt.xlabel("Ion chromatography fraction")

plt.xlim([0, 27])
plt.ylim([15, 0])

plt.tight_layout()

# save as png and svg
# plt.savefig("figures/zn_rpme.png")
# plt.savefig("figures/zn_rpme.svg")


# %% [markdown]
# # Check out some data

# %%
prot_df.filter((pl.col("y") == 7) & (pl.col("x") == 13)).melt(
    id_vars=["Experiment", "x", "y"], variable_name="Gene", value_name="Intensity"
)

# %%
from matplotlib.colors import LogNorm

plt.figure(figsize=(10, 5))

f, ax = plt.subplots(3, 1, figsize=(5, 5))

gene = "cyoA"
gene2 = "cueO"
metal = "63Cu"

# heatmap of 56Fe data against x and y
sns.heatmap(
    metal_join_df.pivot(
        metal,
        "x",
        "y",
    )[:, 1:],
    cmap="Grays",
    square=True,
    cbar=True,
    norm=LogNorm(),
    cbar_kws={"label": "Zn (part per billion)"},
    ax=ax[0],
)

sns.heatmap(
    metal_join_df.pivot(
        metal,
        "x",
        "y",
    )[:, 1:],
    cmap="Grays",
    square=True,
    cbar=True,
    norm=LogNorm(),
    cbar_kws={"label": "Zn (part per billion)"},
    ax=ax[1],
)
sns.heatmap(
    prot_df.pivot(
        gene,
        "x",
        "y",
    )[:, 1:],
    mask=(
        prot_df.pivot(
            gene,
            "x",
            "y",
        )
        < 1
    ).to_numpy()[:, 1:],
    cmap="Blues",
    square=True,
    cbar=False,
    vmin=0,
    cbar_kws={"label": f"{gene} (spectral count)"},
    ax=ax[1],
)

sns.heatmap(
    metal_join_df.pivot(
        metal,
        "x",
        "y",
    )[:, 1:],
    cmap="Grays",
    square=True,
    cbar=True,
    norm=LogNorm(),
    cbar_kws={"label": "Zn (part per billion)"},
    ax=ax[2],
)
sns.heatmap(
    prot_df.pivot(
        gene2,
        "x",
        "y",
    )[:, 1:],
    mask=(
        prot_df.pivot(
            gene2,
            "x",
            "y",
        )
        < 1
    ).to_numpy()[:, 1:],
    cmap="Blues",
    square=True,
    cbar=False,
    vmin=0,
    cbar_kws={"label": f"{gene} (spectral count)"},
    ax=ax[2],
)

# # change x and y axis labels
# ax=ax[0].ylabel("Size-exclusion chromatography fraction")
# ax=ax[0].xlabel("Ion chromatography fraction")
#
#
#
# ax=ax[0].xlim([0, 27])
# ax=ax[0].ylim([15,0])

plt.tight_layout()

# save as png and svg
# plt.savefig("figures/zn_bare.png")
# plt.savefig("figures/zn_bare.svg")


# %% [markdown]
# # Identify protein responsible for unknown peak

# %%
description_df = prot_df_raw.select(["Identified Proteins (1235)", "Accession Number"])

description_df = gene_df.join(description_df, on="Accession Number")

# rename Identified Proteins (1235) to "Description"
description_df = description_df.rename(
    {"Identified Proteins (1235)": "Description", "Gene Name": "Gene"}
)

# Add new column, "is unknown function",
description_df = description_df.with_columns(
    pl.Series(
        name="is unknown function",
        values=[1 if desc.startswith("y") else 0 for desc in description_df["Gene"]],
    )
)

description_df.filter(pl.col("is unknown function") == 1)

# %%


# %%
