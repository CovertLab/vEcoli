import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def dimensionality_reduction_and_clustering(
    csv_path: str,
    zero_threshold: float = 0.999,
    eps: float = 1e-9,
    n_pca_components: int = 2,
    tsne_perplexity: float = 30.0,
    n_clusters: int = 4,
    drop_time0: bool = True,
):
    """
    Performs PCA, t-SNE on time points and clusters reactions using KMeans.

    Parameters
    ----------
    csv_path : str
        Path to the time-series flux CSV. First column must be 'time'.
    zero_threshold : float
        Fraction of zeros above which a reaction is considered inactive.
    eps : float
        Tolerance for zero comparison (abs(flux) <= eps).
    n_pca_components : int
        Number of PCA components (2 or 3).
    tsne_perplexity : float
        Perplexity parameter for t-SNE.
    n_clusters : int
        Number of clusters for KMeans on reactions.
    drop_time0 : bool
        Whether to drop the initial time point (time == 0).

    Outputs
    -------
    - PCA scatter plot saved as 'pca_scatter.png'
    - t-SNE scatter plot saved as 'tsne_scatter.png'
    - Reaction heatmap with cluster ordering saved as 'reaction_clusters_heatmap.png'
    """

    # 1. Load and preprocess data
    df = pd.read_csv(csv_path, header=0)
    if df.columns[0] != "time":
        cols = df.columns.tolist()
        cols[0] = "time"
        df.columns = cols
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    if drop_time0:
        df = df[df["time"] != 0].reset_index(drop=True)

    # Separate time and flux data
    time = df["time"].values
    flux_df = df.drop(columns=["time"])

    # Filter out reactions with high zero_ratio
    zero_ratio = (np.abs(flux_df) <= eps).sum(axis=0) / flux_df.shape[0]
    active_reactions = zero_ratio[zero_ratio < zero_threshold].index.tolist()
    flux_active = flux_df[active_reactions]

    # 2. PCA on time points
    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(flux_active.values)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=time)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Time Points")
    plt.colorbar(sc, label="Time (s)")
    plt.tight_layout()
    plt.savefig("pca_scatter.png")
    plt.clf()

    # 3. t-SNE on time points
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, init="pca", random_state=42)
    X_tsne = tsne.fit_transform(flux_active.values)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=time)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE of Time Points")
    plt.colorbar(sc, label="Time (s)")
    plt.tight_layout()
    plt.savefig("tsne_scatter.png")
    plt.clf()

    # 4. KMeans clustering on reactions
    # Transpose: rows=reactions, cols=time points
    reaction_matrix = flux_active.T.values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reaction_matrix)

    # Create DataFrame for heatmap
    reaction_order = np.argsort(cluster_labels)
    ordered_reactions = [active_reactions[i] for i in reaction_order]
    heatmap_matrix = flux_active[ordered_reactions].T.values

    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_matrix, aspect="auto", origin="lower")
    plt.xlabel("Time Index")
    plt.ylabel("Reactions (clustered)")
    plt.title(f"Reaction Flux Heatmap (KMeans, k={n_clusters})")
    plt.tight_layout()
    plt.savefig("reaction_clusters_heatmap.png")
    plt.clf()

    # Print cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("KMeans cluster sizes:")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} reactions")


if __name__ == "__main__":
    # Example usage
    dimensionality_reduction_and_clustering(
        "agent_0_csv_analysis_solution_fluxes_400.csv"
    )
