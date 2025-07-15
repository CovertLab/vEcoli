"""
This script preprocesses FBA flux data, computes zero-flux ratios, filters reactions,
and visualizes distributions of zero-flux ratios and non-zero flux values.
It can be run directly or imported as a module for further analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def preprocess_and_visualize_flux(
    csv_path: str, zero_threshold: float = 0.999, eps: float = 1e-9
) -> (pd.DataFrame, pd.DataFrame):
    """
    Reads a time-series FBA flux CSV, computes zero-flux ratios per reaction,
    filters out largely inactive reactions, and visualizes distributions.

    Parameters:
    - csv_path: Path to the CSV file. The first column should be time (no header).
    - zero_threshold: Fraction of zeros above which a reaction is considered inactive.
    - eps: Small tolerance for zero comparison (abs(flux) <= eps).

    Returns:
    - zr_df: DataFrame with columns ['reaction', 'zero_ratio'] sorted by ratio.
    - active_flux_df: DataFrame of flux values for reactions with zero_ratio < zero_threshold.
    """

    # Load data and rename first column to 'time'
    df = pd.read_csv(csv_path, header=0)
    if df.columns[0] != "time":
        cols = df.columns.tolist()
        cols[0] = "time"
        df.columns = cols

    # Ensure time column is numeric
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    # Drop initial time point where time == 0
    df = df[df["time"] != 0].reset_index(drop=True)

    # Separate flux data (drop time)
    flux_df = df.drop(columns=["time"])

    # Compute zero-flux ratio per reaction
    zero_counts = (np.abs(flux_df) <= eps).sum(axis=0)
    zero_ratio = zero_counts / flux_df.shape[0]
    zero_ratio = zero_ratio.sort_values()

    zr_df = pd.DataFrame(
        {"reaction": zero_ratio.index, "zero_ratio": zero_ratio.values}
    )

    # Print top/bottom reactions
    print("[INFO] Top 5 reactions with least zero flux (most active):")
    print(zr_df.head(5))
    print("\n[INFO] Top 5 reactions with most zero flux (least active):")
    print(zr_df.tail(5))

    # Filter out reactions with high zero_ratio
    active_reactions = zr_df[zr_df["zero_ratio"] < zero_threshold]["reaction"].tolist()
    active_flux_df = flux_df[active_reactions]
    print(
        f"\n[INFO] Reactions remaining after filtering (zero_ratio < {zero_threshold}): {len(active_reactions)}"
    )

    # Visualization
    # Histogram of zero_flux ratios
    plt.figure(figsize=(8, 4))
    plt.hist(zr_df["zero_ratio"], edgecolor="k")
    plt.xlabel("Zero Flux Ratio")
    plt.ylabel("Number of Reactions")
    plt.title("Distribution of Zero-Flux Ratios")
    plt.tight_layout()
    plt.savefig("zero_flux_ratio_distribution.png", dpi=300)

    # Distribution of non-zero flux values (log10 scale)
    nonzero_values = flux_df.values.flatten()
    nonzero_values = nonzero_values[np.abs(nonzero_values) > eps]

    plt.figure(figsize=(8, 4))
    plt.hist(np.log10(nonzero_values), bins=30, density=True, edgecolor="k")
    plt.xlabel("log10(Flux Value)")
    plt.ylabel("Density")
    plt.title("Log-Scaled Distribution of Non-zero Flux Values")
    plt.tight_layout()
    plt.savefig("log_nonzero_flux_distribution.png", dpi=300)

    return zr_df, active_flux_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess and visualize FBA flux data."
    )
    parser.add_argument(
        "csv_path", type=str, help="Path to the CSV file containing flux data."
    )
    parser.add_argument(
        "--zero_threshold",
        type=float,
        default=0.99,
        help="Fraction of zeros above which a reaction is considered inactive.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-9,
        help="Small tolerance for zero comparison (absolute flux <= eps).",
    )
    args = parser.parse_args()

    # Run the preprocessing and visualization
    zr_df, filtered_flux = preprocess_and_visualize_flux(
        args.csv_path, zero_threshold=args.zero_threshold, eps=args.eps
    )
    filtered_flux.to_csv("filtered_flux.csv", index=False, encoding="utf-8-sig")
    print("\nPreprocessing and visualization complete.")
