"""
Quality control utilities for comparing experimental data against reference datasets.

This module provides stateless, composable functions that operate on DataFrames
(typically already validated by ``wholecell.io.ingestion``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Union
from scipy.stats import spearmanr

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Default paths relative to repository root
DEFAULT_GENES_TSV = Path("reconstruction/ecoli/flat/genes.tsv")
DEFAULT_ESSENTIAL_GENES_TSV = Path("validation/ecoli/flat/essential_genes.tsv")


def load_gene_annotations(
    genes_tsv_path: PathLike = DEFAULT_GENES_TSV,
) -> pd.DataFrame:
    """
    Load gene annotations (id -> symbol mapping) from genes.tsv.

    Parameters
    ----------
    genes_tsv_path:
        Path to genes.tsv file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [gene_id, gene_name].
    """
    df = pd.read_csv(genes_tsv_path, sep="\t", comment="#")
    return df[["id", "symbol"]].rename(columns={"id": "gene_id", "symbol": "gene_name"})


def load_essential_genes(
    essential_genes_tsv_path: PathLike = DEFAULT_ESSENTIAL_GENES_TSV,
) -> Set[str]:
    """
    Load set of essential gene IDs from essential_genes.tsv.

    Parameters
    ----------
    essential_genes_tsv_path:
        Path to essential_genes.tsv file.

    Returns
    -------
    Set[str]
        Set of essential gene FrameIDs (e.g., {"EG10068", "EG10117", ...}).
    """
    df = pd.read_csv(essential_genes_tsv_path, sep="\t", comment="#")
    return set(df["FrameID"].str.strip('"'))


@dataclass
class RnaseqComparisonResult:
    """
    Result of comparing two RNA-seq TPM tables.

    Attributes
    ----------
    comparison_table:
        Outer join of reference and experimental TPMs. Core columns:
        [gene_id, ref_tpm, expt_tpm]. If annotations provided, also includes
        [gene_name, gene_essential].
    summary_stats:
        Dictionary of summary statistics (correlation, RMSE, gene counts, etc.).
    genes_only_in_ref:
        List of gene_ids present in reference but missing from experimental.
    genes_only_in_expt:
        List of gene_ids present in experimental but missing from reference.
    """

    comparison_table: pd.DataFrame
    summary_stats: dict
    genes_only_in_ref: List[str]
    genes_only_in_expt: List[str]


def compare_rnaseq_tables(
    ref_df: pd.DataFrame,
    expt_df: pd.DataFrame,
    gene_annotations: Optional[pd.DataFrame] = None,
    essential_genes: Optional[Set[str]] = None,
) -> RnaseqComparisonResult:
    """
    Compare experimental RNA-seq TPM table against a reference.

    Both DataFrames must follow ``RnaseqTpmTableSchema`` (columns: ``gene_id``,
    ``tpm_mean``).

    Parameters
    ----------
    ref_df:
        Reference TPM table.
    expt_df:
        Experimental TPM table.
    gene_annotations:
        Optional DataFrame with columns [gene_id, gene_name] for adding gene symbols.
        Can be loaded via ``load_gene_annotations()``.
    essential_genes:
        Optional set of essential gene IDs for adding essentiality flag.
        Can be loaded via ``load_essential_genes()``.

    Returns
    -------
    RnaseqComparisonResult
        Contains comparison table, summary statistics, and missing gene lists.
    """
    # Prepare reference subset
    ref_subset = ref_df[["gene_id", "tpm_mean"]].copy()
    ref_subset = ref_subset.rename(columns={"tpm_mean": "ref_tpm"})

    # Prepare experimental subset
    expt_subset = expt_df[["gene_id", "tpm_mean"]].copy()
    expt_subset = expt_subset.rename(columns={"tpm_mean": "expt_tpm"})

    # Outer join
    comparison = pd.merge(
        ref_subset,
        expt_subset,
        on="gene_id",
        how="outer",
    )

    # Add gene annotations if provided
    if gene_annotations is not None:
        comparison = pd.merge(
            comparison,
            gene_annotations,
            on="gene_id",
            how="left",
        )

    # Add essentiality flag if provided
    if essential_genes is not None:
        comparison["gene_essential"] = comparison["gene_id"].isin(essential_genes)

    # Identify missing genes
    genes_only_in_ref = comparison.loc[
        comparison["expt_tpm"].isna(), "gene_id"
    ].tolist()
    genes_only_in_expt = comparison.loc[
        comparison["ref_tpm"].isna(), "gene_id"
    ].tolist()

    # Compute summary stats on genes present in both
    matched = comparison.dropna(subset=["ref_tpm", "expt_tpm"])
    summary_stats = _compute_summary_stats(
        matched["ref_tpm"].values,
        matched["expt_tpm"].values,
        n_ref_total=len(ref_subset),
        n_expt_total=len(expt_subset),
        n_only_ref=len(genes_only_in_ref),
        n_only_expt=len(genes_only_in_expt),
    )

    # Add essentiality-specific stats if available
    if essential_genes is not None:
        matched_essential = matched[matched["gene_essential"]]
        summary_stats["n_essential_genes_matched"] = len(matched_essential)
        summary_stats["n_essential_genes_total"] = len(essential_genes)

        mask_essential = comparison["gene_essential"]
        summary_stats["n_essential_only_in_ref"] = int(
            (comparison["expt_tpm"].isna() & mask_essential).sum()
        )
        summary_stats["n_essential_only_in_expt"] = int(
            (comparison["ref_tpm"].isna() & mask_essential).sum()
        )

    # Reorder columns: gene_id, gene_name, gene_essential, ref_tpm, expt_tpm
    preferred_order = ["gene_id", "gene_name", "gene_essential", "ref_tpm", "expt_tpm"]
    cols = [c for c in preferred_order if c in comparison.columns]
    cols += [c for c in comparison.columns if c not in cols]
    comparison = comparison[cols]

    return RnaseqComparisonResult(
        comparison_table=comparison,
        summary_stats=summary_stats,
        genes_only_in_ref=genes_only_in_ref,
        genes_only_in_expt=genes_only_in_expt,
    )


def _compute_summary_stats(
    ref_tpm: np.ndarray,
    expt_tpm: np.ndarray,
    n_ref_total: int,
    n_expt_total: int,
    n_only_ref: int,
    n_only_expt: int,
) -> dict:
    """
    Compute summary statistics for matched gene pairs.

    Parameters
    ----------
    ref_tpm:
        Reference TPM values (matched genes only).
    expt_tpm:
        Experimental TPM values (matched genes only).
    n_ref_total:
        Total number of genes in reference.
    n_expt_total:
        Total number of genes in experimental.
    n_only_ref:
        Genes present only in reference.
    n_only_expt:
        Genes present only in experimental.

    Returns
    -------
    dict
        Summary statistics including correlations, RMSE, and gene counts.
    """
    n_matched = len(ref_tpm)

    # Handle edge case of no matched genes
    if n_matched == 0:
        return {
            "n_genes_matched": 0,
            "n_genes_ref_total": n_ref_total,
            "n_genes_expt_total": n_expt_total,
            "n_genes_only_in_ref": n_only_ref,
            "n_genes_only_in_expt": n_only_expt,
            "pct_ref_covered": 0.0,
            "pct_expt_covered": 0.0,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "rmse": np.nan,
            "log2_fold_change_mean": np.nan,
            "log2_fold_change_std": np.nan,
        }

    # Pearson correlation
    pearson_r = np.corrcoef(ref_tpm, expt_tpm)[0, 1]

    # Spearman correlation (rank-based)
    spearman_r, _ = spearmanr(ref_tpm, expt_tpm)

    # RMSE
    rmse = np.sqrt(np.mean((ref_tpm - expt_tpm) ** 2))

    # Log2 fold change (with pseudocount to handle zeros)
    pseudocount = 1.0
    log2_fc = np.log2((expt_tpm + pseudocount) / (ref_tpm + pseudocount))
    log2_fc_mean = np.mean(log2_fc)
    log2_fc_std = np.std(log2_fc)

    return {
        "n_genes_matched": n_matched,
        "n_genes_ref_total": n_ref_total,
        "n_genes_expt_total": n_expt_total,
        "n_genes_only_in_ref": n_only_ref,
        "n_genes_only_in_expt": n_only_expt,
        "pct_ref_covered": 100.0 * n_matched / n_ref_total if n_ref_total > 0 else 0.0,
        "pct_expt_covered": (
            100.0 * n_matched / n_expt_total if n_expt_total > 0 else 0.0
        ),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "rmse": rmse,
        "log2_fold_change_mean": log2_fc_mean,
        "log2_fold_change_std": log2_fc_std,
    }
