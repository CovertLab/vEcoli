"""Tests for wholecell.io.data_qc."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from wholecell.io import data_qc

unittest.TestCase.__module__ = "unittest"


class TestCompareRnaseqTables(unittest.TestCase):
    def test_perfect_match(self):
        """Identical tables should have correlation=1, RMSE=0."""
        ref = pd.DataFrame({"gene_id": ["A", "B", "C"], "tpm_mean": [10.0, 20.0, 30.0]})
        expt = pd.DataFrame(
            {"gene_id": ["A", "B", "C"], "tpm_mean": [10.0, 20.0, 30.0]}
        )

        result = data_qc.compare_rnaseq_tables(ref, expt)

        self.assertEqual(len(result.comparison_table), 3)
        self.assertEqual(result.summary_stats["n_genes_matched"], 3)
        self.assertAlmostEqual(result.summary_stats["pearson_r"], 1.0, places=5)
        self.assertAlmostEqual(result.summary_stats["rmse"], 0.0, places=5)
        self.assertEqual(result.genes_only_in_ref, [])
        self.assertEqual(result.genes_only_in_expt, [])

    def test_partial_overlap(self):
        """Tables with partial overlap should report missing genes."""
        ref = pd.DataFrame({"gene_id": ["A", "B", "C"], "tpm_mean": [10.0, 20.0, 30.0]})
        expt = pd.DataFrame(
            {"gene_id": ["B", "C", "D"], "tpm_mean": [20.0, 30.0, 40.0]}
        )

        result = data_qc.compare_rnaseq_tables(ref, expt)

        self.assertEqual(len(result.comparison_table), 4)  # A, B, C, D
        self.assertEqual(result.summary_stats["n_genes_matched"], 2)  # B, C
        self.assertEqual(result.genes_only_in_ref, ["A"])
        self.assertEqual(result.genes_only_in_expt, ["D"])
        self.assertAlmostEqual(result.summary_stats["pct_ref_covered"], 2 / 3 * 100)
        self.assertAlmostEqual(result.summary_stats["pct_expt_covered"], 2 / 3 * 100)

    def test_no_overlap(self):
        """Tables with no overlap should have NaN correlations."""
        ref = pd.DataFrame({"gene_id": ["A", "B"], "tpm_mean": [10.0, 20.0]})
        expt = pd.DataFrame({"gene_id": ["C", "D"], "tpm_mean": [30.0, 40.0]})

        result = data_qc.compare_rnaseq_tables(ref, expt)

        self.assertEqual(result.summary_stats["n_genes_matched"], 0)
        self.assertTrue(np.isnan(result.summary_stats["pearson_r"]))
        self.assertEqual(sorted(result.genes_only_in_ref), ["A", "B"])
        self.assertEqual(sorted(result.genes_only_in_expt), ["C", "D"])

    def test_comparison_table_columns(self):
        """Comparison table should have expected columns in order."""
        ref = pd.DataFrame({"gene_id": ["A"], "tpm_mean": [10.0]})
        expt = pd.DataFrame({"gene_id": ["A"], "tpm_mean": [15.0]})

        result = data_qc.compare_rnaseq_tables(ref, expt)

        self.assertListEqual(
            list(result.comparison_table.columns),
            ["gene_id", "ref_tpm", "expt_tpm"],
        )

    def test_comparison_table_columns_with_annotations(self):
        """Comparison table should have columns in preferred order when annotated."""
        ref = pd.DataFrame({"gene_id": ["A"], "tpm_mean": [10.0]})
        expt = pd.DataFrame({"gene_id": ["A"], "tpm_mean": [15.0]})
        annotations = pd.DataFrame({"gene_id": ["A"], "gene_name": ["geneA"]})
        essential = {"A"}

        result = data_qc.compare_rnaseq_tables(
            ref, expt, gene_annotations=annotations, essential_genes=essential
        )

        self.assertListEqual(
            list(result.comparison_table.columns),
            ["gene_id", "gene_name", "gene_essential", "ref_tpm", "expt_tpm"],
        )

    def test_fold_change_with_zeros(self):
        """Fold change should handle zeros via pseudocount."""
        ref = pd.DataFrame({"gene_id": ["A", "B"], "tpm_mean": [0.0, 100.0]})
        expt = pd.DataFrame({"gene_id": ["A", "B"], "tpm_mean": [100.0, 0.0]})

        result = data_qc.compare_rnaseq_tables(ref, expt)

        self.assertFalse(np.isnan(result.summary_stats["log2_fold_change_mean"]))
        self.assertFalse(np.isinf(result.summary_stats["log2_fold_change_mean"]))

    def test_with_gene_annotations(self):
        """Should add gene_name column when annotations provided."""
        ref = pd.DataFrame({"gene_id": ["A", "B", "C"], "tpm_mean": [10.0, 20.0, 30.0]})
        expt = pd.DataFrame(
            {"gene_id": ["A", "B", "C"], "tpm_mean": [15.0, 25.0, 35.0]}
        )
        annotations = pd.DataFrame(
            {"gene_id": ["A", "B", "C"], "gene_name": ["geneA", "geneB", "geneC"]}
        )

        result = data_qc.compare_rnaseq_tables(ref, expt, gene_annotations=annotations)

        self.assertIn("gene_name", result.comparison_table.columns)
        self.assertEqual(
            list(result.comparison_table["gene_name"]), ["geneA", "geneB", "geneC"]
        )

    def test_with_essential_genes(self):
        """Should add gene_essential column and stats when essentiality provided."""
        ref = pd.DataFrame({"gene_id": ["A", "B", "C"], "tpm_mean": [10.0, 20.0, 30.0]})
        expt = pd.DataFrame(
            {"gene_id": ["A", "B", "C"], "tpm_mean": [15.0, 25.0, 35.0]}
        )
        essential = {"A", "C"}

        result = data_qc.compare_rnaseq_tables(ref, expt, essential_genes=essential)

        self.assertIn("gene_essential", result.comparison_table.columns)
        self.assertEqual(
            list(result.comparison_table["gene_essential"]), [True, False, True]
        )
        self.assertEqual(result.summary_stats["n_essential_genes_matched"], 2)
        self.assertEqual(result.summary_stats["n_essential_genes_total"], 2)

    def test_essential_genes_missing_tracking(self):
        """Should track essential genes that are only in ref or expt."""
        ref = pd.DataFrame({"gene_id": ["A", "B", "C"], "tpm_mean": [10.0, 20.0, 30.0]})
        expt = pd.DataFrame(
            {"gene_id": ["B", "C", "D"], "tpm_mean": [25.0, 35.0, 45.0]}
        )
        essential = {"A", "B", "D"}

        result = data_qc.compare_rnaseq_tables(ref, expt, essential_genes=essential)

        self.assertEqual(result.summary_stats["n_essential_only_in_ref"], 1)  # A
        self.assertEqual(result.summary_stats["n_essential_only_in_expt"], 1)  # D
        self.assertEqual(result.summary_stats["n_essential_genes_matched"], 1)  # B

    def test_with_both_annotations_and_essentiality(self):
        """Should include both gene_name and gene_essential columns."""
        ref = pd.DataFrame({"gene_id": ["A", "B"], "tpm_mean": [10.0, 20.0]})
        expt = pd.DataFrame({"gene_id": ["A", "B"], "tpm_mean": [15.0, 25.0]})
        annotations = pd.DataFrame(
            {"gene_id": ["A", "B"], "gene_name": ["geneA", "geneB"]}
        )
        essential = {"A"}

        result = data_qc.compare_rnaseq_tables(
            ref, expt, gene_annotations=annotations, essential_genes=essential
        )

        cols = list(result.comparison_table.columns)
        self.assertIn("gene_name", cols)
        self.assertIn("gene_essential", cols)
        self.assertEqual(result.comparison_table.loc[0, "gene_name"], "geneA")
        self.assertEqual(result.comparison_table.loc[0, "gene_essential"], True)
        self.assertEqual(result.comparison_table.loc[1, "gene_essential"], False)


if __name__ == "__main__":
    unittest.main()
