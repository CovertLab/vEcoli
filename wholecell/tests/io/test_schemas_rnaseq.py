"""Tests for RNA-seq Pandera schemas in wholecell.io.schemas.rnaseq."""

from __future__ import annotations

import unittest

import pandas as pd
import pandera.errors as pa_errors

from wholecell.io.schemas.rnaseq import (
    RnaseqSamplesManifestSchema,
    RnaseqTpmTableSchema,
)

# Silence Sphinx autodoc warning
unittest.TestCase.__module__ = "unittest"


class TestRnaseqSchemas(unittest.TestCase):
    def test_tpm_schema_accepts_valid_data(self):
        df = pd.DataFrame(
            {
                "gene_id": ["EG10001", "EG10002"],
                "tpm_mean": [0.0, 10.5],
                "tpm_std": [0.1, 0.2],
            }
        )

        validated = RnaseqTpmTableSchema.validate(df)

        self.assertEqual(list(validated["gene_id"]), ["EG10001", "EG10002"])
        self.assertTrue((validated["tpm_mean"] >= 0).all())

    def test_tpm_schema_rejects_negative_tpm(self):
        df = pd.DataFrame(
            {
                "gene_id": ["EG10001"],
                "tpm_mean": [-1.0],
            }
        )

        with self.assertRaises(pa_errors.SchemaError):
            RnaseqTpmTableSchema.validate(df)

    def test_manifest_schema_accepts_valid_data(self):
        df = pd.DataFrame(
            {
                "dataset_id": ["ds1"],
                "dataset_description": ["test dataset"],
                "file_path": ["/tmp/rnaseq_test.tsv"],
                "data_source": ["test_source"],
            }
        )

        validated = RnaseqSamplesManifestSchema.validate(df)

        self.assertEqual(list(validated["dataset_id"]), ["ds1"])
        self.assertEqual(validated.loc[0, "data_source"], "test_source")

    # --- Required columns missing ---

    def test_tpm_schema_rejects_missing_gene_id(self):
        df = pd.DataFrame({"tpm_mean": [10.0, 20.0]})

        with self.assertRaises(pa_errors.SchemaError):
            RnaseqTpmTableSchema.validate(df)

    def test_tpm_schema_rejects_missing_tpm_mean(self):
        df = pd.DataFrame({"gene_id": ["EG10001", "EG10002"]})

        with self.assertRaises(pa_errors.SchemaError):
            RnaseqTpmTableSchema.validate(df)

    def test_manifest_schema_rejects_missing_required_column(self):
        df = pd.DataFrame(
            {
                "dataset_id": ["ds1"],
                "dataset_description": ["test"],
                # missing file_path and data_source
            }
        )

        with self.assertRaises(pa_errors.SchemaError):
            RnaseqSamplesManifestSchema.validate(df)

    # --- Wrong type (non-coercible) ---

    def test_tpm_schema_rejects_non_numeric_tpm_mean(self):
        df = pd.DataFrame(
            {
                "gene_id": ["EG10001"],
                "tpm_mean": ["not_a_number"],
            }
        )

        with self.assertRaises((pa_errors.SchemaError, ValueError)):
            RnaseqTpmTableSchema.validate(df)

    def test_tpm_schema_rejects_duplicate_gene_id(self):
        df = pd.DataFrame(
            {
                "gene_id": ["EG10001", "EG10001"],
                "tpm_mean": [10.0, 20.0],
            }
        )

        with self.assertRaises(pa_errors.SchemaError):
            RnaseqTpmTableSchema.validate(df)


if __name__ == "__main__":
    unittest.main()
