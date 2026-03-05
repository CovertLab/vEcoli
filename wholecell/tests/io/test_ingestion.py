"""Tests for wholecell.io.ingestion."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors

from wholecell.io import ingestion

# Silence Sphinx autodoc warning
unittest.TestCase.__module__ = "unittest"

# Fixtures: real-shaped TPM + manifest (same layout as experimental_data)
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestIngestion(unittest.TestCase):
    def _write_tsv(self, path: Path, df: pd.DataFrame) -> None:
        df.to_csv(path, sep="\t", index=False)

    def test_ingest_rnaseq_tpm_table_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tpm_path = Path(tmpdir) / "rnaseq_test_tpm.tsv"
            df = pd.DataFrame(
                {
                    "gene_id": ["EG10001", "EG10002"],
                    "tpm_mean": [10.0, 20.0],
                }
            )
            self._write_tsv(tpm_path, df)

            validated = ingestion.ingest_rnaseq_tpm_table(tpm_path)

            self.assertEqual(list(validated["gene_id"]), ["EG10001", "EG10002"])
            self.assertTrue((validated["tpm_mean"] >= 0).all())

    def test_ingest_rnaseq_manifest_normalizes_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tpm_path = tmpdir_path / "rnaseq_test_tpm.tsv"
            df_tpm = pd.DataFrame(
                {
                    "gene_id": ["EG10001"],
                    "tpm_mean": [1.0],
                }
            )
            self._write_tsv(tpm_path, df_tpm)

            manifest_path = tmpdir_path / "manifest.tsv"
            df_manifest = pd.DataFrame(
                {
                    "dataset_id": ["ds1"],
                    "dataset_description": ["test dataset"],
                    "file_path": [tpm_path.name],  # relative path
                    "data_source": ["test_source"],
                }
            )
            self._write_tsv(manifest_path, df_manifest)

            manifest = ingestion.ingest_rnaseq_manifest(manifest_path)

            self.assertEqual(len(manifest), 1)
            file_path = manifest.loc[0, "file_path"]
            self.assertTrue(os.path.isabs(file_path))
            self.assertTrue(file_path.endswith(str(tpm_path.name)))

    def test_ingest_transcriptome_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tpm_path = tmpdir_path / "rnaseq_test_tpm.tsv"
            df_tpm = pd.DataFrame(
                {
                    "gene_id": ["EG10001", "EG10002"],
                    "tpm_mean": [5.0, 15.0],
                }
            )
            self._write_tsv(tpm_path, df_tpm)

            manifest_path = tmpdir_path / "manifest.tsv"
            df_manifest = pd.DataFrame(
                {
                    "dataset_id": ["ds1"],
                    "dataset_description": ["test dataset"],
                    "file_path": [tpm_path.name],
                    "data_source": ["test_source"],
                }
            )
            self._write_tsv(manifest_path, df_manifest)

            tpm_table, metadata = ingestion.ingest_transcriptome(manifest_path, "ds1")

            self.assertEqual(list(tpm_table["gene_id"]), ["EG10001", "EG10002"])
            self.assertEqual(metadata["dataset_id"], "ds1")
            self.assertTrue(os.path.isabs(metadata["file_path"]))

    def test_ingest_transcriptome_from_fixture(self):
        """Load real-shaped fixture TPM via manifest (validates fixture format)."""
        manifest_path = FIXTURES_DIR / "rnaseq_manifest_small.tsv"
        if not manifest_path.exists():
            self.skipTest("fixtures not found")
        tpm_table, metadata = ingestion.ingest_transcriptome(manifest_path, "ref_small")
        self.assertGreater(len(tpm_table), 0)
        self.assertIn("gene_id", tpm_table.columns)
        self.assertIn("tpm_mean", tpm_table.columns)
        self.assertEqual(metadata["dataset_id"], "ref_small")

    # --- Validation failures (missing columns / wrong types) ---

    def test_ingest_rnaseq_tpm_table_fails_missing_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.tsv"
            df = pd.DataFrame({"tpm_mean": [1.0, 2.0]})  # missing gene_id
            df.to_csv(path, sep="\t", index=False)
            with self.assertRaises(pa_errors.SchemaError):
                ingestion.ingest_rnaseq_tpm_table(path)

    def test_ingest_rnaseq_tpm_table_fails_wrong_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.tsv"
            df = pd.DataFrame(
                {
                    "gene_id": ["EG10001"],
                    "tpm_mean": ["not_a_number"],
                }
            )
            df.to_csv(path, sep="\t", index=False)
            with self.assertRaises((pa_errors.SchemaError, ValueError)):
                ingestion.ingest_rnaseq_tpm_table(path)

    def test_ingest_rnaseq_manifest_fails_missing_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.tsv"
            df = pd.DataFrame(
                {
                    "dataset_id": ["ds1"],
                    "dataset_description": ["desc"],
                    # missing file_path, data_source
                }
            )
            df.to_csv(path, sep="\t", index=False)
            with self.assertRaises(pa_errors.SchemaError):
                ingestion.ingest_rnaseq_manifest(path)

    def test_ingest_transcriptome_fails_unknown_dataset_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tpm_path = tmpdir_path / "tpm.tsv"
            pd.DataFrame({"gene_id": ["EG10001"], "tpm_mean": [1.0]}).to_csv(
                tpm_path, sep="\t", index=False
            )
            manifest_path = tmpdir_path / "manifest.tsv"
            pd.DataFrame(
                {
                    "dataset_id": ["ds1"],
                    "dataset_description": ["d"],
                    "file_path": [tpm_path.name],
                    "data_source": ["s"],
                }
            ).to_csv(manifest_path, sep="\t", index=False)
            with self.assertRaises(KeyError):
                ingestion.ingest_transcriptome(manifest_path, "nonexistent_id")


if __name__ == "__main__":
    unittest.main()
