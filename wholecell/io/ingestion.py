"""
Utilities for ingesting experimental data (e.g. RNA-seq transcriptomes)
using the canonical Pandera schemas in ``wholecell.io.schemas``.

This module is intentionally narrow for now:
- Load TSVs into pandas DataFrames.
- Validate them against the RNA-seq schemas.
- Provide a small convenience wrapper to fetch a single transcriptome
  given a manifest and ``dataset_id``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from wholecell.io.schemas.rnaseq import (
    RnaseqSamplesManifestSchema,
    RnaseqTpmTableSchema,
)

PathLike = Union[str, Path]


def _read_tsv(path: PathLike) -> pd.DataFrame:
    """Read a tab-delimited file into a DataFrame."""
    path = Path(path)
    return pd.read_csv(path, sep="\t")


def ingest_rnaseq_tpm_table(path: PathLike) -> pd.DataFrame:
    """
    Load and validate a single RNA-seq TPM table.

    Parameters
    ----------
    path:
        Path to a TSV file with columns matching ``RnaseqTpmTableSchema``.

    Returns
    -------
    pandas.DataFrame
        Validated DataFrame; extra columns are preserved but only the
        required/optional schema columns are validated.
    """
    df = _read_tsv(path)
    return RnaseqTpmTableSchema.validate(df)


def ingest_rnaseq_manifest(path: PathLike) -> pd.DataFrame:
    """
    Load and validate an RNA-seq samples manifest.

    Relative ``file_path`` entries are resolved relative to the manifest
    directory for convenience.

    Parameters
    ----------
    path:
        Path to the manifest TSV file.

    Returns
    -------
    pandas.DataFrame
        Validated manifest with ``file_path`` normalized to absolute paths.
    """
    path = Path(path)
    df = _read_tsv(path)
    manifest = RnaseqSamplesManifestSchema.validate(df)

    base_dir = path.parent

    def _normalize_file_path(p: str) -> str:
        if os.path.isabs(p):
            return p
        return str((base_dir / p).resolve())

    manifest["file_path"] = manifest["file_path"].astype(str).map(_normalize_file_path)
    return manifest


def ingest_transcriptome(
    manifest_path: PathLike, dataset_id: str
) -> Tuple[pd.DataFrame, dict]:
    """
    Ingest a single transcriptome (TPM table) specified by ``dataset_id``.

    This is a convenience wrapper that:
    1) Validates the manifest.
    2) Looks up the row with the given ``dataset_id``.
    3) Loads and validates the corresponding TPM table.

    Parameters
    ----------
    manifest_path:
        Path to the RNA-seq samples manifest TSV.
    dataset_id:
        Identifier of the dataset to load (must match a ``dataset_id`` row).

    Returns
    -------
    (pandas.DataFrame, dict)
        - Validated TPM table for the requested dataset.
        - Metadata dict for the selected manifest row.

    Raises
    ------
    KeyError
        If ``dataset_id`` is not found in the manifest.
    ValueError
        If multiple rows share the same ``dataset_id``.
    """
    manifest = ingest_rnaseq_manifest(manifest_path)
    matches = manifest[manifest["dataset_id"] == dataset_id]

    if matches.empty:
        raise KeyError(
            f"Dataset_id {dataset_id!r} not found in manifest {manifest_path!r}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Dataset_id {dataset_id!r} appears more than once in manifest "
            f"{manifest_path!r}."
        )

    row = matches.iloc[0]
    tpm_path = row["file_path"]
    tpm_table = ingest_rnaseq_tpm_table(tpm_path)

    return tpm_table, row.to_dict()
