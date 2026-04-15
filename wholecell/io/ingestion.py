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


def _default_ecoli_sources_dir() -> str:
    """Sibling-repo default: <vEcoli-repo-root>/../ecoli-sources."""
    import wholecell
    repo_root = Path(wholecell.__file__).resolve().parents[1]
    return str((repo_root.parent / "ecoli-sources").resolve())


def resolve_ecoli_sources_path(path: PathLike | None) -> str | None:
    """
    Expand ``$ECOLI_SOURCES`` (or ``${ECOLI_SOURCES}``) in a config path.

    If the environment variable is unset, falls back to the sibling
    ``<vEcoli-repo-root>/../ecoli-sources`` directory. Returns ``None`` for
    a ``None`` input so legacy "no manifest" configs keep working.
    """
    if path is None:
        return None
    s = str(path)
    if "$ECOLI_SOURCES" in s or "${ECOLI_SOURCES}" in s:
        sources = os.environ.get("ECOLI_SOURCES") or _default_ecoli_sources_dir()
        s = s.replace("${ECOLI_SOURCES}", sources).replace("$ECOLI_SOURCES", sources)
    return os.path.expanduser(s)


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


def _overlay_manifest_paths() -> list[str]:
    """
    Parse ``$ECOLI_SOURCES_OVERLAYS`` into a list of manifest paths.

    Value format: colon-separated paths to overlay ``manifest.tsv`` files.
    Each path may include ``$ECOLI_SOURCES`` (resolved the same way as the
    primary manifest path) or ``~`` for home-dir expansion. Empty entries
    are ignored. Missing env var → empty list.
    """
    raw = os.environ.get("ECOLI_SOURCES_OVERLAYS", "")
    out: list[str] = []
    for item in raw.split(":"):
        item = item.strip()
        if not item:
            continue
        resolved = resolve_ecoli_sources_path(item)
        if resolved:
            out.append(resolved)
    return out


def ingest_rnaseq_manifest_with_overlays(
    primary_path: PathLike,
    overlay_paths: list[PathLike] | None = None,
) -> pd.DataFrame:
    """
    Load a primary manifest plus any overlay manifests and return the
    concatenated result.

    ``file_path`` entries are resolved relative to each manifest's own
    directory before concatenation, so TSV lookups work regardless of
    which repo an entry came from. ``dataset_id`` must be unique across
    the union; collisions raise ``ValueError``.

    If ``overlay_paths`` is None, the list is taken from the
    ``ECOLI_SOURCES_OVERLAYS`` environment variable.
    """
    if overlay_paths is None:
        overlay_paths = _overlay_manifest_paths()

    primary_path = resolve_ecoli_sources_path(primary_path)
    primary = ingest_rnaseq_manifest(primary_path)
    primary["_manifest_source"] = str(primary_path)
    combined_frames = [primary]

    for op in overlay_paths:
        overlay = ingest_rnaseq_manifest(op)
        overlay["_manifest_source"] = str(op)
        combined_frames.append(overlay)

    combined = pd.concat(combined_frames, ignore_index=True)

    dup_mask = combined["dataset_id"].duplicated(keep=False)
    if dup_mask.any():
        dups = (
            combined[dup_mask]
            .groupby("dataset_id")["_manifest_source"]
            .apply(list)
            .to_dict()
        )
        raise ValueError(
            "dataset_id collisions across primary + overlays: "
            + "; ".join(f"{k!r} in {v}" for k, v in dups.items())
        )

    return combined.drop(columns="_manifest_source")


def ingest_transcriptome(
    manifest_path: PathLike, dataset_id: str
) -> Tuple[pd.DataFrame, dict]:
    """
    Ingest a single transcriptome (TPM table) specified by ``dataset_id``.

    This is a convenience wrapper that:
    1) Loads the primary manifest plus any overlay manifests from
       ``$ECOLI_SOURCES_OVERLAYS``.
    2) Looks up the row with the given ``dataset_id``.
    3) Loads and validates the corresponding TPM table.

    Parameters
    ----------
    manifest_path:
        Path to the primary RNA-seq samples manifest TSV.
    dataset_id:
        Identifier of the dataset to load (must match a ``dataset_id`` row
        in the primary manifest or one of its overlays).

    Returns
    -------
    (pandas.DataFrame, dict)
        - Validated TPM table for the requested dataset.
        - Metadata dict for the selected manifest row.

    Raises
    ------
    KeyError
        If ``dataset_id`` is not found in the primary manifest or overlays.
    ValueError
        If multiple rows share the same ``dataset_id`` across primary +
        overlays.
    """
    manifest_path = resolve_ecoli_sources_path(manifest_path)
    manifest = ingest_rnaseq_manifest_with_overlays(manifest_path)
    matches = manifest[manifest["dataset_id"] == dataset_id]

    if matches.empty:
        overlay_paths = _overlay_manifest_paths()
        msg = f"Dataset_id {dataset_id!r} not found in manifest {manifest_path!r}"
        if overlay_paths:
            msg += f" or overlays {overlay_paths}"
        raise KeyError(msg + ".")
    if len(matches) > 1:
        raise ValueError(
            f"Dataset_id {dataset_id!r} appears more than once in the "
            f"combined primary + overlay manifest."
        )

    row = matches.iloc[0]
    tpm_path = row["file_path"]
    tpm_table = ingest_rnaseq_tpm_table(tpm_path)

    return tpm_table, row.to_dict()
