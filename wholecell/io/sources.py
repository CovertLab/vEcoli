"""
Resolver for the ecoli-sources data bundle.

A *bundle* is a complete ``{canonical_key -> source_path}`` mapping that
defines, for a given ParCa run, which file (or directory) provides the
data for each addressable role in the model. The default reference
bundle ships with the ``ecoli-sources`` package as
``ecoli_sources/data/reference_bundle.tsv``; campaigns / variants pin
alternative bundle manifests.

vEcoli's ParCa-time consumers (``KnowledgeBaseRaw``, ``transcription.py``,
etc.) read each canonical key through ``SourceBundle`` rather than
hardcoded path joins, so any data source can be substituted at the
bundle level without code changes.

See SMS/.claude/data-ingestion/bundle-migration-plan.md for the full
design context.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


PathLike = Union[str, os.PathLike]


class SourceBundle:
    """Resolve canonical keys to source paths via a bundle manifest.

    Loads the manifest TSV once at construction time and caches
    ``{canonical_key: absolute_path}`` for the rest of the bundle's
    lifetime.

    Parameters
    ----------
    manifest_path:
        Path to a bundle manifest TSV (e.g.
        ``ecoli_sources/data/reference_bundle.tsv``). If ``None``,
        defaults to the installed ``ecoli_sources`` package's
        ``BUNDLE_PATH`` (the default reference bundle).

    Notes
    -----
    The bundle manifest schema (columns: ``canonical_key``,
    ``source_path``, ``description``, ``schema_name``) is owned by the
    ``ecoli-sources`` package. ``source_path`` values are interpreted
    relative to the manifest's parent directory's parent (i.e., the
    package's ``data/`` root) so that relative paths in the manifest
    work whether the bundle lives in the default location or in an
    alternative bundle file shipped alongside it.
    """

    def __init__(self, manifest_path: Optional[PathLike] = None):
        if manifest_path is None:
            # Default to the bundle shipped with ecoli_sources.
            from ecoli_sources import BUNDLE_PATH
            manifest_path = BUNDLE_PATH

        self._manifest_path: Path = Path(manifest_path).resolve()
        if not self._manifest_path.is_file():
            raise FileNotFoundError(
                f"Bundle manifest not found: {self._manifest_path}"
            )

        # Bundle root is the directory holding the manifest. source_path
        # values in the manifest are relative to this root, so that the
        # default reference_bundle.tsv (sibling of rnaseq_experimental/
        # and flat/) resolves entries correctly.
        self._bundle_root: Path = self._manifest_path.parent

        df = pd.read_csv(self._manifest_path, sep="\t", comment="#")

        # Defensive validation: shape + canonical-key contract. Catches
        # malformed or incomplete bundles at load time rather than at
        # consumer call sites deep in ParCa. Raises with a clear error
        # listing missing required keys when applicable.
        from schemas import ReferenceBundleSchema  # ecoli-sources package
        ReferenceBundleSchema.validate(df, lazy=True)

        self._index: dict[str, Path] = {
            str(row["canonical_key"]): (self._bundle_root / str(row["source_path"])).resolve()
            for _, row in df.iterrows()
        }

    @property
    def manifest_path(self) -> Path:
        """Absolute path to the loaded bundle manifest."""
        return self._manifest_path

    @property
    def bundle_root(self) -> Path:
        """Bundle root directory (manifest's containing directory)."""
        return self._bundle_root

    def has(self, canonical_key: str) -> bool:
        """True if ``canonical_key`` is registered in this bundle."""
        return canonical_key in self._index

    def get(self, canonical_key: str) -> Path:
        """Absolute path (file or directory) for ``canonical_key``.

        Raises
        ------
        KeyError
            If the canonical key is not registered in the loaded bundle.
            The error message names the missing key and the manifest
            that was loaded so debugging is straightforward.
        """
        try:
            return self._index[canonical_key]
        except KeyError:
            raise KeyError(
                f"canonical_key {canonical_key!r} not found in bundle "
                f"manifest {self._manifest_path}. "
                f"Available keys: {sorted(self._index)}"
            ) from None

    def keys(self) -> list[str]:
        """Sorted list of canonical keys registered in this bundle."""
        return sorted(self._index)

    def __contains__(self, canonical_key: str) -> bool:
        return self.has(canonical_key)

    def __repr__(self) -> str:
        return (
            f"SourceBundle(manifest_path={self._manifest_path}, "
            f"keys={len(self._index)})"
        )
