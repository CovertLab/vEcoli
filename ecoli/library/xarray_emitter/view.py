
"""
Specification of the schema transformation from the input hierarchy (Vivarium
stores) to the output hierarchy (Xarray nodes/variables), which is used to
construct the in-memory representation of :py:class:`.XarrayBuffer`.

See :ref:`variable_layout` for an overview.
"""


from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from operator import itemgetter
from itertools import chain
from functools import cached_property
from typing import Any, Self

import numpy as np
from xarray.core.datatree import NodePath

import unum
import pint

from vivarium.core.types import HierarchyPath
from vivarium.library.topology import get_in, dict_to_paths

from .utils import emitter_arg_error
from .emit_path import EmitPath


# ==============================================================================


@dataclass(kw_only=True, slots=True)
class LeafView:
    """
    Specification for how an individual Vivarium variable should be mapped onto
    a :py:class:`~xarray.DataArray` inside an eventual
    :py:class:`~xarray.DataTree` hierarchy.

    Example JSON configuration with default codecs::

      {
        "path": "metabolism/fluxes/internal/rxn",
        "unit": "[mmol/L.s]",
        "dtype": "<f4"
      }

    Example JSON configuration with custom codecs for the Zarr backend::

      {
        "path": "bulk/bulk_molecule",
        "dtype": "<i8",
        "codecs": {
          "filters_v2": [],
          "filters_v3": [],
          "compressors_v2": [{
            "id": "lzma", "format": 3, "check": -1, "preset": null,
            "filters": [{"id": 3, "dist": 8}, {"id": 33, "preset": 5}]
          }],
          "compressors_v3": [{
            "name": "numcodecs.lzma",
            "configuration": {
              "format": 3,
              "filters": [{"id": 3, "dist": 8}, {"id": 33, "preset": 5}]
            }
          }]
        }
      }

    The value of ``codecs`` is interpreted by
    :py:meth:`.AsyncBufferWriter.coo_codecs` and
    :py:meth:`.AsyncBufferWriter.var_codecs`.

    .. hint::
      In this example, ``_v2``/``_v3`` refers to the Zarr format, which is
      chosen by the configuration value ``emitter_arg.writer.zarr.format``. Zarr
      codecs only need to be specified in both formats if both formats will be
      used.
    """

    #: Target variable path inside the eventual :py:class:`~xarray.DataTree`.
    path: NodePath
    #: `Data type`_ for the output :py:class:`~xarray.DataArray`, provided either
    #: as a string or as a :py:class:`numpy.dtype`.
    #:
    #: .. _Data type: https://numpy.org/doc/stable/reference/arrays.dtypes.html
    dtype: str
    #: Unit string to store as an attribute inside the node, provided either
    #: as a string, as a :py:class:`pint.Unit`, or as a :py:class:`unum.Unum`.
    unit: str | None = None
    #: Backend-specific configuration of output codecs.
    codecs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # parse variable name
        match self.path:
            case NodePath():
                pass
            case str():
                self.path = NodePath(self.path)
            case p:
                raise TypeError(
                    f"Expected (str|NodePath) in `.path`, but received: {p}")
        if not len(self.path.parts):
            raise ValueError(
                f"Expected non-empty `.path`, but received: {self.path}")
        # parse dtype
        match self.dtype:
            case str() | np.dtype() | type():
                self.dtype = np.dtype(self.dtype).str
            case t:
                raise TypeError(
                    f"Expected (str|np.dtype) in `.dtype` for output variable "
                    f"\"{self.path}\", but received: {t}")
        # parse unit
        match self.unit:
            case None:
                pass
            case "":
                raise TypeError(
                    f"Use `None` instead of \"\" in `.unit` for output variable "
                    f"\"{self.path}\".")
            case str(u):
                if not (u.startswith("[") and u.endswith("]")):
                    raise ValueError(
                        f"Expected \"[...]\" in `.unit` for output variable "
                        f"\"{self.path}\", but received: {u}")
            case pint.Unit() as u:
                self.unit = f"[{u}]"
            case unum.Unum() as u:
                if u.asNumber() != 1.0:
                    raise ValueError(
                        f"No numerical value expected in `.unit` for "
                        f"output variable \"{self.path}\", but received: {u}")
                self.unit = u.strUnit()
            case u:
                raise TypeError(
                    f"Expected (None|str|pint.Unit|unum.Unum) in `.unit` for"
                    f"output variable \"{self.path}\", but received: {u}")
        if not isinstance(self.codecs, dict):
            raise TypeError(
                f"Expected (dict) in `.codecs` for output variable "
                f"\"{self.path}\", but received: {self.codec}")

    @classmethod
    def from_dict(cls, config: dict[str, Any], /) -> Self:
        return cls(**config)

    def to_dict(self) -> dict[str, str | None]:
        return {"path": str(self.path), "dtype": self.dtype, "unit": self.unit}

    # ~~~~~~~~~~~~~~~~~ #

    @property
    def var_name(self) -> str:
        return self.path.name


# ==============================================================================


@dataclass(kw_only=True)
class TreeView:
    """
    A mapping from a Vivarium schema to a partial :py:class:`~xarray.DataTree`
    specification, assuming that the input schema has a uniform metadata
    provider --- i.e., either metadata with a common root path, or no metadata
    at all.

    Example JSON configuration::

      {
        "root": ["log_update", "ecoli-metabolism", "listeners"],
        "variables": {
          "fba_results": {
            "coefficient": [{...}],
            "reaction_fluxes": [{...}],
            "external_exchange_fluxes": [{...}]
          },
          "enzyme_kinetics": {
            "counts_to_molar": [{...}],
            "actual_fluxes": [{...}],
            "target_fluxes": [{...}]
          }
        }
      }

    Here, ``variables`` represents a Vivarium schema, and should map directly
    onto the :py:class:`~vivarium.core.store.Store` hierarchy under ``root``.
    Each ``[{...}]`` is parsed into a :py:class:`.LeafView`, and is enclosed by
    a JSON array in order to distinguish it syntactically from the arbitrary
    nesting of JSON objects.
    """

    #: Path within an agent store, which is used for extracting both
    #: simulation data and associated metadata. Provided as a
    #: :py:data:`~vivarium.core.types.HierarchyPath`, and parsed into an
    #: :py:class:`.EmitPath`.
    root: EmitPath
    #: Flag for extracting coordinate annotations from the result of
    #: :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.output_metadata`.
    metadata: bool = True
    #: Schema relative to :py:attr:`.root`, holding :py:class:`.LeafView`
    #: arguments as leaf values.
    variables: dict[str, Any]

    #: Input :py:data:`~vivarium.core.types.HierarchyPath`\ s
    #: extracted from :py:attr:`.variables`.
    paths: list[HierarchyPath] = field(init=False)
    #: Output :py:class:`.LeafView`\ s parsed from :py:attr:`.variables`.
    leaves: list[LeafView] = field(init=False)

    def __post_init__(self) -> None:
        self.root = EmitPath(self.root)
        assert not self.root.type.is_agent
        assert isinstance(self.metadata, bool)
        assert isinstance(self.variables, dict)
        if not len(self.variables):
            raise self._arg_error("Missing arguments", "")
        paths, leaves = map(list, zip(*dict_to_paths((), self.variables)))
        if not all(map(len, paths)):
            raise self._arg_error("Empty path", "...")
        if len(frozenset(paths)) != len(paths):
            raise self._arg_error("Duplicate paths", "...")
        for (path, leaf) in zip(paths, leaves):
            if not len(leaf) == 1:
                raise self._arg_error(
                    f"Expected single output spec for path f{path}", "...")
        self.paths = paths
        self.leaves = list(map(LeafView.from_dict, map(itemgetter(0), leaves)))

    def _arg_error(self, msg: str, variables: str) -> ValueError:
        return ValueError(emitter_arg_error(
            self, msg,
            f"\"view\": ["
            f"\n      {{\"root\": {self.root.path},"
            f"\n        \"variables\": {{{variables}}}}}]"))

    @classmethod
    def from_dict(cls, config: dict[str, Any], /) -> Self:
        return cls(**config)

    def to_dict(self) -> dict[str, str | bool | dict[str, Any]]:
        return {"root": self.root.path, "metadata": self.metadata,
                "variables": self.variables}

    # ~~~~~~~~~~~~~~~~~ #

    @cached_property
    def emitting_paths(self) -> list[HierarchyPath]:
        r"""
        Collection of :py:attr:`.EmitPath.emitting_path`\ s.
        """
        if self.root.type.is_update:
            return [self.root.emitting_path]
        else:
            return [self.root.emitting_path + p for p in self.paths]

    @cached_property
    def emitted_paths(self) -> list[HierarchyPath]:
        """
        Composition of :py:attr:`.root` and :py:attr:`.paths`.
        """
        return [self.root.path + p for p in self.paths]

    @cached_property
    def emitted_prefix_paths(self) -> list[HierarchyPath]:
        """
        If a path is contained in :py:attr:`.emitted_paths` but not in
        :py:attr:`.emitting_paths`, then it should have a prefix matching this
        collection.
        """
        if self.root.type.is_update:
            return [self.root.emitting_path]
        else:
            return []

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def serialize_coord(coo: Any, /) -> np.ndarray | None:
        """
        Called by: :py:meth:`.make_coords`.
        """
        return None if coo is None else np.array(coo, copy=None)

    def make_coords(
        self, coords: dict[str, Any], /
    ) -> Iterator[np.ndarray | None]:
        """
        Extract coordinate annotations from the result of
        :py:meth:`.EcoliSim.output_metadata`.

        Called by: :py:meth:`.XarrayBuffer.assemble`.

        Calls: :py:meth:`.serialize_coord`.

        Args:
          coords: Result of :py:meth:`.XarrayEmitter.extract_coords`.
        """
        if self.metadata:
            root: dict[str, Any] = get_in(coords, self.root.metadata_path)
            for p in self.paths:
                yield self.serialize_coord(get_in(root, p))
        else:
            for _ in self.paths:
                yield None


# ==============================================================================


@dataclass
class ForestView:
    """
    Specification for how a collection of Vivarium schema entries and their
    metadata should be mapped onto a complete :py:class:`~xarray.DataTree`.

    Example JSON configuration::

      [
        {...},
        {...},
        {...}
      ]

    Here, each ``{...}`` is parsed into a :py:class:`.TreeView`.
    """

    #: Full schema with :py:class:`.LeafView` leaves.
    forest: list[TreeView]

    def __post_init__(self) -> None:
        assert isinstance(self.forest, list)
        assert all(isinstance(t, TreeView) for t in self.forest)
        if not len(self.forest):
            raise ValueError(emitter_arg_error(
                self, "Missing arguments", "\"view\": [...]"))
        roots = [t.root.path for t in self.forest]
        if len(frozenset(roots)) != len(roots):
            raise ValueError(emitter_arg_error(
                self, "Duplicate roots", "\"view\": [...]"))

    @classmethod
    def from_dict(cls, config: list[dict[str, Any]], /) -> Self:
        return cls(list(map(TreeView.from_dict, config)))

    def to_dict(self) -> list[dict[str, str | bool | dict]]:
        return [t.to_dict() for t in self.forest]

    # ~~~~~~~~~~~~~~~~~ #

    @cached_property
    def emitting_paths(self) -> list[HierarchyPath]:
        """
        Union of :py:attr:`.TreeView.emitting_paths`.
        """
        return [p for t in self.forest for p in t.emitting_paths]

    @cached_property
    def emitted_paths(self) -> list[HierarchyPath]:
        """
        Union of :py:attr:`.TreeView.emitted_paths`.
        """
        return [p for t in self.forest for p in t.emitted_paths]

    def matches_emitted_prefix_path(self, path: HierarchyPath, /) -> bool:
        """
        Compare ``path`` against the union of
        :py:attr:`.TreeView.emitted_prefix_paths`.
        """
        return any(path[:len(p)] == p
                   for t in self.forest for p in t.emitted_prefix_paths)

    @cached_property
    def leaves(self) -> dict[HierarchyPath, LeafView]:
        r"""
        Map emitted Vivarium paths onto :py:class:`.LeafView`\ s.

        Called by: :py:meth:`.XarrayBuffer.assemble`.
        """
        return dict(chain.from_iterable(
            zip(t.emitted_paths, t.leaves) for t in self.forest))
