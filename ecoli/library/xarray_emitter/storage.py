
"""
Constants, parameters and iterators defining the output :ref:`storage
<storage_layout>` and :ref:`variable <variable_layout>` layouts.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Generator, Mapping
from dataclasses import astuple, dataclass, field, fields
from functools import cached_property
from glob import glob
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING

import numpy
import numpy as np
from fsspec import get_fs_token_paths
from xarray import Dataset

from ..emitter import StoragePartition

if TYPE_CHECKING:

    from typing import Any, Self

    from .writer import AsyncBufferWriter

# ==============================================================================
# constants
# ==============================================================================


EXPERIMENT_PREFIX = "experiment_id="
""" Prefix for a part of :py:attr:`.XarrayStoragePartition.independent_path`. """
VARIANT_PREFIX = "variant="
""" Prefix for a part of :py:attr:`.XarrayStoragePartition.independent_path`. """
LINEAGE_PREFIX = "lineage_seed="
""" Prefix for a part of :py:attr:`.XarrayStoragePartition.independent_path`. """


# ------------------------------------------------------------------------------


TIME_COO_PREFIX = "emitstep_"
""" Prefix for :py:attr:`.XarrayStoragePartition.time_coo_name`. """
TIME_VAR_PREFIX = "time_"
""" Prefix for :py:attr:`.XarrayStoragePartition.time_var_name`. """
VAR_COO_PREFIX = "id_"
""" Prefix for :py:attr:`.VariableSpec.var_coo_name`. """
LOG_ATTR_PREFIX = "last_write_"
""" Prefix for :py:attr:`.XarrayStoragePartition.log_attr_name`. """
SUCCESS_ATTR_PREFIX = "division_reached_"
""" Prefix for :py:attr:`.XarrayStoragePartition.success_attr_name`. """

TIME_COO_DTYPE = np.dtype(np.uint32)
""" Data type for :py:attr:`.XarrayStoragePartition.time_coo_name`. """
TIME_VAR_DTYPE = np.dtype(np.float32)
""" Data type for :py:attr:`.XarrayStoragePartition.time_var_name`. """


# ==============================================================================
# workflow configuration
# ==============================================================================


@dataclass(kw_only=True, frozen=True)
class WorkflowConfig:
    """
    Simulation and analysis workflow configuration, parsed directly from JSON.

    This object is intended for use by analysis pipelines.
    """

    #: Full workflow configuration.
    sim: dict[str, Any]
    #: Indicates whether the workflow store is remote or local.
    is_uri: bool
    #: Variant parameters, parsed from :py:attr:`.sim`.
    variants: list[dict[str, Any]]

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Self:
        """
        Calls: :py:meth:`.build`.
        """
        with open(path, "r") as f:
            return cls.build(json.load(f))

    @classmethod
    def build(cls, config: dict[str, Any]) -> Self:
        """
        Parse and validate high-level information about the simulation workflow.

        Called by: :py:meth:`.load`.
        """
        assert config["experiment_id"]
        assert config["emitter"] == "xarray"
        assert any(map(config["emitter_arg"].__contains__, ["out_dir", "out_uri"]))

        from runscripts.create_variants import parse_variants
        variants = config["variants"]
        assert isinstance(variants, dict)
        assert len(variants) <= 1
        variants = (parse_variants(next(iter(variants.values())))
                    if variants else
                    [])
        return cls(is_uri="out_uri" in config["emitter_arg"],
                   sim=config, variants=variants)

    # ~~~~~~~~~~~~~~~~~ #

    def variant_params(self, variant: str) -> dict[str, Any]:
        """
        Translate a variant string into variant parameters.

        Calls: :py:meth:`.variant_index`.
        """
        ix = self.variant_index(variant)
        ix -= int(not self.sim.get("skip_baseline", False))
        return self.variants[ix] if ix >= 0 else {}

    @staticmethod
    def variant_index(variant: str) -> int:
        """
        Interpret a variant string as a pointer into :py:attr:`.variants`.

        Called by: :py:meth:`.variant_params`.
        """
        ix = int(variant.lstrip(VARIANT_PREFIX))
        assert variant == f"{VARIANT_PREFIX}{ix}"
        return ix


# ==============================================================================
# workflow storage layout
# ==============================================================================


@dataclass(slots=True, kw_only=True, frozen=True)
class WorkflowPaths:
    """
    Efficiently locate all independent substore paths within a simulation
    workflow that was emitted using the :py:class:`.XarrayStoragePartition` path
    scheme.

    This object is intended for use by analysis pipelines.
    """

    #: Root path to the persistent store of a simulation workflow.
    root: str
    #: Tree of located :py:attr:`.XarrayStoragePartition.independent_path`\ s.
    substores: dict[str, list[str]]

    def __post_init__(self) -> None:
        assert self.root.rsplit(sep="/", maxsplit=1)[-1].startswith(EXPERIMENT_PREFIX)
        for (variant, lineages) in self.substores.items():
            assert variant.startswith(VARIANT_PREFIX)
            assert all(lin.startswith(LINEAGE_PREFIX) for lin in lineages)

    # ~~~~~~~~~~~~~~~~~ #

    def __len__(self) -> int:
        """
        Number of independent substores.
        """
        return sum(map(len, self.substores.values()))

    def __iter__(self) -> Generator[Substore]:
        """
        Iterator over :py:attr:`.substores`.
        """
        for (variant, lineages) in self.substores.items():
            for lineage in lineages:
                yield Substore(variant, lineage)

    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    def locate(cls, config: WorkflowConfig) -> Self:
        """
        Find all substore paths using a single ``glob()`` call to the file
        system.
        """
        # load config
        assert isinstance(config, WorkflowConfig)
        experiment_id = config.sim["experiment_id"]
        emitter = config.sim["emitter_arg"]
        num_variants = len(config.variants)
        num_variants += int(not config.sim.get("skip_baseline", False))
        num_lineages = config.sim["n_init_sims"]

        # find workflow store
        if config.is_uri:
            store_path = join(emitter["out_uri"], experiment_id, "store")
            fs, _, store_path = get_fs_token_paths(store_path)
        else:
            store_path = join(emitter["out_dir"], experiment_id, "store")
            assert Path(store_path).exists()

        # find independent substore paths
        substore_glob = XarrayStoragePartition.independent_path_glob(experiment_id)
        substores = cls.group(
            fs.glob(join(store_path, substore_glob))
            if config.is_uri else
            glob(substore_glob, root_dir=store_path))

        # check consistency with workflow config
        assert set(substores.keys()) == {f"{VARIANT_PREFIX}{v}"
                                         for v in range(num_variants)}
        for lineages in substores.values():
            assert len(lineages) == num_lineages

        return cls(root=join(store_path, f"{EXPERIMENT_PREFIX}{experiment_id}"),
                   substores=substores)

    @staticmethod
    def group(paths: list[str]) -> dict[str, list[str]]:
        """
        Reassemble a partition hierarchy from a flat list of substore paths.
        """
        substores: dict[str, list[str]] = {}
        for path in paths:
            substores.setdefault(
                XarrayStoragePartition.get_variant(path), []
            ).append(XarrayStoragePartition.get_lineage(path))
        return substores


# ------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Substore:
    """
    Hashable identifier of an independent substore.
    """

    variant: str
    lineage: str

    def __post_init__(self) -> None:
        assert self.variant.startswith(VARIANT_PREFIX)
        assert self.lineage.startswith(LINEAGE_PREFIX)

    def __str__(self) -> str:
        return join(*astuple(self))

    @classmethod
    def identity(cls, path: Self) -> Self:
        return path

    @staticmethod
    def groupby_variant[ResultT](
        results: Mapping[Substore, ResultT]
    ) -> Mapping[str, Mapping[str, ResultT]]:
        """
        Group a hash map over substore identifiers into ``variant``/``lineage``
        levels.
        """
        grouped: dict[str, dict[str, ResultT]] = {}
        for (s, res) in results.items():
            grouped.setdefault(s.variant, {})[s.lineage] = res
        return grouped


# ==============================================================================
# substore storage layout
# ==============================================================================


@dataclass(eq=True, kw_only=True)
class XarrayStoragePartition(StoragePartition):
    """
    Relative storage paths and coordinate names used by
    :py:class:`.XarrayEmitter` to place the output from a single-generation
    :py:class:`.EcoliSim` within a workflow store.

    See :ref:`storage_layout` for the design rationale.
    """

    @classmethod
    def cast(cls, partition: StoragePartition) -> Self:
        assert isinstance(partition, StoragePartition)
        return cls(**{f.name: getattr(partition, f.name)
                      for f in fields(partition) if f.init})

    @classmethod
    def from_substore(
        cls, config: WorkflowConfig, substore: Substore, generation: int
    ) -> Self:
        assert isinstance(config, WorkflowConfig)
        assert isinstance(substore, Substore)
        assert isinstance(generation, int)
        return cls(
            experiment_id=config.sim["experiment_id"],
            variant=int(substore.variant.removeprefix(VARIANT_PREFIX)),
            lineage_seed=int(substore.lineage.removeprefix(LINEAGE_PREFIX)),
            agent_id="0" * generation)

    def __hash__(self):
        return hash(tuple(getattr(self, f.name) for f in fields(self)))

    # ~~~~~~~~~~~~~~~~~ #

    @cached_property
    def independent_path(self) -> Path:
        """
        The most specific location within a workflow store that has the
        following properties:

        - It holds a *stochastically independent* simulation subensemble.
        - It is *representationally independent*.

        A simulation subensemble is considered *stochastically independent* if
        no numerical values from other subensembles are involved in its
        simulation; Note that this concern is separate from the choices of
        software versions and parameters, which are coupled at the project
        level. A substorage is *representationally independent* if it is
        self-contained in terms of semantic coordinate annotations, and if it
        does not rely on any external synchronisation mechanism for maintaining
        the consistency of its storage layout metadata.
        """
        path_segments = {EXPERIMENT_PREFIX: self.experiment_id,
                         VARIANT_PREFIX: self.variant,
                         LINEAGE_PREFIX: self.lineage_seed}
        return Path(*(f"{pfx}{val}" for (pfx, val) in path_segments.items()))

    @staticmethod
    def independent_path_glob(experiment_id: str) -> str:
        return join(f"{EXPERIMENT_PREFIX}{experiment_id}",
                    f"{VARIANT_PREFIX}[0-9]*",
                    f"{LINEAGE_PREFIX}[0-9]*")

    @staticmethod
    def get_variant(path: str) -> str:
        return path.split(sep="/")[1]

    @staticmethod
    def get_lineage(path: str) -> str:
        return path.split(sep="/")[2]

    # ~~~~~~~~~~~~~~~~~ #

    @cached_property
    def dynamic_suffix(self) -> str:
        """
        Uniquely identifying suffix path for variables which occur in multiple
        realisations within an independent substore.
        """
        return f"generation={self.generation}"

    @cached_property
    def sim_id(self) -> str:
        """
        Suffix used in :py:attr:`.time_coo_name`. This information is logically
        equivalent to :py:attr:`.dynamic_suffix`.
        """
        return f"gen={self.generation}"

    # ~~~~~~~~~~~~~~~~~ #

    @cached_property
    def time_coo_name(self) -> str:
        r"""
        Name of the integer-valued `Xarray dimension coordinate`_ for the
        current simulation that is located in the root node of the output
        :py:class:`~xarray.DataTree`. All emitted `Xarray data variable`_\ s
        inherit this dimension coordinate, including :py:attr:`.time_var_name`.

        .. _Xarray data variable: https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Variable
        .. _Xarray dimension coordinate: https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Dimension-coordinate
        """
        return f"{TIME_COO_PREFIX}{self.sim_id}"

    @staticmethod
    def is_time_coo_name(key: str) -> bool:
        return key.startswith(TIME_COO_PREFIX)

    # ~~~~~~~~~~~~~~~~~ #

    @cached_property
    def time_var_name(self) -> str:
        """
        Name of the real-valued `Xarray data variable`_ holding simulation
        timestamps.
        """
        return f"{TIME_VAR_PREFIX}{self.sim_id}"

    # ~~~~~~~~~~~~~~~~~ #

    @cached_property
    def log_attr_name(self) -> str:
        """
        Attribute name used by :py:meth:`.AsyncBufferWriter.log_effect`.
        """
        return f"{LOG_ATTR_PREFIX}{self.sim_id}"

    @cached_property
    def success_attr_name(self) -> str:
        """
        Attribute name used by :py:meth:`.AsyncBufferWriter.mark_success`.
        """
        return f"{SUCCESS_ATTR_PREFIX}{self.sim_id}"


# ==============================================================================
# Xarray output variable
# ==============================================================================


type VariablePath = str
type VariableEncoding = dict[str, Any]


# ------------------------------------------------------------------------------


def var_name(path: VariablePath) -> VariablePath:
    """
    Extract the variable name from a full variable path.
    """
    return path.rsplit("/", maxsplit=1)[-1]


def coo_path(path: VariablePath) -> VariablePath:
    """
    Compute the coordinate path associated with a variable path.
    """
    return join(path, VariableSpec.var_coo_name(var_name(path)))


# ------------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True, frozen=True)
class VariableSpec:
    """
    Complete configuration of an output variable for :py:class:`.XarrayEmitter`,
    including:

    - its name, data type and metadata,
    - its coordinate data,
    - its allocation inside :py:class:`.XarrayBuffer`,
    - and its encoding for :py:class:`.AsyncBufferWriter`.

    This object is created by :py:meth:`.XarrayBuffer.assemble` from a
    :py:class:`.LeafView` and dynamic metadata.

    .. note::
      In accordance with `Xarray's view of the Zarr format`_, annotations are
      placed in :py:attr:`xarray.Dataset.attrs` rather than in
      :py:attr:`xarray.DataArray.attrs`.

    .. _Xarray's view of the Zarr format: https://docs.xarray.dev/en/stable/internals/zarr-encoding-spec.html
    """

    #: Simulation metadata.
    partition: XarrayStoragePartition
    #: Variable name, determining the output paths both of the coordinate array
    #: and of the data arrays. This is set automatically for the time variable.
    var_name: str
    #: Variable data type.
    dtype: str
    #: Unit annotation.
    unit: str | None
    #: Coordinate array.
    coord: numpy.ndarray | None
    #: Backend-specific configuration of compression codecs.
    codecs: dict[str, Any] = field(default_factory=dict)
    #: Flag for time variables.
    is_time: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.partition, XarrayStoragePartition)
        assert isinstance(self.var_name, str)
        assert isinstance(self.dtype, str)
        assert isinstance(self.unit, str | None)
        assert isinstance(self.coord, np.ndarray | None)
        assert isinstance(self.codecs, dict)
        assert isinstance(self.is_time, bool)
        assert bool(self.var_name) is not self.is_time
        assert not (self.is_time and self.coord is None)

    # ~~~~~~~~~~~~~~~~~ #

    @property
    def attr_name(self) -> str:
        """
        Attribute name used by :py:meth:`.alloc_coord`.
        """
        return (self.partition.time_var_name if self.is_time
                else self.var_name)

    @staticmethod
    def var_coo_name(var_name: str, /) -> str:
        """
        Name of the `Xarray coordinate`_ for a simulation variable.
        """
        return f"{VAR_COO_PREFIX}{var_name}"

    @property
    def coo_name(self) -> str:
        """
        Coordinate name used by :py:meth:`.alloc_coord`, which is either a
        :py:attr:`.XarrayStoragePartition.time_coo_name` or a
        :py:attr:`.var_coo_name`.
        """
        return (self.partition.time_coo_name if self.is_time
                else self.var_coo_name(self.var_name))

    @property
    def datavar_name(self) -> str:
        """
        Variable name used by :py:meth:`.alloc_var`.
        """
        return (self.partition.time_var_name if self.is_time
                else self.partition.dynamic_suffix)

    @property
    def dim_names(self) -> tuple[str, ...]:
        """
        Dimension names used by :py:meth:`.alloc_var`, which are composed of
        :py:attr:`.XarrayStoragePartition.time_coo_name` and
        :py:attr:`.var_coo_name`.
        """
        return (self.partition.time_coo_name,) + (
            () if self.coord is None or self.is_time
            else (self.var_coo_name(self.var_name),))

    # ~~~~~~~~~~~~~~~~~ #

    def dims(self, buf_size: int, /) -> tuple[int, ...]:
        """
        Dimension sizes used by :py:meth:`.zeros` and :py:meth:`.encoding`,
        which are composed of ``buf_size`` and the shape of :py:attr:`.coord`.
        """
        return (buf_size,) + (
            () if self.coord is None or self.is_time
            else (len(self.coord),))

    def zeros(self, buf_size: int, /) -> np.ndarray:
        """
        Allocate a buffer array for storing simulation data.
        """
        return np.zeros(self.dims(buf_size), dtype=self.dtype)

    def encoding(
        self, writer: AsyncBufferWriter, buf_size: int, *, include_coo: bool
    ) -> dict[str, VariableEncoding]:
        r"""
        Parameters used for writing a variable array and its coordinate array to
        persistent storage, including chunk sizes and compression codecs.

        Called by: :py:meth:`.XarrayBuffer.render`.

        Calls: :py:meth:`.AsyncBufferWriter.coo_codecs` and
        :py:meth:`.AsyncBufferWriter.var_codecs`.

        Args:
          writer:       Used for interpreting backend-specific codecs and for
                        retrieving the `writer.buffers_per_chunk` configuration.
          buf_size:     :py:attr:`.XarrayTransducer.buf_size`.
          include_coo:  Include :py:meth:`.AsyncBufferWriter.coo_codecs`.
        """
        b = writer.config["buffers_per_chunk"]
        # coordinate encoding
        coo_enc: dict[str, VariableEncoding] = {}
        match (include_coo, self.is_time, self.coord):
            case (True, False, np.ndarray() as coo):
                coo_enc |= {self.coo_name: writer.coo_codecs(self) | {
                    # use 1 storage chunk for non-time coordinate
                    "chunks": coo.shape}}
            case (True, True, np.ndarray() as coo):
                if writer.transducer.buf_shifts:
                    assert coo.shape == (buf_size,)
                coo_enc |= {self.coo_name: writer.coo_codecs(self) | {
                    # use 1 storage chunk for `b` buffers of time coordinate
                    "chunks": (b * buf_size,)}}
        # variable encoding
        var_enc = {self.datavar_name: writer.var_codecs(self) | {
            # use 1 storage chunk for `b` buffers of simulation data
            "chunks": self.dims(b * buf_size)}}
        return coo_enc | var_enc

    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    def make_time(
        cls, partition: XarrayStoragePartition, buf_size: int, /
    ) -> Self:
        """
        Create the :py:class:`.VariableSpec` for simulation time.

        Called by: :py:meth:`.alloc_time`.
        """
        # avoid circular import at module level
        from ecoli.processes.metabolism import TIME_UNITS
        assert isinstance(buf_size, int)
        return cls(
            partition=partition, var_name="",
            # type and units for real-valued time stamps
            dtype=TIME_VAR_DTYPE.str, unit=TIME_UNITS.strUnit(),
            # integer-valued Xarray dimension coordinate
            coord=np.arange(buf_size, dtype=TIME_COO_DTYPE),
            is_time=True)

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def alloc_metadata(
        partition: XarrayStoragePartition, metadata: dict, /
    ) -> Dataset:
        """
        Allocate the `Xarray attribute`_ for simulation metadata.

        Called by: :py:meth:`.XarrayBuffer.alloc`.

        .. _Xarray attribute: https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-DataTree
        """
        return Dataset(attrs={partition.sim_id: metadata})

    def alloc_time(self, buf_size: int, /) -> Dataset:
        """
        Allocate the `Xarray dimension coordinate`_ and `Xarray data variable`_
        for simulation time. ``self`` must be produced by :py:meth:`.make_time`.

        Called by: :py:meth:`.XarrayBuffer.alloc`.

        Calls: :py:meth:`.alloc_coord` and :py:meth:`.alloc_var`.
        """
        assert self.is_time and self.coord is not None
        assert self.coord.shape == (buf_size,)
        return self.alloc_coord().assign(self.alloc_var(buf_size)._variables)

    def alloc_coord(self) -> Dataset:
        """
        Allocate the `Xarray coordinate`_ and `Xarray attributes`_ for an output
        variable, which *are not* placed under
        :py:attr:`XarrayStoragePartition.dynamic_suffix`.

        Called by: :py:meth:`.XarrayBuffer.alloc`.

        .. _Xarray coordinate: https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Coordinate
        .. _Xarray attributes: https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Dataset
        """
        return Dataset(
            coords={} if self.coord is None else {self.coo_name: self.coord},
            attrs={} if self.unit is None else {self.attr_name: self.unit})

    def alloc_var(self, buf_size: int, /) -> Dataset:
        """
        Allocate the `Xarray data variable`_ for an output variable, which *is*
        placed under :py:attr:`XarrayStoragePartition.dynamic_suffix`.

        Called by: :py:meth:`.XarrayBuffer.alloc`.
        """
        return Dataset(data_vars={
            self.datavar_name: (self.dim_names, self.zeros(buf_size))})
