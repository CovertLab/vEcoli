
"""
Constants and parameters defining the output :ref:`storage <storage_layout>` and
:ref:`variable <variable_layout>` layouts.
"""


from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import Any, Self, TYPE_CHECKING

import numpy as np
from xarray import Dataset
from xarray.core.datatree import NodePath

from ..emitter import StoragePartition

if TYPE_CHECKING:
    from .writer import AsyncBufferWriter


# ==============================================================================
# constants
# ==============================================================================


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
# Xarray storage layout
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
        return Path(*(f"{k}={getattr(self, k)}" for k in
                      ["experiment_id", "variant", "lineage_seed"]))

    # ~~~~~~~~~~~~~~~~~ #

    @cached_property
    def dynamic_suffix(self) -> str:
        """
        Uniquely identifying suffix path for variables which occur in multiple
        realisations within an independent substore.
        """
        return str(NodePath(*(f"{k}={getattr(self, k)}" for k in
                              ["generation"])))

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

    @cached_property
    def time_var_name(self) -> str:
        """
        Name of the real-valued `Xarray data variable`_ holding simulation timestamps.
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


type VariableEncoding = dict[str, Any]


# ==============================================================================


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
    coord: np.ndarray | None
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
        self, writer: AsyncBufferWriter, buf_size: int, /
    ) -> dict[str, VariableEncoding]:
        """
        Parameters used for writing a variable array and its coordinate array to
        persistent storage, including chunk sizes and compression codecs.

        Called by: :py:meth:`.XarrayBuffer.render`.

        Calls: :py:meth:`.AsyncBufferWriter.coo_codecs` and
        :py:meth:`.AsyncBufferWriter.var_codecs`.
        """
        b = writer.config["buffers_per_chunk"]
        # coordinate encoding
        match (self.is_time, self.coord):
            case (False, None):
                coo_enc = {}
            case (False, np.ndarray() as coo):
                coo_enc = {self.coo_name: {
                    # use 1 storage chunk for the coordinate array
                    "chunks": coo.shape} | writer.coo_codecs(self)}
            case (True, np.ndarray() as coo):
                assert coo.shape == (buf_size,)
                coo_enc = {self.coo_name: {
                    # use 1 storage chunk for `b` buffers of the time coordinate
                    "chunks": (b * buf_size,)} | writer.coo_codecs(self)}
        # variable encoding
        var_enc = {self.datavar_name: {
            # use 1 storage chunk for `b` buffers of simulation data
            "chunks": self.dims(b * buf_size)} | writer.var_codecs(self)}
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
            attrs={} if self.unit is None else {self.datavar_name: self.unit})

    def alloc_var(self, buf_size: int, /) -> Dataset:
        """
        Allocate the `Xarray data variable`_ for an output variable, which *is*
        placed under :py:attr:`XarrayStoragePartition.dynamic_suffix`.

        Called by: :py:meth:`.XarrayBuffer.alloc`.
        """
        return Dataset(data_vars={
            self.datavar_name: (self.dim_names, self.zeros(buf_size))})
