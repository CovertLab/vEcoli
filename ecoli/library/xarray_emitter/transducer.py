
"""
Core data structures and logic of :py:mod:`.xarray_emitter`.

All other submodules of :py:mod:`.xarray_emitter` are either interfacing with
upstream (:py:class:`~vivarium.core.engine.Engine`) or downstream
(:py:class:`.AsyncBufferWriter`) APIs, or configuring those interfaces.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast, TYPE_CHECKING

import xarray
from xarray import Dataset, DataTree
from xarray.core.datatree import NodePath

from vivarium.core.types import HierarchyPath
from vivarium.library.topology import get_in, dict_to_paths

from .emit_predicate import ConjunctiveEmitPredicate
from .view import ForestView
from .storage import XarrayStoragePartition, VariableSpec, VariableEncoding
from .utils import emitter_arg_error, indent

if TYPE_CHECKING:
    from .writer import AsyncBufferWriter


# ==============================================================================


@dataclass
class XarrayBuffer:
    """
    Memory layout for the simulation data held by :py:class:`.XarrayTransducer`.

    This class contains only the logic required for marshalling simulation data
    into an in-memory hierarchical representation that is aligned with the
    output :ref:`storage <storage_layout>` and :ref:`variable <variable_layout>`
    layouts.

    .. note::
      :py:class:`.XarrayBuffer` *does not* use a `chunked array library`_ for
      its in-memory Xarray data structures, because it depends on
      "chunk-unaware" in-place operations for optimizing sequential emission
      performance. However, the :py:class:`.AsyncZarrBufferWriter` backend
      *does* control the `Zarr chunks`_ used for writing to persistent storage.

    .. _chunked array library: https://docs.xarray.dev/en/stable/internals/chunked-arrays.html
    .. _Zarr chunks: https://docs.xarray.dev/en/stable/user-guide/io.html#specifying-chunks-in-a-zarr-store
    """

    #: Statically configured variable layout transformation.
    view: ForestView
    #: Dynamic metadata, received via :py:meth:`!Engine._emit_configuration`.
    partition: XarrayStoragePartition = field(init=False)

    #: Descriptor for the time variable.
    time_spec: VariableSpec = field(init=False)
    #: Descriptors for simulation variables.
    var_specs: dict[NodePath, VariableSpec] = field(default_factory=dict)

    #: Root node of the :py:class:`~xarray.DataTree` buffer, holding the `Xarray
    #: attribute`_ for simulation metadata and the cyclic buffer for the
    #: `Xarray dimension coordinate`_ for simulation time.
    #:
    #: .. _Xarray attribute: https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-DataTree
    #: .. _Xarray dimension coordinate: https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Dimension-coordinate
    root: Dataset = field(init=False)
    #: Child arrays of the :py:class:`~xarray.DataTree` buffer, holding the
    #: `Xarray coordinates`_ for simulation variables.
    #:
    #: .. _Xarray coordinates: https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Coordinate
    child_coords: dict[NodePath, Dataset] = field(default_factory=dict)
    #: Child arrays of the :py:class:`~xarray.DataTree` buffer, holding the
    #: cyclic buffers for the `Xarray data variables`_ for simulation variables.
    #:
    #: .. _Xarray data variables: https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Variable
    child_vars: dict[NodePath, Dataset] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert isinstance(self.view, ForestView)

    # ~~~~~~~~~~~~~~~~~ #

    @property
    def time_coo(self) -> str:
        """
        Reference to :py:attr:`.XarrayStoragePartition.time_coo_name`.
        """
        return self.partition.time_coo_name

    @property
    def time_var(self) -> str:
        """
        Reference to :py:attr:`.XarrayStoragePartition.time_var_name`.
        """
        return self.partition.time_var_name

    @cached_property
    def output_paths(self) -> dict[HierarchyPath, tuple[NodePath, str]]:
        """
        Mapping from input (Vivarium store) to output (Xarray node/variable)
        hierarchy locations for simulation variables, as defined by
        :py:attr:`.view` and :py:attr:`.partition`.

        Used by: :py:meth:`.write`.
        """
        return {path: (leaf.path, self.partition.dynamic_suffix)
                for (path, leaf) in self.view.leaves.items()}

    @cached_property
    def modified_paths(self) -> set[NodePath]:
        """
        Relative paths inside the independent substore that are modified during
        a *daughter* generation. This information may be used by
        :py:class:`.AsyncBufferWriter` backends for maintaining metadata
        consistency.
        """
        return {NodePath()}

    @cached_property
    def added_paths(self) -> set[NodePath]:
        """
        Relative paths inside the independent substore that are added during a
        *daughter* generation. This information may be used by
        :py:class:`.AsyncBufferWriter` backends for maintaining metadata
        consistency.
        """
        root_paths = set(map(NodePath, self.root._variables.keys()))
        child_var_paths = set(
            path / cast(str, var)
            for (path, node) in self.child_vars.items()
            for var in node._variables.keys())
        return child_var_paths | root_paths

    # ~~~~~~~~~~~~~~~~~ #

    def check_layout(self) -> None:
        """
        Basic consistency check, performed before each buffer-level operation.
        """
        assert len(self.child_coords) == len(self.child_vars)

    def assemble(self, partition: XarrayStoragePartition, coords: dict) -> None:
        """
        Compute :py:attr:`.var_specs` by combining the static configuration
        :py:attr:`.view` with dynamically obtained metadata.

        Called by: :py:meth:`.XarrayTransducer.alloc`.

        Calls: :py:meth:`.TreeView.make_coords` and :py:class:`.VariableSpec`.

        Args:
          partition: Result of :py:meth:`.XarrayEmitter.extract_partition`.
          coords:    Result of :py:meth:`.XarrayEmitter.extract_coords`.
        """
        assert isinstance(partition, XarrayStoragePartition)
        self.partition = partition
        assert not(self.var_specs)
        for tree in self.view.forest:
            for (lf, coo) in zip(tree.leaves, tree.make_coords(coords)):
                self.var_specs[lf.path] = VariableSpec(
                    partition=partition, coord=coo,
                    var_name=lf.var_name, dtype=lf.dtype, unit=lf.unit,
                    codecs=lf.codecs)

    def alloc(self, buf_size: int, metadata: dict) -> None:
        """
        Allocate the in-memory Xarray data structures defined by
        :py:attr:`.time_spec` and :py:attr:`.var_specs`.

        Called by: :py:meth:`.XarrayTransducer.alloc`.

        Calls: :py:meth:`.VariableSpec.alloc_metadata`,
        :py:meth:`.VariableSpec.alloc_time`,
        :py:meth:`.VariableSpec.alloc_coord` and
        :py:meth:`.VariableSpec.alloc_var`.

        Args:
          buf_size: :py:attr:`.XarrayTransducer.buf_size`.
          metadata: Result of :py:meth:`.XarrayEmitter.extract_metadata`.
        """
        assert not(self.child_coords)
        self.time_spec = VariableSpec.make_time(self.partition, buf_size)
        self.root = self.time_spec.alloc_time(buf_size).assign_attrs(
            VariableSpec.alloc_metadata(self.partition, metadata)._attrs)
        for (path, var) in self.var_specs.items():
            self.child_coords[path] = var.alloc_coord()
            self.child_vars[path] = var.alloc_var(buf_size)

    def write(
        self, buf_tix: int, sim_tix: int, t: float, data: dict[str, Any], /
    ) -> None:
        """
        Marshal the simulation data for a single emit step into the output
        buffer.

        Called by: :py:meth:`XarrayTransducer.step`.

        Args:
          buf_tix: :py:attr:`.XarrayTransducer.buf_tix`.
          sim_tix: :py:attr:`.XarrayTransducer.sim_tix`.
          t:       Simulation time stamp.
          data:    Input received from :py:meth:`!Engine._emit_store_data`.
        """
        # index into buffer along time coordinate
        t_ix = {self.time_coo: buf_tix}

        # write time stamp to buffer
        self.root[self.time_var][t_ix] = t

        # strip agent prefix and remove schema paths with empty emit values
        agent_path = ("agents", self.partition.agent_id)
        emit_data = dict_to_paths((), get_in(data, agent_path))

        # check for expected emit paths
        emit_queue = set(self.view.emitted_paths)
        for (v_path, val) in emit_data:
            # find output schema location
            match self.output_paths.get(v_path):
                case None:
                    # `v_path` is not an expected emitted path
                    if sim_tix == 0:
                        # executed inside `Engine.__init__()`,
                        # and hence before `XarrayEmitter.reset_emit_flags()`
                        continue
                    if self.view.matches_emitted_prefix_path(v_path):
                        # ignored member of an expected emitted store
                        continue
                    raise KeyError(f"Unexpected emit path: {v_path}")
                case (x_node, x_var):
                    # write to output schema location inside buffer
                    self.child_vars[x_node][x_var][t_ix] = val
                    emit_queue.discard(v_path)
        if len(emit_queue) and sim_tix > 0:
            raise KeyError(f"Missing emit paths: {list(emit_queue)}")

    def render(
        self, writer: AsyncBufferWriter | None, buf_size: int,
        *, include_static: bool, copy: bool
    ) -> tuple[xarray.DataTree, dict[str, VariableEncoding]]:
        r"""
        Assemble the output buffer components.

        Called by: :py:meth:`.XarrayTransducer.flush`.

        Calls: :py:meth:`.VariableSpec.encoding` and
        :py:meth:`xarray.DataTree.from_dict`.

        Args:
          writer:         Used for choosing backend-specific
                          :py:type:`.VariableEncoding`\ s.
          buf_size:       :py:attr:`.XarrayTransducer.buf_size`.
          include_static: Include :py:attr:`.child_coords`
                          and all :py:type:`.VariableEncoding`\ s.
          copy:           Return a deep copy of arrays.

        .. note::
          The deep copy performed here is a conservative choice, which allows
          the previously allocated buffer to be immediately reused for
          subsequent writes, without relying on private implementation details
          about whether and when Xarray or storage backends copy data during
          their validation, encoding and serialization phases.

          Another similar option would be to force deep copying via a custom
          ``encoder`` argument to
          :py:func:`!xarray.backends.writers.dump_to_store`. However, this would
          merely delay the moment at which the output buffer is handed off from
          the main thread to the writer thread.

          An alternative conservative choice would be to allocate a new buffer
          for subsequent writes; this is the approach taken in
          :py:meth:`.ParquetEmitter.emit`. The present choice is premised on the
          assumption that, while allocating new arrays is faster than copying
          arrays, it may be advantageous not to force the hardware cache and
          main memory to adjust to a new heap representation of the output
          variable hierarchy along the :py:meth:`.write` code path, particularly
          when many simulations are running in parallel on the same compute
          node.
        """
        # fetch root node
        root = {NodePath(): self.root._copy(deep=True) if copy else self.root}

        # fetch child nodes
        assert set(self.child_coords) == set(self.child_vars)
        match (include_static, copy):
            case (False, False):
                children = self.child_vars
            case (False, True):
                children = {
                    p: n._copy(deep=True)
                    for (p, n) in self.child_vars.items()}
            case (True, False):
                children = {
                    # `self.child_vars[p]` holds only `data_vars` by construction
                    p: c.assign(self.child_vars[p]._variables)
                    for (p, c) in self.child_coords.items()}
            case (True, True):
                children = {
                    p: c._copy(deep=True).assign({
                        k: v._copy(deep=True)
                        for (k, v) in self.child_vars[p]._variables.items()})
                    for (p, c) in self.child_coords.items()}

        # assemble nodes
        buf = DataTree.from_dict(cast(dict[str, Dataset], root | children))

        # check consistency between composition logic and update logic
        assert set(str(NodePath("/") / p.parent)
                   for p in (self.added_paths | self.modified_paths)
                   ).issubset(buf.groups)

        # fetch encodings
        enc: dict[str, VariableEncoding] = {}
        if include_static and writer is not None:
            enc |= {"": self.time_spec.encoding(writer, buf_size)}
            enc |= {str(path): var.encoding(writer, buf_size)
                    for (path, var) in self.var_specs.items()}
        return (buf, enc)

    def get_time(self, buf_tix: int) -> float:
        """
        Called by: :py:meth:`.XarrayTransducer.flush`.

        Args:
          buf_tix: :py:attr:`.XarrayTransducer.buf_tix`.
        """
        t_ix = {self.time_coo: buf_tix}
        return self.root[self.time_var][t_ix].values.item()

    def shift(self, buf_size: int) -> None:
        """
        Called by: :py:meth:`.XarrayTransducer.shift`.

        Args:
          buf_size: :py:attr:`.XarrayTransducer.buf_size`.
        """
        self.root.coords[self.time_coo] = self.root.coords[self.time_coo] + buf_size

    def truncate(self, buf_tix: int) -> None:
        """
        Called by: :py:meth:`.XarrayTransducer.truncate`.

        Args:
          buf_tix: :py:attr:`.XarrayTransducer.buf_tix`.
        """
        time_sel = {self.time_coo: slice(0, buf_tix)}
        self.root = self.root.isel(time_sel)
        for (path, var) in self.child_vars.items():
            self.child_vars[path] = var.isel(time_sel)

    def clear(self) -> None:
        """
        Called by: :py:meth:`.XarrayTransducer.clear`.
        """
        self.root = Dataset()
        self.child_coords = {}
        self.child_vars = {}


# ==============================================================================


class XarrayTransducer:
    """
    Essential logical state of :py:class:`.XarrayEmitter`, managing a cyclic
    buffer of hierarchically organized arrays.

    This class establishes the temporal coupling with
    :py:class:`~vivarium.core.engine.Engine` and :py:class:`.AsyncBufferWriter`,
    whereas the :py:class:`.XarrayBuffer` instance it owns is responsible for
    transforming and holding simulation data.

    Example JSON configuration::

      {
        "predicate": [...],
        "buffer": {
          "size": 3
        }
      }

    Here,

      - ``predicate`` defines the criterion for which *simulation steps* also
        become *emit steps*, and is parsed by
        :py:class:`.ConjunctiveEmitPredicate`,
      - while ``size`` is the number of *emit steps* stored in memory by
        :py:class:`.XarrayBuffer`.

    .. note::
      The parameter ``size`` is intended to constrain the memory cost of each
      simulation process, when many parallel simulations are executed in
      parallel on a node with shared memory. Within that memory budget, larger
      buffer sizes will result in fewer calls to the transport layer.
    """

    __slots__ = (
        "__dict__", "predicate", "buffer",
        "buf_size", "buf_tix", "sim_tix", "debug"
    )

    def __init__(self, config: dict[str, Any], /, *, debug: bool=False) -> None:
        self.validate_config(_config := config["transducer"])

        self.predicate = ConjunctiveEmitPredicate.build(_config["predicate"])
        """ Criterion for which *simulation steps* also become *emit steps*. """

        view = ForestView.from_dict(config["view"])
        self.buffer: XarrayBuffer = XarrayBuffer(view)
        """ In-memory cyclic buffer for simulation data. """

        self.buf_size: int = _config["buffer"]["size"]
        """ Size of time dimension. """
        self.buf_tix: int = 0
        """
        Current relative *emit step* inside the cyclic buffer; advanced at the
        end of a :py:meth:`.step` call.
        """
        self.sim_tix: int = 0
        """
        Current absolute *simulation step*; advanced at the end of a
        :py:meth:`.step` call.
        """
        self.debug: bool = debug
        """ Flag for debug-level printing. Defaults to ``False``. """

    @classmethod
    def validate_config(cls, config: dict[str, Any], /) -> None:
        match config.get("predicate"):
            case None:
                raise KeyError(emitter_arg_error(
                    cls, "Missing argument", "\"buffer\": {\"size\": ...}"))
        match config.get("buffer", {}).get("size"):
            case None:
                raise KeyError(emitter_arg_error(
                    cls, "Missing argument", "\"buffer\": {\"size\": ...}"))
            case int(buf_size) if buf_size > 2:
                pass
            case buf_size:
                raise TypeError(emitter_arg_error(
                    cls, "Invalid argument",
                    f"\"buffer\": {{\"size\": {buf_size}}}"))

    def __str__(self) -> str:
        return self.display(self.buffer.render(
            None, self.buf_size, include_static=True, copy=False)[0])

    def display(self, buf: DataTree, /) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  buf_size: {self.buf_size}\n"
            f"  sim_tix: {self.sim_tix}, buf_tix: {self.buf_tix}\n"
            f"  buffer:{indent(4, buf)}")

    # ~~~~~~~~~~~~~~~~~ #

    def check_buffer(self) -> None:
        """
        Basic consistency check, performed before each buffer-level operation.
        """
        assert 0 <= self.buf_tix <= self.buf_size
        self.buffer.check_layout()

    def alloc(
        self, *, partition: XarrayStoragePartition, metadata: dict, coords: dict
    ) -> None:
        """
        Allocate an :py:class:`XarrayBuffer` conforming to the output schema.
        This buffer will be populated by :py:meth:`.step` at every *emit step*,
        and wll later be sent to the transport layer via :py:meth:`.flush`.

        Called by: :py:meth:`.XarrayEmitter.emit`.

        Args:
          metadata: Result of :py:meth:`.XarrayEmitter.extract_metadata`.
          coords: Result of :py:meth:`.XarrayEmitter.extract_coords`.
        """
        self.check_buffer()
        self.buffer.assemble(partition, coords)
        self.buffer.alloc(self.buf_size, metadata)
        self.check_buffer()

    def step(self, data: dict[str, Any], /) -> bool:
        r"""
        If :py:attr:`.predicate` is satisfied for the current *simulation step*,
        then create a new *emit step* by writing the simulation data into
        :py:attr:`.buffer`.

        Called by: :py:meth:`.XarrayEmitter.emit`.

        Calls: :py:meth:`.ConjunctiveEmitPredicate.__call__` and
        :py:meth:`.XarrayBuffer.write`.

        Args:
          data: Payload from :py:meth:`.XarrayEmitter.emit`.

        Returns:
          `False` if the buffer is full and the operation cannot be performed
          without first :py:meth:`.flush`\ ing, otherwise `True`.
        """
        if self.predicate(self.sim_tix, t := get_in(data, ("time",)), data):
            if self.buf_tix < self.buf_size:
                # fill current emit step
                self.buffer.write(self.buf_tix, self.sim_tix, t, data)
                # increment emit step
                self.buf_tix += 1
            else:
                # writing now would result in an `IndexError`
                return False
        # increment simulation step
        self.sim_tix += 1
        return True

    def flush(
        self, writer: AsyncBufferWriter, *, include_static: bool, final: bool
    ) -> tuple[xarray.DataTree, dict[str, VariableEncoding], dict[str, Any]]:
        r"""
        Assemble the output buffer that will be sent to persistent storage, and
        perform associated cache and memory management tasks.

        Called by: :py:meth:`.AsyncBufferWriter.write`.

        Calls: :py:meth:`.XarrayBuffer.render` and
        :py:meth:`.AsyncBufferWriter.merge_attributes`.

        Args:
          writer:         Used for choosing backend-specific
                          :py:type:`.VariableEncoding`\ s and for combining
                          metadata.
          include_static: Include :py:attr:`.XarrayBuffer.child_coords`
                          and all :py:type:`.VariableEncoding`\ s.
          final:          Indicate the final buffer.

        Returns:
          - A deep copy of the in-memory buffer.
          - Backend-specific variable encodings, only if ``include_static``.
          - A JSON-serializable reference to the latest emitted simulation step.
        """
        self.check_buffer()
        if final:
            assert not include_static
            if self.buf_tix < self.buf_size:
                # at least one unfilled emit step inside allocated buffer
                self.truncate()
        else:
            assert self.buf_tix == self.buf_size
        (buf, enc) = self.buffer.render(
            writer, self.buf_size,
            include_static=include_static, copy=not final)
        writer.merge_attributes(buf)
        ref = {"sim_step": self.sim_tix,
               "sim_time": self.buffer.get_time(self.buf_tix - 1)}
        if final:
            # reference to buffer components no longer needed
            self.clear()
        if self.debug:
            hline = "-" * 79
            print(hline, "\n", self.display(buf), "\n", hline)
        return (buf, enc, ref)

    def shift(self) -> None:
        """
        Shift the time coordinate by the buffer size, without modifying the
        buffer content otherwise. This is used after a full buffer has been
        flushed, and before new values are written to it.

        Calls: :py:meth:`.XarrayBuffer.shift`.
        """
        self.check_buffer()
        assert self.buf_tix == self.buf_size
        self.buf_tix = 0
        self.buffer.shift(self.buf_size)

    def truncate(self) -> None:
        """
        Remove excess buffer space before flushing the final buffer.

        Called by: :py:meth:`.flush`.

        Calls: :py:meth:`.XarrayBuffer.truncate`.
        """
        self.buf_size = self.buf_tix
        self.buffer.truncate(self.buf_tix)

    def clear(self) -> None:
        """
        Empty all buffer components after flushing the final buffer.

        Called by: :py:meth:`.flush`.

        Calls: :py:meth:`.XarrayBuffer.clear`.
        """
        self.buf_tix = 0
        self.sim_tix = 0
        self.buffer.clear()
