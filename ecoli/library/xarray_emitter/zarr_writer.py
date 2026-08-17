
"""
Transport and session layers for the Zarr storage backend.

This module defines subclasses that inherit from :py:mod:`.writer`, and modifies
the internal state in Xarray and Zarr where necessary.
"""


from __future__ import annotations

from asyncio import TaskGroup
from collections.abc import Coroutine, Mapping
from dataclasses import replace
from typing import Any, Literal, final

import zarr
from numpy import dtype
from xarray import DataTree
from xarray.backends import ZarrStore
from xarray.backends.writers import dump_to_store
from xarray.core.datatree import NodePath
from zarr.core.array import Array
from zarr.core.group import AsyncGroup, Group
from zarr.core.sync import sync

from .storage import VariableEncoding, VariableSpec
from .utils import WarningFilter, emitter_arg_error, filter_warnings
from .writer import AsyncArrayWriter, AsyncBufferWriter
from .zarr_utils import (
    _replace_consolidated_metadata,
    consolidate_metadata,
    get_group,
    parse_codecs,
    reconsolidate_metadata,
    zarr_warnings,
)

# ==============================================================================
# constants
# ==============================================================================


ZARR_ASYNC_CONCURRENCY: int = 4
""" Default bound on the number of Zarr's concurrent operations. """
ZARR_MAX_WORKERS: int = 4
""" Default bound on the size of Zarr's internal thread pool. """


# ==============================================================================
# Xarray internals
# ==============================================================================


def _datatree_to_zarr(
    dt: DataTree, store: ZarrStore, encoding: Mapping[str, Any] | None = None, /
) -> AsyncZarrArrayWriter:
    """
    Construct the :py:class:`.AsyncZarrArrayWriter` effect from a
    :py:class:`~xarray.DataTree` to an already open
    :py:class:`xarray.backends.ZarrStore`, possibly along some
    :py:attr:`!xarray.backends.ZarrStore._append_dim`.

    This function checks the following assumptions:

    - No `Dask chunks`_ are used within ``dt``.
    - In the first buffer, Zarr chunks are specified for all variables via
      ``encoding``.
    - In subsequent buffers, ``encoding`` is left empty.

    Adapted from: :py:meth:`!xarray.backends.writers._datatree_to_zarr`.

    .. _Dask chunks: https://docs.xarray.dev/en/stable/user-guide/dask.html
    """
    if encoding is None:
        encoding = {}
    if absolute := [p for p in encoding if p.startswith("/")]:
        raise ValueError(f"unexpected absolute paths in `encoding`: {absolute}")
    # TODO: fix in `_datatree_to_zarr()` (xarray==2026.04)
    encoding = {f"/{p}": e for (p, e) in encoding.items()}
    if unexpected := set(encoding.keys()) - set(dt.groups):
        raise ValueError(
            f"unexpected encoding group name(s) provided: {unexpected}")
    if any(dt.chunksizes.values()):
        raise ValueError("unexpected Dask chunks before Zarr export")

    writer = AsyncZarrArrayWriter()
    for (rel_path, node) in dt.subtree_with_keys:
        # materialise a node
        if not (len(node.dataset) or len(node.dataset.attrs)):
            # skip nodes without any data or metadata, in order to avoid
            # a failing check against `store._append_dim`
            continue
        elif node is dt:
            # root node
            ds = node.to_dataset(inherit=True)
            node_store = store
        else:
            # descendant node: do not duplicate stored coordinates
            ds = node.to_dataset(inherit=False)
            node_store = store.get_child_store(rel_path)
        # generate write operations for a node
        ds = node_store._validate_and_autodetect_region(ds)
        node_enc = encoding.get(node.path)
        if node_enc is None and encoding:
            raise KeyError(f"missing encoding for \"{node.path}\"")
        # TODO: fix in `_datatree_to_zarr()` (xarray==2026.04)
        node_store._validate_encoding(node_enc)
        dump_to_store(ds, node_store, writer, encoding=node_enc)
    return writer


# ==============================================================================
# array writer
# ==============================================================================


@final
class AsyncZarrArrayWriter(AsyncArrayWriter[Array]):
    """
    Implementation of asynchronous write operations from in-memory Xarray data
    structures to a Zarr store.
    """

    @property
    def target_type(self) -> type[Array]:
        return Array

    def _sync(self, coro: Coroutine[None, None, None], /) -> None:
        """
        Run a coroutine on Zarr's event loop thread.
        """
        sync(coro)

    async def _async(self) -> None:
        """
        Construct a coroutine using the :py:mod:`~zarr.api.asynchronous` Zarr
        API.
        """
        # wait for all write operations to finish
        async with TaskGroup() as tg:
            # iterate over write operations
            for (s, t, r) in zip(self.sources, self.targets, self.regions):
                # acceess the async array API
                tg.create_task(t.async_array.setitem(r, s))


# ==============================================================================
# buffer writer
# ==============================================================================


@final
class AsyncZarrBufferWriter(AsyncBufferWriter[ZarrStore]):
    """
    Session layer for writing :py:class:`.XarrayBuffer` contents to a persistent
    Zarr store.

    Within each simulation, the Zarr store handle is reused and data is
    `appended`_ along the time dimension. After the simulation has finished,
    `consolidated metadata`_ is either created or updated.

    Example JSON configuration::

      {
        "format": 3,
        "async.concurrency": 3,
        "threading.max_workers": 3
      }

    Here,

      - ``format`` is an explicit choice of the `Zarr format`_,
      - and the other options are forwarded to the `Zarr concurrency
        configuration`_.

    .. hint::
      The Zarr :py:mod:`~zarr.api.asynchronous` API is used for writing *data
      variables* during a simulation, but the :py:mod:`~zarr.api.synchronous`
      API is still used to write *metadata attributes* at the beginning and end
      of the simulation. Once Xarray's ``async`` support has `matured`_, it
      should be fully leveraged.

    .. note::
      Zarr is currently configured to use the Rust pipeline `zarrs-python`_ for
      performance. However, the `numcodecs`_ compression used as a default in
      :py:meth:`.var_codecs` is currently not supported by `zarrs-python`_, and
      therefore, variables using such codecs will fall back to the
      `zarr-python`_ implementation. These choices need to be revisited at a
      later point, based on profiling of large-scale simulations.

    .. _appended: https://docs.xarray.dev/en/stable/user-guide/io.html#modifying-existing-zarr-stores
    .. _consolidated metadata: https://docs.xarray.dev/en/stable/user-guide/io.html#io-zarr-consolidated-metadata
    .. _Zarr format: https://zarr.readthedocs.io/en/stable/user-guide/v3_migration/
    .. _Zarr concurrency configuration: https://zarr.readthedocs.io/en/stable/user-guide/performance/#parallel-computing-and-synchronization
    .. _matured: https://github.com/pydata/xarray/issues/10622
    .. _zarrs-python: https://github.com/zarrs/zarrs-python
    .. _numcodecs: https://numcodecs.readthedocs.io/en/stable/
    .. _zarr-python: https://github.com/zarr-developers/zarr-python
    """

    @classmethod
    def validate_backend_config(cls, config: dict[str, Any], /) -> None:
        match config.get("format"):
            case None:
                raise KeyError(emitter_arg_error(
                    cls, "Missing argument",
                    "\"writer\": {\"backend_config\": {\"format\": ...}}"))
            case 2 | 3:
                pass
            case fmt:
                raise ValueError(emitter_arg_error(
                    cls, "Invalid Zarr format",
                    f"\"writer\": {{\"backend_config\": {{\"format\": {fmt}}}}}"))

    # ~~~~~~~~~~~~~~~~~ #

    @property
    def group(self) -> Group:
        return self.store.zarr_group

    def _open_group(self) -> Group:
        """
        Open Zarr API handles.

        Called by: :py:meth:`._open_store`.

        Calls: :py:func:`zarr.open_group`.
        """
        with filter_warnings(self._warnings_eval_effect):
            return zarr.open_group(
                # URI for global store holding entire workflow
                self.config["store"],
                # independent substore holding current simulation subensemble
                path=str(self.partition.independent_path),
                # enforce explicit format choice
                zarr_format=self.config["backend_config"]["format"],
                # load consolidated metadata from previous generations
                use_consolidated=True,
                # only allow appending
                mode="a",
            )

    def _check_group(self, group: Group) -> Group:
        """
        Perform basic consistency checks on the persistent storage state.

        Called by: :py:meth:`._open_store`.
        """
        if self.partition.generation == 1:
            if group.nmembers() > 0:
                raise FileExistsError(
                    f"({type(self).__name__})\n"
                    f"  Path for new independent substore already exists:\n"
                    f"    {group.store_path}")
        else:
            parent = self.partition.parent
            try:
                assert isinstance(group[parent.time_coo_name], Array)
            except KeyError:
                raise FileNotFoundError(
                    f"({type(self).__name__})\n"
                    f"  Missing path from previous generation:\n"
                    f"    {group.store_path / parent.time_coo_name}")
            if not group.attrs.get(parent.success_attr_name, False):
                raise ValueError(
                    f"({type(self).__name__})\n"
                    f"  Missing cell division event from previous generation:\n"
                    f"    {parent.success_attr_name}")
        return group

    def _cache_consolidated_metadata(self, group: Group) -> Group:
        """
        Read consolidated metadata from persistent storage, and hide it from the
        Zarr API, before it either interferes with, or is overwritten by, new
        emits. The cached value is later used by :py:meth:`.consolidate`.

        Called by: :py:meth:`._open_store`.
        """
        self.consolidated_metadata = group.metadata.consolidated_metadata
        if self.partition.generation == 1:
            assert self.consolidated_metadata is None
        else:
            assert self.consolidated_metadata is not None
            async_group = _replace_consolidated_metadata(group._async_group, None)
            group = replace(group, _async_group=async_group)
        return group

    # ~~~~~~~~~~~~~~~~~ #

    @property
    def store_type(self) -> type[ZarrStore]:
        return ZarrStore

    def _open_store(self) -> ZarrStore:
        """
        Configure the Zarr transport layer and open Xarray API handles.

        Called by: :py:meth:`.AsyncBufferWriter.open_store`.

        Calls: :py:meth:`._open_group`, :py:meth:`._check_group`,
        :py:meth:`._cache_consolidated_metadata`.
        """
        # configure Zarr
        zarr_config = self.config["backend_config"]
        zarr.config.update({
            "async": {"concurrency": zarr_config.get(
                "async.concurrency", ZARR_ASYNC_CONCURRENCY)},
            "threading": {"max_workers": zarr_config.get(
                "threading.max_workers", ZARR_MAX_WORKERS)},
            # skip overhead of fill value checks
            "array.write_empty_chunks": True,
            "codec_pipeline": {
                # use `zarrs-python`
                "path": "zarrs.ZarrsCodecPipeline",
                # limit array-level parallelism
                "batch_size": 1,
                # subordinate to `threading.max_workers`
                "chunk_concurrent_minimum": 1,
                "chunk_concurrent_maximum": None,
                # prioritise robustness across formats and platforms
                "validate_checksums": True,
                "strict": False,
                "direct_io": False,
            }
        })
        assert zarr.config.get("async.concurrency") == zarr_config.get(
            "async.concurrency", ZARR_ASYNC_CONCURRENCY)
        assert zarr.config.get("threading.max_workers") == zarr_config.get(
            "threading.max_workers", ZARR_MAX_WORKERS)

        # open Zarr store
        group = self._cache_consolidated_metadata(
            self._check_group(
                self._open_group()))
        with filter_warnings(self._warnings_eval_effect):
            return ZarrStore(
                group,
                # only allow appending along time axis
                mode="a-",
                # manage cache updates in `self.update_transport()`
                cache_members=True,
                # consolidate only after simulation finishes through Zarr API,
                # rather than after every write through Xarray API
                consolidate_on_close=False,
                # finalise Zarr API
                close_store_on_close=True,
            )

    # ~~~~~~~~~~~~~~~~~ #

    def coo_codecs(self, var: VariableSpec, /) -> VariableEncoding:
        """
        Currently, only Zarr's own default codecs are applied to a coordinate
        array.
        """
        return self._coo_codecs(self.group.metadata.zarr_format, var)

    def var_codecs(self, var: VariableSpec, /) -> VariableEncoding:
        """
        Parse the Zarr codecs for a data array, if they are specified in the
        JSON config, and otherwise, apply :py:data:`.ZARR_FILTERS` and
        :py:data:`.ZARR_COMPRESSORS`.
        """
        return self._var_codecs(self.group.metadata.zarr_format, var)

    @classmethod
    def _coo_codecs(
        cls, zarr_format: Literal[2, 3], var: VariableSpec, /
    ) -> VariableEncoding:
        # use Zarr default
        return parse_codecs(zarr_format, dtype=var.dtype)

    @classmethod
    def _var_codecs(
        cls, zarr_format: Literal[2, 3], var: VariableSpec, /
    ) -> VariableEncoding:
        z = zarr_format
        if var.codecs:
            # use variable-specific JSON config
            try:
                return parse_codecs(z, codecs=var.codecs)
            except KeyError:
                raise KeyError(emitter_arg_error(
                    cls, "Missing arguments",
                    f"...: {{\"codecs\": "
                    f"{{\"filters_v{z}\": ..., \"compressors_v{z}\": ...}}}}"))
        elif var.is_time:
            # use `vEcoli` preset for monotonic arrays
            return parse_codecs(z, category="delta", dtype=var.dtype)
        elif dtype(var.dtype).kind in ["i", "f"]:
            # use `vEcoli` preset for numeric arrays
            return parse_codecs(z, category="num", dtype=var.dtype)
        else:
            # use Zarr default
            return parse_codecs(z, dtype=var.dtype)

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def warnings_make_effect() -> list[WarningFilter]:
        return list(zarr_warnings.values())

    @staticmethod
    def warnings_eval_effect() -> list[WarningFilter]:
        return [zarr_warnings[w] for w in ["numcodecs", "zarrs"]]

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def to_zarr_path(path: NodePath) -> str:
        return "" if path == NodePath() else str(path)

    def get_zarr_path(self, path: NodePath) -> Group:
        assert isinstance(path, NodePath)
        return (self.group if path == NodePath()
                else get_group(self.group, self.to_zarr_path(path)))

    # ~~~~~~~~~~~~~~~~~ #

    def merge_attributes(self, payload: DataTree) -> None:
        """
        Combine attributes from the existing Zarr store and the Xarray buffer
        update at :py:attr:`.XarrayBuffer.modified_paths`.

        Calls: :py:attr:`.XarrayBuffer.modified_paths`.
        """
        for path in self.buffer.modified_paths:
            # empty in-memory attribute containers do not produce write operations
            if (node := payload._get_item(path)).attrs:
                node.attrs = dict(self.get_zarr_path(path).attrs) | node.attrs

    def make_effect(
        self, payload: DataTree, encoding: Mapping[str, Any], /
    ) -> AsyncZarrArrayWriter:
        """
        Calls: :py:func:`._datatree_to_zarr`.
        """
        return _datatree_to_zarr(payload, self.store, encoding)

    def update_attributes(self, path: NodePath, attrs: dict[str, Any], /) -> None:
        self.get_zarr_path(path).update_attributes(attrs)

    def update_transport(self) -> None:
        """
        After writing the first buffer for a generation, emulate reinstantiating
        the :py:class:`xarray.backends.ZarrStore` by updating its cache, and
        enforce that subsequent writes can only append along the
        generation-specific time axis.
        """
        assert self.group.metadata.consolidated_metadata is None
        assert not self.is_1st_buf_in_generation
        if self.num_writes == 1:
            with filter_warnings(self._warnings_eval_effect):
                # find direct children in the Zarr hierarchy
                self.store._members = self.store._fetch_members()
            # set appending axis
            self.store._append_dim = self.partition.time_coo_name
            assert self.store._append_dim in self.store.get_dimensions()
        assert len(self.store.members)

    def consolidate(self) -> None:
        """
        Update existing consolidated metadata in the Zarr store with the outputs
        of a newly finished simulation.

        Calls: Either :py:func:`zarr.consolidate_metadata`, or
        :py:attr:`.XarrayBuffer.modified_paths`,
        :py:attr:`.XarrayBuffer.added_paths` and
        :py:func:`.reconsolidate_metadata`.
        """
        assert self.group.metadata.consolidated_metadata is None
        with filter_warnings(self._warnings_make_effect):
            if self.partition.generation == 1:
                # create from scratch, calling `Store.list_dir()` recursively
                assert self.consolidated_metadata is None
                sync(consolidate_metadata(self.group._async_group))
            else:
                # retrieve cached consolidated metadata from previous generations
                assert self.consolidated_metadata is not None
                async_group: AsyncGroup = _replace_consolidated_metadata(
                    self.group._async_group, self.consolidated_metadata)
                # combine with metadata for new paths
                sync(reconsolidate_metadata(
                    async_group,
                    set(map(self.to_zarr_path, self.buffer.modified_paths)),
                    set(map(self.to_zarr_path, self.buffer.added_paths))))
