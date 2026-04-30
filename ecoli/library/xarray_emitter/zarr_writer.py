
"""
Transport and session layers for the Zarr storage backend.

This module defines subclasses that inherit from :py:mod:`.writer`, and modifies
the internal state in Xarray and Zarr where necessary.
"""


from __future__ import annotations

from asyncio import Semaphore, create_task, as_completed, gather
from collections.abc import AsyncGenerator, Coroutine
from collections import deque
from dataclasses import replace
from html import escape as html_escape
from typing import Any, Mapping, final, cast
import sys
import warnings

from xarray import DataTree
from xarray.core.datatree import NodePath
from xarray.backends import ZarrStore
from xarray.backends.writers import dump_to_store

import zarr
from zarr.abc.codec import Codec
from zarr.abc.numcodec import Numcodec
from zarr.core.metadata import v2, v3
from zarr.core._tree import TreeRepr
from zarr.types import AnyAsyncArray
from zarr.core.array import Array, AsyncArray
from zarr.core.sync import sync
from zarr.core.group import (
    Group, AsyncGroup, GroupMetadata, ConsolidatedMetadata, _getitem_semaphore)
from zarr.errors import ZarrUserWarning, UnstableSpecificationWarning

from .utils import  WarningFilter, filter_warnings, emitter_arg_error
from .writer import AsyncArrayWriter, AsyncBufferWriter
from .storage import VariableSpec, VariableEncoding


# ==============================================================================
# constants
# ==============================================================================


ZARR_ASYNC_CONCURRENCY: int = 4
""" Default bound on the number of Zarr's concurrent operations. """
ZARR_MAX_WORKERS: int = 4
""" Default bound on the size of Zarr's internal thread pool. """

ZARR_FILTERS: dict[int, list[dict[str, Any]]] = {
    2: [{"id": "delta", "dtype": None}],
    3: [{"name": "numcodecs.delta", "configuration": {"dtype": None}}]
}
""" Default filter codecs, as a function of the Zarr format. """
ZARR_COMPRESSORS: dict[int, list[dict[str, Any]]] = {
    2: [{"id": "blosc", "cname": "zstd", "clevel": 6,
         "shuffle": -1, "blocksize": 0}],
    3: [{"name": "blosc", "configuration": {
        "cname": "zstd", "clevel": 6,
        "typesize": None, "shuffle": None, "blocksize": 0}}]
}
""" Default compression codecs, as a function of the Zarr format. """


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
    if absolute := [p for p in encoding.keys() if p.startswith("/")]:
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
# Zarr internals
# ==============================================================================


async def consolidate_metadata(
    group: AsyncGroup,
) -> AsyncGroup:
    """
    Consolidate the metadata of all nodes in a hierarchy, including the root
    node.

    Adapted from: :py:func:`zarr.api.asynchronous.consolidate_metadata`.
    """
    # check store properties
    assert isinstance(group, AsyncGroup)
    assert group.store.supports_listing
    assert group.store.supports_consolidated_metadata
    group.store._check_writable()
    assert group.metadata.consolidated_metadata is None

    # traverse store and read all metadata
    members_metadata = {
        k: v.metadata
        async for (k, v) in
        group.members(max_depth=None, use_consolidated_for_children=False)}
    # TODO: fix in `consolidate_metadata()` (zarr==3.1.6)
    members_metadata |= {"": group.metadata}

    # combine and write consolidated metadata
    for k, v in members_metadata.items():
        if isinstance(v, GroupMetadata) and v.consolidated_metadata is None:
            members_metadata[k] = _replace_consolidated_metadata(
                v, ConsolidatedMetadata(metadata={}))
    ConsolidatedMetadata._flat_to_nested(members_metadata)
    group = _replace_consolidated_metadata(
        group, ConsolidatedMetadata(metadata=members_metadata))
    await group._save_metadata()
    return group


async def reconsolidate_metadata(
    group: AsyncGroup, modified_keys: set[str], added_keys: set[str], /
) -> AsyncGroup:
    """
    Incrementally update consolidated metadata. Rather than recursing through
    the entire store tree and recomputing the consolidated metadata afresh, load
    the existing consolidated metadata, and update it with the current metadata
    from a known list of modified and added paths.

    Adapted from: :py:func:`zarr.api.asynchronous.consolidate_metadata`.
    """
    # check paths
    assert isinstance(modified_keys, set)
    assert isinstance(added_keys, set)
    assert all(isinstance(k, str) for k in modified_keys)
    assert all(isinstance(k, str) for k in added_keys)
    assert modified_keys.isdisjoint(added_keys)

    # check store properties
    assert isinstance(group, AsyncGroup)
    assert group.store.supports_listing
    assert group.store.supports_consolidated_metadata
    group.store._check_writable()
    assert group.metadata.consolidated_metadata is not None

    # read existing consolidated metadata
    members_metadata = {
        k: n.metadata
        async for (k, n) in
        group.members(max_depth=None, use_consolidated_for_children=True)}
    for (_, v) in members_metadata.items():
        if isinstance(v, GroupMetadata):
            assert v.consolidated_metadata is not None

    # read metadata at updated paths
    group = _replace_consolidated_metadata(group, None)
    mod_members_metadata, add_members_metadata = [
        {k: n.metadata async for (k, n) in _iter_from_keys(group, keys)}
        for keys in [modified_keys, added_keys]]

    # check assumptions about metadata updates
    old_keys = set(members_metadata.keys()) | {""}
    assert set(mod_members_metadata.keys()).issubset(old_keys)
    assert set(add_members_metadata.keys()).isdisjoint(old_keys)

    # combine and write consolidated metadata
    for metadata in [mod_members_metadata, add_members_metadata]:
        for (k, v) in metadata.items():
            if isinstance(v, GroupMetadata):
                assert v.consolidated_metadata is None
                metadata[k] = _replace_consolidated_metadata(
                    v, ConsolidatedMetadata(metadata={}))
    members_metadata |= mod_members_metadata | add_members_metadata
    del old_keys, mod_members_metadata, add_members_metadata
    # TODO: fix in `ConsolidatedMetadata._flat_to_nested()` (zarr==3.1.6)
    members_metadata = dict(sorted(members_metadata.items(),
                                   key=lambda kv: bfs_key(kv[0])))
    ConsolidatedMetadata._flat_to_nested(members_metadata)
    group = _replace_consolidated_metadata(
        group, ConsolidatedMetadata(metadata=members_metadata))
    await group._save_metadata()
    return group


def _replace_consolidated_metadata[NodeT: (AsyncGroup, GroupMetadata)](
    node: NodeT, consolidated: ConsolidatedMetadata | None
) -> NodeT:
    match node:
        case AsyncGroup():
            _metadata = _replace_consolidated_metadata(node.metadata, consolidated)
            return replace(node, metadata=_metadata)
        case GroupMetadata():
            assert isinstance(consolidated, ConsolidatedMetadata | None)
            return replace(node, consolidated_metadata=consolidated)
        case _:
            raise ValueError(node)


# ------------------------------------------------------------------------------


async def _iter_from_keys(
    node: AsyncGroup, keys: set[str], /
) -> AsyncGenerator[tuple[str, AnyAsyncArray | AsyncGroup], None]:
    """
    Iterate over a known list of arrays and groups contained within a group,
    returning relative paths and node objects.

    Called by: :py:func:`.reconsolidate_metadata`.

    Adapted from: :py:func:`!zarr.core.group._iter_members`.
    """
    semaphore = Semaphore(zarr.config.get("async.concurrency"))
    node_tasks = tuple(
        create_task(_getitem_semaphore(node, key, semaphore), name=key)
        for key in keys)
    for fetched_node_coro in as_completed(node_tasks):
        try:
            fetched_node = await fetched_node_coro
        except KeyError as e:
            warnings.warn(
                f"Object at {e.args[0]} is not recognized as a component of a Zarr hierarchy.",
                ZarrUserWarning, stacklevel=1)
            continue
        match fetched_node:
            case AsyncArray() | AsyncGroup():
                # remove prefix path, accommodating normalised root path
                rel_path = fetched_node.name.removeprefix(node.name).removeprefix("/")
                yield (rel_path, fetched_node)
            case _:
                raise ValueError(f"Unexpected type: {type(fetched_node)}")


def bfs_key(path: str) -> tuple:
    """
    Corrected sorting key for the pre-processing step in
    :py:meth:`!zarr.core.group.ConsolidatedMetadata._flat_to_nested`.

    Called by: :py:func:`.reconsolidate_metadata`.
    """
    segments = path.split("/")
    return (len(segments), *segments)


# ------------------------------------------------------------------------------


def group_tree(
    group: Group,
    level: int | None = None,
    *,
    max_nodes: int = 500,
    plain: bool = False,
) -> TreeRepr:
    """
    Adapted from: :py:meth:`!zarr.Group.tree`.

    Calls: :py:func:`.group_tree_async`.
    """
    return sync(group_tree_async(
        group._async_group,
        max_depth=level, max_nodes=max_nodes, plain=plain))


async def group_tree_async(
    group: AsyncGroup,
    max_depth: int | None = None,
    *,
    max_nodes: int = 500,
    plain: bool = False,
) -> TreeRepr:
    """
    Fix edge case with infinite recursion in
    :py:func:`!zarr.core._tree.group_tree_async`.

    Called by: :py:func:`.group_tree`.
    """
    members: list[tuple[str, Any]] = []
    truncated = False
    async for item in group.members(max_depth=max_depth):
        if len(members) == max_nodes:
            truncated = True
            break
        members.append(item)
    members.sort(key=lambda key_node: key_node[0])

    # Set up styling tokens: ANSI bold for terminals, HTML <b> for Jupyter,
    # or empty strings when plain=True (useful for LLMs, logging, files).
    if plain:
        ansi_open = ansi_close = html_open = html_close = ""
    else:
        # Avoid emitting ANSI escape codes when output is piped or in CI.
        use_ansi = sys.stdout.isatty()
        ansi_open = "\x1b[1m" if use_ansi else ""
        ansi_close = "\x1b[0m" if use_ansi else ""
        html_open = "<b>"
        html_close = "</b>"

    # Group members by parent key so we can render the tree level by level.
    nodes: dict[str, list[tuple[str, Any]]] = {}
    for key, node in members:
        # TODO: fix in `group_tree_async()` (zarr==3.1.6)
        if key == "":
            # avoid self-loop at root node
            continue
        elif key.count("/") == 0:
            parent_key = ""
        else:
            parent_key = key.rsplit("/", 1)[0]
        nodes.setdefault(parent_key, []).append((key, node))

    # Render the tree iteratively (not recursively) to avoid hitting
    # Python's recursion limit on deeply nested hierarchies.
    # Each stack frame is (prefix_string, remaining_children_at_this_level).
    text_lines = [f"{ansi_open}{group.name}{ansi_close}"]
    html_lines = [f"{html_open}{html_escape(group.name)}{html_close}"]
    stack = [("", deque(nodes.get("", [])))]
    while stack:
        prefix, remaining = stack[-1]
        if not remaining:
            stack.pop()
            continue
        key, node = remaining.popleft()
        name = key.rsplit("/")[-1]
        escaped_name = html_escape(name)
        # if we popped the last item then remaining will
        # now be empty - that's how we got past the if not remaining
        # above, but this can still be true.
        is_last = not remaining
        connector = "└── " if is_last else "├── "
        if isinstance(node, AsyncGroup):
            text_lines.append(f"{prefix}{connector}{ansi_open}{name}{ansi_close}")
            html_lines.append(f"{prefix}{connector}{html_open}{escaped_name}{html_close}")
        else:
            text_lines.append(
                f"{prefix}{connector}{ansi_open}{name}{ansi_close} {node.shape} {node.dtype}"
            )
            html_lines.append(
                f"{prefix}{connector}{html_open}{escaped_name}{html_close}"
                f" {html_escape(str(node.shape))} {html_escape(str(node.dtype))}"
            )
        # Descend into children with an accumulated prefix:
        # Example showing how prefix accumulates:
        #   /
        #   ├── a              prefix = ""
        #   │   ├── b          prefix = "" + "│   "
        #   │   │   └── x      prefix = "" + "│   " + "│   "
        #   │   └── c          prefix = "" + "│   "
        #   └── d              prefix = ""
        #       └── e          prefix = "" + "    "
        if children := nodes.get(key, []):
            if is_last:
                child_prefix = prefix + "    "
            else:
                child_prefix = prefix + "│   "
            stack.append((child_prefix, deque(children)))
    text = "\n".join(text_lines) + "\n"
    html = "\n".join(html_lines) + "\n"
    note = (
        f"Truncated at max_nodes={max_nodes}, some nodes and their children may be missing\n"
        if truncated
        else ""
    )
    return TreeRepr(text, html, truncated=note)


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
        await gather(*(
            # acceess the async array API
            t.async_array.setitem(r, s)
            # iterate over write operations
            for (s, t, r) in zip(self.sources, self.targets, self.regions)))


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
    def validate_config(cls, config: dict[str, Any], /) -> None:
        super().validate_config(config)
        zarr_config = config["backend_config"]
        match zarr_config.get("format"):
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
        zarr_config = self.config["backend_config"]
        zarr.config.update({
            "async.concurrency": zarr_config.get(
                "async.concurrency", ZARR_ASYNC_CONCURRENCY),
            "threading.max_workers": zarr_config.get(
                "threading.max_workers", ZARR_MAX_WORKERS),
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
        return ZarrStore(
            self._cache_consolidated_metadata(
                self._check_group(self._open_group())),
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
        Currently, no Zarr codecs are applied to coordinate arrays.
        """
        return {}

    def var_codecs(self, var: VariableSpec, /) -> VariableEncoding:
        """
        Parse the Zarr codecs for a simulation variable, if they are specified
        in the JSON config, and otherwise, apply the default codecs.
        """
        z: int = self.group.metadata.zarr_format
        if var.codecs:
            # fetch variable-specific JSON config
            _filters = var.codecs.get(f"filters_v{z}", [])
            _compressors = var.codecs.get(f"compressors_v{z}", [])
            if not (_filters or _compressors):
                raise ValueError(emitter_arg_error(
                    self, "Missing arguments",
                    f"...: {{\"codecs\": "
                    f"{{\"filters_v{z}\": ..., \"compressors_v{z}\": ...}}}}"))
        else:
            # fetch default config and supply variable-specific information
            _filters = ZARR_FILTERS[z]
            _compressors = ZARR_COMPRESSORS[z]
            for f in _filters:
                if z == 2:
                    f["dtype"] = var.dtype
                else:
                    f["configuration"]["dtype"] = var.dtype
        # parse codec config
        filters: tuple[Codec | Numcodec, ...] | None
        compressors: tuple[Codec | Numcodec | None, ...]
        with filter_warnings(self._warnings_make_effect):
            if z == 2:
                filters = v2.parse_filters(_filters)
                compressors = tuple(map(v2.parse_compressor, _compressors))
            else:
                filters = v3.parse_codecs(_filters)
                compressors = v3.parse_codecs(_compressors)
        return {"filters": filters, "compressors": compressors}

    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    def warnings_make_effect(cls) -> list[WarningFilter]:
        return [
            WarningFilter(
                module="zarr.api.asynchronous",
                category=ZarrUserWarning,
                message="Consolidated metadata.*Zarr format 3",
                action="ignore"),
            WarningFilter(
                module="zarr.core.dtype.npy.string",
                category=UnstableSpecificationWarning,
                message=".*data type.*Zarr V3",
                action="ignore"),
            WarningFilter(
                module="zarr.codecs.numcodecs",
                category=ZarrUserWarning,
                message=".*Numcodecs codecs.*Zarr version 3",
                action="ignore"),
            WarningFilter(
                module="zarrs.pipeline",
                category=UserWarning,
                message="Array is unsupported by ZarrsCodecPipeline",
                action="ignore")]

    @classmethod
    def warnings_eval_effect(cls) -> list[WarningFilter]:
        return [
            WarningFilter(
                module="zarrs.pipeline",
                category=UserWarning,
                message="Array is unsupported by ZarrsCodecPipeline",
                action="ignore")]

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def to_zarr_path(path: NodePath) -> str:
        return "" if path == NodePath() else str(path)

    def get_zarr_path(self, path: NodePath) -> Group:
        assert isinstance(path, NodePath)
        return (self.group if path == NodePath()
                else cast(Group, self.group[self.to_zarr_path(path)]))

    # ~~~~~~~~~~~~~~~~~ #

    def merge_attributes(self, payload: DataTree) -> None:
        """
        Combine attributes from the existing Zarr store and the Xarray buffer
        update at :py:attr:`.XarrayBuffer.modified_paths`.
        """
        for path in self.buffer.modified_paths:
            # empty in-memory attribute containers do not produce write operations
            if (xr_attrs := payload._get_item(path).attrs):
                zr_attrs = dict(self.get_zarr_path(path).attrs)
                payload._get_item(path).attrs = zr_attrs | xr_attrs

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
        assert self.num_writes > 0
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

        Calls: :py:func:`zarr.consolidate_metadata` or
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
