
"""
Utilities for controlling low-level Zarr internals.
"""

import warnings
from asyncio import Semaphore, as_completed, create_task
from collections.abc import AsyncGenerator
from dataclasses import replace
from lzma import FILTER_LZMA2, FORMAT_RAW
from typing import Any, Literal, cast

import zarr
from numpy.typing import NDArray
from zarr.abc.codec import Codec
from zarr.abc.numcodec import Numcodec
from zarr.core.array import (
    Array,
    AsyncArray,
    _parse_chunk_encoding_v2,
    default_compressors_v3,
)
from zarr.core.dtype import parse_dtype
from zarr.core.group import (
    AsyncGroup,
    ConsolidatedMetadata,
    Group,
    GroupMetadata,
    _getitem_semaphore,
)
from zarr.core.indexing import BlockIndex
from zarr.core.metadata import v2, v3
from zarr.errors import UnstableSpecificationWarning, ZarrUserWarning
from zarr.types import AnyAsyncArray

from .storage import VariableEncoding
from .utils import WarningFilter, filter_warnings

# ==============================================================================
# Zarr warnings
# ==============================================================================


zarr_warnings: dict[str, WarningFilter] = {
    "consolidated_metadata": WarningFilter(
        module="zarr.api.asynchronous",
        category=ZarrUserWarning,
        message="Consolidated metadata.*Zarr format 3",
        action="ignore"),
    "string": WarningFilter(
        module="zarr.core.dtype.npy.string",
        category=UnstableSpecificationWarning,
        message=".*data type.*Zarr V3",
        action="ignore"),
    "numcodecs": WarningFilter(
        module="zarr.codecs.numcodecs",
        category=ZarrUserWarning,
        message=".*Numcodecs codecs.*Zarr version 3 specification",
        action="ignore"),
    "zarrs": WarningFilter(
        module="zarrs.pipeline",
        category=UserWarning,
        message="Array is unsupported by ZarrsCodecPipeline",
        action="ignore")
}


# ==============================================================================
# Zarr codecs
# ==============================================================================


ZARR_FILTERS: dict[str, dict[int, list[dict[str, Any]]]] = {
    "delta": {
        2: [{"id": "delta", "dtype": None}],
        3: [{"name": "numcodecs.delta", "configuration": {"dtype": None}}]
    },
    "num": {
        2: [],
        3: []
    },
}
"""
Default filter codecs for :py:meth:`.AsyncZarrBufferWriter.var_codecs`, as a
function of the array category and the Zarr format.
"""
ZARR_COMPRESSORS: dict[str, dict[int, list[dict[str, Any]]]] = {
    "delta": {
        2: [{"id": "blosc", "cname": "zstd", "clevel": 6,
            "shuffle": -1, "blocksize": 0}],
        3: [{"name": "blosc", "configuration": {
            "cname": "zstd", "clevel": 6,
            "typesize": None, "shuffle": None, "blocksize": 0}}]
    },
    "num": {
        2: [{"id": "lzma", "check": -1, "preset": None,
             "format": FORMAT_RAW,
             "filters": [{"id": FILTER_LZMA2, "preset": 6}]}],
        3: [{"name": "numcodecs.lzma",
             "configuration": {"format": FORMAT_RAW,
                               "filters": [{"id": FILTER_LZMA2, "preset": 6}]}}]
    },
}
"""
Default compression codecs for :py:meth:`.AsyncZarrBufferWriter.var_codecs`, as
a function of the array category and the Zarr format.
"""


# ------------------------------------------------------------------------------


def parse_codecs(
    zarr_format: Literal[2, 3], /, *,
    codecs: dict[str, Any] | None = None,
    category: str | None = None,
    dtype: str | None = None
) -> VariableEncoding:
    """
    Translate a Vivarium JSON configuration into a codec specification that is
    interpretable by the Zarr API, leveraging Zarr internal functions for
    parsing.

    Used by: :py:meth:`.AsyncZarrBufferWriter.coo_codecs`,
    :py:meth:`.AsyncZarrBufferWriter.var_codecs`.
    """
    filters: tuple[Codec | Numcodec, ...] | None
    compressors: tuple[Codec | Numcodec | None, ...]

    # dispatch on spec type
    z = zarr_format
    if codecs or (category is not None):
        # fetch non-default config
        if codecs:
            # fetch custom JSON config
            assert category is None
            assert dtype is None
            _filters = codecs.get(f"filters_v{z}", [])
            _compressors = codecs.get(f"compressors_v{z}", [])
            if not (_filters or _compressors):
                raise KeyError(
                    f"Missing arguments:\n  "
                    f"{{\"filters_v{z}\": ..., \"compressors_v{z}\": ...}}")
        elif category is not None:
            # fetch library preset, supply data type information
            _filters = ZARR_FILTERS[category][z]
            _compressors = ZARR_COMPRESSORS[category][z]
            assert dtype is not None
            for f in _filters:
                if z == 2:
                    f["dtype"] = dtype
                else:
                    f["configuration"]["dtype"] = dtype
        # parse non-default config
        with filter_warnings(list(zarr_warnings.values())):
            if z == 2:
                filters = v2.parse_filters(_filters)
                compressors = tuple(map(v2.parse_compressor, _compressors))
            else:
                filters = v3.parse_codecs(_filters)
                compressors = v3.parse_codecs(_compressors)
    else:
        # fetch default config, supply data type information
        assert dtype is not None
        _dtype = parse_dtype(dtype, zarr_format=z)
        if z == 2:
            filters, compressor = _parse_chunk_encoding_v2(
                filters="auto", compressor="auto", dtype=_dtype
            )
            compressors = (compressor,)
        else:
            filters = ()
            compressors = default_compressors_v3(_dtype)

    return {"filters": filters, "compressors": compressors}


# ==============================================================================
# Zarr store access
# ==============================================================================


def get_group(group: Group, path: str) -> Group:
    """
    Access a Zarr store path known to be a :py:class:`~zarr.Group`.
    """
    return cast(Group, group[path])


def get_array(group: Group, path: str) -> Array:
    """
    Access a Zarr store path known to be an :py:class:`~zarr.Array`.
    """
    return cast(Array,  group[path])


def get_ndarray(group: Group, path: str) -> NDArray:
    """
    Access a Zarr store path known to be an :py:class:`~zarr.Array`, and
    retrieve the uncompressed array data.
    """
    return cast(NDArray,  get_array(group, path)[:])


def get_rectilinear_ndarray(group: Group, path: str) -> list:
    """
    Given a Zarr array that was stored using a `rectilinear chunk grid`_, return
    a list over its block views.

    .. _rectilinear chunk grid: https://zarr.readthedocs.io/en/stable/user-guide/examples/rectilinear_chunks/
    """
    arr = get_array(group, path)
    blk_view: BlockIndex = arr.blocks
    blk_ix = [range(len(dim)) for dim in arr.write_chunk_sizes]
    def get_blocks(outer: tuple[int,...], inner: list[range]):
        return ([get_blocks(outer + (i,), inner[1:]) for i in inner[0]]
                if inner else
                blk_view[outer])
    return get_blocks((), blk_ix)


async def get_async_group(group: AsyncGroup, path: str) -> AsyncGroup:
    """
    Access a Zarr store path known to be an :py:class:`~zarr.AsyncGroup`.
    """
    return cast(AsyncGroup, await group.getitem(path))


async def get_async_array(group: AsyncGroup, path: str) -> AsyncArray:
    """
    Access a Zarr store path known to be an :py:class:`~zarr.AsyncArray`.
    """
    return cast(AsyncArray, await group.getitem(path))


# ==============================================================================
# Zarr internals
# ==============================================================================


async def consolidate_metadata(
    group: AsyncGroup,
) -> AsyncGroup:
    """
    Consolidate the metadata of all nodes in a hierarchy (except from the root
    node).

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
    from a known list of modified and added paths (except from the root node).

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
    for v in members_metadata.values():
        if isinstance(v, GroupMetadata):
            assert v.consolidated_metadata is not None

    # read metadata at updated paths
    group = _replace_consolidated_metadata(group, None)
    mod_members_metadata, add_members_metadata = [
        {k: n.metadata
         async for (k, n) in _iter_from_keys(group, keys)
         # NOTE: no updates for root node:
         # - *children* and *attributes* for root node were already persisted
         # - no *metadata* for root node allowed inside consolidated metadata
         if k}
        for keys in [modified_keys, added_keys]]

    # check assumptions about metadata updates
    old_keys = set(members_metadata.keys())
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

    # This is a workaround for a pre-sorting key with
    # too strict assumptions inside `ConsolidatedMetadata._flat_to_nested()`.
    # TODO: Fix upstream (as of zarr==3.3)
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
