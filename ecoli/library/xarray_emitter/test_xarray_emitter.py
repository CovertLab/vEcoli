
"""
Unit and integration tests for :py:mod:`.xarray_emitter` and its submodules.
"""


from contextlib import ContextDecorator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from random import randint
from typing import Any, Literal, Self, Callable, final, cast

import numpy as np
from pytest import MonkeyPatch, mark, param, raises
from xarray import DataArray, DataTree, open_datatree
from xarray.core.datatree import NodePath
from zarr import Array, Group, open_consolidated

from ecoli.library.test_utils import PatchConfig, filter_warnings
from ecoli.library.xarray_emitter.emit_path import EmitPath, EmitPathType
from ecoli.library.xarray_emitter.view import LeafView, ForestView
from ecoli.library.xarray_emitter.storage import (
    XarrayStoragePartition, VariableSpec, VariableEncoding)
from ecoli.library.xarray_emitter.zarr_writer import (
    AsyncZarrBufferWriter, group_tree)
from ecoli.library.xarray_emitter.emitter import XarrayEmitter
from ecoli.library.xarray_emitter.utils import WarningFilter
from ecoli.processes.metabolism import TIME_UNITS

from runscripts.test_workflow import MockEcoliSimWorkflow


# mypy: disable-error-code="attr-defined"


# ==============================================================================
# unit tests
# ==============================================================================


class TestEmitPath:

    @classmethod
    def test_path_type(cls):
        assert EmitPath(()).type == EmitPathType(0)
        assert EmitPath(("agents", "13")).type.is_agent
        assert EmitPath(("listeners", "foo", "bar")).type.is_listener
        assert EmitPath(("log_update", "elan_vital")).type.is_update
        assert EmitPath(("log_update", "", "listeners")).type.is_update_listener
        with raises(AssertionError):
            EmitPath(("agents", "0", "listeners"))
        with raises(AssertionError):
            EmitPath(("agents", "log_update"))
        with raises(AssertionError):
            EmitPath(("log_update", "baz", "log_update"))

    @classmethod
    def test_metadata_path(cls):
        non_listener = ("foo", "bar")
        assert EmitPath(non_listener).metadata_path == non_listener
        listener = ("listeners", "foo", "bar", "7")
        assert EmitPath(listener).metadata_path == listener
        assert EmitPath(("log_update", "baz") + listener).metadata_path == listener
        with raises(AssertionError):
            EmitPath(("agents", "13")).metadata_path


# ==============================================================================
# integration tests
# ==============================================================================


ecolisim_warnings = [
    WarningFilter(
        module="scipy.integrate._ivp.bdf",
        category=RuntimeWarning,
        message="invalid value encountered",
        action="ignore"),
]


# ------------------------------------------------------------------------------


@final
@dataclass
class XarrayEmitterConfig(PatchConfig):
    """
    :py:class:`.PatchConfig` for the :py:class:`.XarrayEmitter`.
    """

    outdir: Path
    zarr_format: Literal[2, 3]
    threaded: bool
    debug: bool
    interval: int
    buffers_per_chunk: int

    def __post_init__(self) -> None:
        assert isinstance(self.outdir, Path)
        assert self.outdir.is_absolute()

    def to_dict(self) -> dict[str, Any]:
        return {
            "emitter": "xarray",
            "emitter_arg": {
                "out_dir": str(self.outdir),
                "writer": {
                    "threaded": self.threaded,
                    "buffers_per_chunk": self.buffers_per_chunk,
                    "backend": "zarr",
                    "backend_config": {
                        "format": self.zarr_format
                    },
                },
                "transducer": {
                    "predicate": [
                        [
                            {"subsample": {"interval": self.interval}},
                            {"fixed": {"steps": [0]}}
                        ]
                    ]
                },
                "debug": self.debug
            }
        }


# ------------------------------------------------------------------------------


@dataclass(slots=True)
class StoreResult(ContextDecorator):
    """
    Context manager for opening, using both the Xarray API and the Zarr API, an
    output store that was produced by :py:class:`.XarrayEmitter` via the
    :py:class:`.AsyncZarrBufferWriter`.
    """

    store: Path
    partition: XarrayStoragePartition
    zarr_format: Literal[2, 3]
    zarr: Group = field(init=False)
    xarray: DataTree = field(init=False)

    def __post_init__(self) -> None:
        isinstance(self.store, Path)
        isinstance(self.partition, XarrayStoragePartition)

    def __enter__(self) -> Self:
        ind_store = self.store / self.partition.independent_path
        self.zarr = open_consolidated(
            ind_store, mode="r", zarr_format=self.zarr_format)
        self.xarray = open_datatree(
            ind_store, mode="r", chunks=None, engine="zarr",
            zarr_format=self.zarr_format, consolidated=True)
        assert ind_store == self.zarr.store.root
        assert str(ind_store) == self.xarray.encoding["source"]
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.close()

    def close(self) -> None:
        self.zarr.store.close()
        self.xarray.close()


# ------------------------------------------------------------------------------


class TestEcoliSim:
    """
    Complete integration test for :py:class:`.XarrayEmitter`, running an
    abridged multi-generation simulation workflow using
    :py:class:`.MockEcoliSimWorkflow`, and validating the full round-tripped
    data structure from the output Zarr store.
    """

    @classmethod
    @filter_warnings(ecolisim_warnings)
    @filter_warnings(AsyncZarrBufferWriter.warnings_all())
    @mark.parametrize(
        "num_generations, last_success, interval, "
        "buffers_per_chunk, zarr_format, threaded, debug",
        [
            param(*args, **kwargs, id=(
                "gen:{}_succ:{}_intvl:{}_buf:{}_zarr:{}_thrd:{}_dbg:{}"
            ).format(*args))
            for (args, kwargs) in [
                ((1, False, 1, 1, 2, False, True ), {}),
                ((2, True,  3, 2, 2, True,  False), {}),
                ((2, True,  2, 1, 3, True,  True), {"marks": mark.basic_workflow}),
                ((3, False, 1, 3, 3, True,  False), {})
            ]
        ]
    )
    def test_workflow(
        cls, monkeypatch: MonkeyPatch, tmp_path: Path,
        num_generations: int, last_success: bool, interval: int,
        buffers_per_chunk: int, zarr_format: Literal[2, 3],
        threaded: bool, debug: bool
    ):
        """
        Driver for the integration test.
        """
        # set repository paths
        sim_data_path = Path.cwd() / "out" / "kb" / "simData.cPickle"
        config_name = "test_xarray_emitter"

        # set unique test directories
        workdir = tmp_path
        assert workdir.is_absolute()
        assert workdir.exists()
        assert not(list(workdir.iterdir()))
        daughter_outdir = workdir / config_name / "daughter_states"
        store = workdir / config_name / "store"

        # configure simulation workflow
        wf = MockEcoliSimWorkflow(
            monkeypatch=monkeypatch, workdir=workdir,
            config_name=config_name, sim_data_path=sim_data_path,
            daughter_outdir=daughter_outdir,
            lineage_seed=randint(0, 2**10 - 1),
            emitter_config=XarrayEmitterConfig(
                workdir, zarr_format, threaded, debug,
                interval, buffers_per_chunk))

        # step through workflow
        hline = "=" * 79
        for g in range(1, num_generations + 1):
            print(f"\n{hline}\nGeneration: {g}\n{hline}")
            # execute simulation
            success = (g < num_generations) or last_success
            (partition, config) = wf.sim_gen(success)
            assert isinstance(partition, XarrayStoragePartition)
            # read emitted data
            view = ForestView.from_dict(config["emitter_arg"]["view"])
            with StoreResult(store, partition, zarr_format) as result:
                # validate emitted data
                cls.check_tree_shape(result, view)
                cls.check_log(result, success)
                cls.check_time(result, interval)
                cls.check_chunks(result, view, interval, buffers_per_chunk, config)
                cls.check_codecs(result, view, zarr_format, config)

    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    def check_tree_shape(cls, res: StoreResult, view: ForestView) -> None:
        """
        Check the syntactic integrity of the round-tripped store data structure,
        for *all generations emitted so far*.

        Calls: :py:meth:`.check_root_node_shape` and
        :py:meth:`.check_child_node_shape`.
        """
        # let Zarr traverse the store
        z = res.zarr
        print()
        print(group_tree(z))

        # let Xarray traverse the store
        x = res.xarray
        print("\n", "-" * 79, "\n")
        print(x)
        print()

        # inspect current and previous generations
        p: XarrayStoragePartition | None = res.partition
        while p is not None:
            # traverse the store tree
            cls.check_root_node_shape(p, x)
            t_size = len(x._data_variables[p.time_var_name])
            for tree in view.forest:
                for leaf in tree.leaves:
                    cls.check_child_node_shape(p, t_size, leaf, x)
            p = p.parent if p.generation > 1 else None

    @staticmethod
    def check_root_node_shape(p: XarrayStoragePartition, n: DataTree) -> None:
        """
        Look for expected fields in the root node of the output store.

        Called by: :py:meth:`.check_tree_shape`.
        """
        assert isinstance(n, DataTree)

        # simulation metadata
        assert set(n.attrs[p.sim_id].keys()) == set(XarrayEmitter.metadata_keys)

        # unit annotation
        assert p.time_var_name in n.attrs

        # coordinate & data variables
        ti = n._node_coord_variables[p.time_coo_name]
        t = n._data_variables[p.time_var_name]
        assert ti.shape == t.shape

    @staticmethod
    def check_child_node_shape(
        p: XarrayStoragePartition, t_size: int, leaf: LeafView, n: DataTree
    ) -> None:
        """
        Look for expected fields in a child node of the output store.

        Called by: :py:meth:`.check_tree_shape`.
        """
        m = n[str(leaf.path)]
        assert isinstance(m, DataTree)
        v = leaf.var_name

        # unit annotation
        if leaf.unit:
            assert isinstance(u := m.attrs[v], str) and u

        # coordinate & data variables
        assert (d := m.data_vars[p.dynamic_suffix]).shape[0] == t_size
        if c := m._node_coord_variables:
            assert set(c.keys()) == {o := VariableSpec.var_coo_name(v)}
            assert d.shape[1:] == c[o].shape
        else:
            assert d.shape[1:] == ()

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def check_log(res: StoreResult, success: bool) -> None:
        """
        Check the integrity of the write log and the validity of the success
        flag for *the latest emitted generation*.
        """
        assert {"sim_step", "sim_time"} == set(
            res.xarray.attrs.get(res.partition.log_attr_name, {}).keys())
        assert success == res.xarray.attrs.get(
            res.partition.success_attr_name, False)

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def check_time(res: StoreResult, interval: int) -> None:
        """
        Check the validity of the time variable for *the latest emitted
        generation*.
        """
        # attribute names
        sim_id = res.partition.sim_id
        t_coo = res.partition.time_coo_name
        t_var = res.partition.time_var_name
        t_log = res.partition.log_attr_name

        # check time unit
        assert res.xarray.attrs[t_var] == TIME_UNITS.strUnit()

        # check time coordinate
        t_ix = res.xarray[t_coo]
        t_n = len(t_ix)
        assert np.array_equal(t_ix, np.arange(t_n))

        # time scale, as read from data variable
        t = cast(DataArray, res.xarray[t_var])
        t_01 = t[{t_coo: slice(0, 2)}].values
        t_end = t[{t_coo: -1}].values.item()

        # time scale, as deduced from metadata
        g = res.partition.generation
        T_0, dt, dT = map(res.xarray.attrs[sim_id].get,
                          ["initial_global_time", "time_step", "max_duration"])
        T_01 = np.cumsum(np.array([1, interval]) * [T_0, dt])
        T_end = res.xarray.attrs[t_log]["sim_time"]
        idt = interval * dt

        # check time grid
        assert T_01[0] == (g - 1) * dT
        assert T_end == T_01[0] + (dT // idt) * idt
        assert np.array_equal(T_01, t_01)
        assert T_end == t_end
        assert t.values.min() == t_01[0]
        assert t.values.max() == t_end
        assert np.array_equiv(np.diff(t), idt)

    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    def check_chunks(
        cls, res: StoreResult, view: ForestView,
        interval: int, buffers_per_chunk: int, config: dict[str, Any]
    ) -> None:
        """
        Check the integrity of all Zarr chunks for *the latest emitted
        generation*.

        Calls: :py:meth:`.check_root_node_chunks` and
        :py:meth:`.check_child_node_chunks`.
        """
        # config variables
        p = res.partition
        T: float = config["max_duration"]
        dt: float = config["time_step"]
        cnf = config["emitter_arg"]["transducer"]["predicate"]
        pred = {"subsample": {"interval": interval}}
        assert any(pred in clause for clause in cnf)
        b: int = config["emitter_arg"]["transducer"]["buffer"]["size"]
        c: int = config["emitter_arg"]["writer"]["buffers_per_chunk"]
        assert buffers_per_chunk == c

        # store accessor
        def z(path: str | NodePath) -> Array:
            return cast(Array, res.zarr[str(path)])

        # traverse the store tree
        t_size = 1 + int(T / dt) // interval
        c_size = b * c
        cls.check_root_node_chunks(p, t_size, c_size, z)
        for tree in view.forest:
            for leaf in tree.leaves:
                cls.check_child_node_chunks(p, t_size, c_size, leaf, z)

    @staticmethod
    def check_root_node_chunks(
        p: XarrayStoragePartition, t_size: int, c_size: int,
        g: Callable[[str | NodePath], Array]
    ) -> None:
        """
        Check the integrity of the Zarr chunks in the root node of the output
        store.

        Called by: :py:meth:`.check_chunks`.
        """
        # data variable
        t = g(p.time_var_name)
        assert t.shape == (t_size,)
        assert t.chunks == (c_size,)

        # coordinate variable
        ti = g(p.time_coo_name)
        assert ti.shape == (t_size,)
        assert ti.chunks == (c_size,)

    @staticmethod
    def check_child_node_chunks(
        p: XarrayStoragePartition, t_size: int, c_size: int,
        leaf: LeafView, g: Callable[[str | NodePath], Array]
    ) -> None:
        """
        Check the integrity of the Zarr chunks in a child node of the output
        store.

        Called by: :py:meth:`.check_chunks`.
        """
        # data variable
        d = g(leaf.path / p.dynamic_suffix)
        assert d.shape[0] == t_size
        assert d.chunks[0] == c_size

        # coordinate variable
        try:
            c = g(leaf.path / VariableSpec.var_coo_name(leaf.var_name))
            assert d.shape[1:] == c.shape
            assert d.chunks[1:] == c.chunks
            assert c.shape == c.chunks
        except KeyError:
            assert d.shape[1:] == ()
            assert d.chunks[1:] == ()

    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    def check_codecs(
        cls, res: StoreResult, view: ForestView,
        zarr_format: Literal[2, 3], config: dict[str, Any]
    ) -> None:
        """
        Check the integrity of all Zarr codecs for *the latest emitted
        generation*.

        Calls: :py:meth:`.check_root_node_codecs` and
        :py:meth:`.check_child_node_codecs`.
        """
        # config variables
        p = res.partition
        b: int = config["emitter_arg"]["transducer"]["buffer"]["size"]

        # transport backend
        writer = AsyncZarrBufferWriter(config["emitter_arg"]["writer"])

        # store accessor
        def z(path: str | NodePath) -> Array:
            return cast(Array, res.zarr[str(path)])

        # traverse the store tree
        cls.check_root_node_codecs(p, writer, zarr_format, b, z)
        for tree in view.forest:
            for leaf in tree.leaves:
                cls.check_child_node_codecs(p, writer, zarr_format, leaf, z)

    @classmethod
    def check_root_node_codecs(
        cls, p: XarrayStoragePartition, writer: AsyncZarrBufferWriter,
        zarr_format: Literal[2, 3], b: int,
        g: Callable[[str | NodePath], Array]
    ) -> None:
        """
        Check the integrity of the Zarr codecs in the root node of the output
        store.

        Called by: :py:meth:`.check_codecs`.

        Calls: :py:meth:`.compare_zarr_codecs`.
        """
        time_spec = VariableSpec.make_time(p, b)
        # coordinate variable
        cls.compare_zarr_codecs(zarr_format,
                                writer._coo_codecs(zarr_format, time_spec),
                                g(p.time_coo_name))
        # data variable
        cls.compare_zarr_codecs(zarr_format,
                                writer._var_codecs(zarr_format, time_spec),
                                g(p.time_var_name))

    @classmethod
    def check_child_node_codecs(
        cls, p: XarrayStoragePartition, writer: AsyncZarrBufferWriter,
        zarr_format: Literal[2, 3], leaf: LeafView,
        g: Callable[[str | NodePath], Array]
    ) -> None:
        """
        Check the integrity of the Zarr codecs in a child node of the output
        store.

        Called by: :py:meth:`.check_codecs`.

        Calls: :py:meth:`.compare_zarr_codecs`.
        """
        v = leaf.var_name
        var_spec = VariableSpec(
            # skip coord array construction for the purposes of this test
            partition=p, coord=None,
            var_name=v, dtype=leaf.dtype, unit=leaf.unit,
            codecs=leaf.codecs)
        # data variable
        cls.compare_zarr_codecs(zarr_format,
                                writer._var_codecs(zarr_format, var_spec),
                                g(leaf.path / p.dynamic_suffix))
        # coordinate variable
        try:
            cls.compare_zarr_codecs(zarr_format,
                                    writer._coo_codecs(zarr_format, var_spec),
                                    g(leaf.path / VariableSpec.var_coo_name(v)))
        except KeyError:
            pass

    @classmethod
    def compare_zarr_codecs(
        cls, zarr_format: Literal[2, 3], spec: VariableEncoding, array: Array
    ) -> None:
        """
        Compare the codec specified by a :py:class:`.VariableSpec` with the
        codec retrieved from the corresponding store array metadata.

        Called by: :py:meth:`.check_root_node_codecs` and
        :py:meth:`.check_child_node_codecs`.

        Calls: :py:meth:`.zarr_codec_key`.
        """
        # define comparison key
        assert isinstance(spec, dict)
        assert isinstance(array, Array)
        codec_key = partial(cls.zarr_codec_key, zarr_format)
        for k in ["filters", "compressors"]:
            spec_k = [] if (spec_k := spec.get(k)) is None else spec_k
            array_k = getattr(array, k)
            assert set(map(codec_key, spec_k)) == set(map(codec_key, array_k))

    @staticmethod
    def zarr_codec_key(zarr_format: Literal[2, 3], codec: Any) -> tuple:
        """
        Hashable partial Zarr codec specification, used as a comparison key.
        This key avoids direct comparisons for those configuration options that
        may be set adaptively by Zarr during the encoding process.

        Called by: :py:meth:`.compare_zarr_codecs`.
        """
        z = zarr_format
        if (d := codec.get_config() if z == 2 else codec.to_dict()):
            c = d if z == 2 else d["configuration"]
            match (n := d["id"] if z == 2 else d["name"]):
                case "zstd":
                    return (n, c["level"])
                case "delta" if z == 2:
                    return (n, c["dtype"])
                case "numcodecs.delta" if z == 3:
                    return (n, c["dtype"])
                case "lzma" if z == 2:
                    return (n, c["format"],
                            tuple(sorted(f["id"] for f in c["filters"])))
                case "numcodecs.lzma" if z == 3:
                    return (n, c["format"],
                            tuple(sorted(f["id"] for f in c["filters"])))
                case "blosc":
                    return (n, c["cname"], c["clevel"])
                case _:
                    raise NotImplementedError
        else:
            return ()
