
"""
Unit and integration tests for :py:mod:`.xarray_emitter` and its submodules.
"""


from contextlib import ContextDecorator
from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from typing import Any, Self, final, cast

import numpy as np
from pytest import MonkeyPatch, mark, param, raises
from xarray import DataArray, DataTree, open_datatree
from zarr import Group, open_consolidated

from ecoli.library.test_utils import PatchConfig, filter_warnings
from ecoli.library.xarray_emitter.emit_path import EmitPath, EmitPathType
from ecoli.library.xarray_emitter.storage import XarrayStoragePartition
# from ecoli.library.xarray_emitter.emitter import XarrayEmitter
from ecoli.library.xarray_emitter.zarr_writer import (
    AsyncZarrBufferWriter, group_tree)
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
    zarr_format: int
    threaded: bool
    debug: bool
    interval: int

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
    zarr_format: int
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

    @classmethod
    @filter_warnings(ecolisim_warnings)
    @filter_warnings(AsyncZarrBufferWriter.warnings_all())
    @mark.parametrize(
        "num_generations, last_success, interval, zarr_format, threaded, debug",
        [param(*args, **kwargs,
               id="gen_{}-succ_{}-intvl_{}_zarr_{}-thrd_{}-dbg_{}".format(*args))
         for (args, kwargs) in [
            ((1, False, 1, 2, False, True ), {}),
            ((2, True,  3, 2, True,  False), {}),
            ((2, True,  2, 3, True,  True), {"marks": mark.basic_workflow}),
            ((3, False, 1, 3, True,  False), {})
         ]])
    def test_workflow(
        cls, monkeypatch: MonkeyPatch, tmp_path: Path,
        num_generations: int, last_success: bool, interval: int,
        zarr_format: int, threaded: bool, debug: bool
    ):
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
                workdir, zarr_format, threaded, debug, interval))

        # step through workflow
        hline = "=" * 79
        for g in range(1, num_generations + 1):
            print(f"\n{hline}\nGeneration: {g}\n{hline}")
            # execute simulation
            success = (g < num_generations) or last_success
            partition = cast(XarrayStoragePartition, wf.sim_gen(success))
            # read emitted data
            with StoreResult(store, partition, zarr_format) as result:
                # validate emitted data
                cls.check_tree(result)
                cls.check_encoding(result)
                cls.check_log(result)
                cls.check_success(result, success)
                cls.check_time(result, interval)

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def check_tree(res: StoreResult) -> None:
        """
        Check the basic integrity of the round-tripped
        :py:class:`xarray.DataTree`.
        """
        # let Zarr traverse the store
        print()
        print(group_tree(res.zarr))

        # let Xarray traverse the store
        print()
        print(res.xarray)

        # inspect current and previous generations
        p: XarrayStoragePartition | None = res.partition
        while p is not None:
            # look for expected fields in the root node
            assert len(res.xarray.attrs[p.sim_id])
            assert p.time_var_name in res.xarray.attrs
            assert p.time_coo_name in res.xarray.coords
            assert p.time_var_name in res.xarray.data_vars
            p = p.parent if p.generation > 1 else None

    @staticmethod
    def check_encoding(res: StoreResult) -> None:
        pass

    @staticmethod
    def check_log(res: StoreResult) -> None:
        """
        Check the integrity of the write log.
        """
        assert {"sim_step", "sim_time"} == set(
            res.xarray.attrs.get(res.partition.log_attr_name, {}).keys())

    @staticmethod
    def check_success(res: StoreResult, success: bool) -> None:
        """
        Check the validity of the success flag.
        """
        assert success == res.xarray.attrs.get(
            res.partition.success_attr_name, False)

    @staticmethod
    def check_time(res: StoreResult, interval: int) -> None:
        """
        Check the validity of the time variable.
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
