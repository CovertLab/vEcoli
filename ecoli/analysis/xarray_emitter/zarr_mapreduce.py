
"""
Abstract base classes defining a `map-reduce`_ framework, which is intended
for executing complex numerical analysis pipelines on multi-variant, multi-seed,
multi-generation simulation workflows that were stored using the
:py:class:`.AsyncZarrBufferWriter` backend of the :py:class:`.XarrayEmitter`.

This framework is designed to take into account the full (in-)dependence
assumptions underlying the :ref:`storage <storage_layout>` and :ref:`variable
<variable_layout>` layouts of the :py:class:`.XarrayEmitter`, and to leverage
both :py:mod:`multiprocessing` parallelism and the
:py:mod:`~zarr.api.asynchronous` Zarr API.

Concrete analysis pipelines should be defined in separate modules, by
subclassing the following abstract classes: :py:class:`.ZarrMapReduceConfig`,
:py:class:`.ZarrMapReduceResult`, :py:class:`.ZarrMapReduce`, and, when
appropriate, :py:class:`.ZarrMapReducePlotConfig` and
:py:class:`.ZarrMapReducePlot`.

:py:meth:`ZarrMapReduceConfig.make_cli_parser` defines analysis-agnostic CLI
:py:class:`~argparse.ArgumentParser` arguments, including profiling
(`cProfile`_/`memray`_) and debugging options, as well as controls over whether
the numerical pipeline results are stored into or loaded from a ZIP file and
whether they are directly post-processed, e.g., by rendering plots.

.. _map-reduce: https://en.wikipedia.org/wiki/MapReduce
.. _cProfile: https://docs.python.org/3/library/profile.html#module-cProfile
.. _memray: https://bloomberg.github.io/memray/getting_started.html
"""

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from dataclasses import astuple, dataclass, field
from functools import partial
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import zarr
from zarr.api.asynchronous import open_group
from zarr.core.sync import sync

from ecoli.library.xarray_emitter.storage import (
    Substore,
    WorkflowConfig,
    WorkflowPaths,
    XarrayStoragePartition,
    coo_path,
    var_name,
)
from ecoli.library.xarray_emitter.view import ForestView
from ecoli.library.xarray_emitter.zarr_utils import get_async_array, get_async_group

from .mapreduce import AsyncDictItem, PoolDictItem, async_dict, pool_dict

if TYPE_CHECKING:

    from zarr.core.buffer import NDArrayLikeOrScalar
    from zarr.core.group import AsyncGroup
    from zarr.core.indexing import Selector

    from ecoli.library.xarray_emitter.storage import VariablePath

    from .mapreduce import Reducer

# ==============================================================================
# analysis configuration
# ==============================================================================


@dataclass(kw_only=True, slots=True)
class ZarrMapReduceConfig(ABC):
    """
    Configuration of a :py:class:`ZarrMapReduce` pipeline, excluding
    post-processing steps such as plot rendering.
    """

    # workflow config
    # ~~~~~~~~~~~~~~~~~ #
    #: Name of the analysis pipeline.
    #: Used to access :py:attr:`.WorkflowConfig.sim`.
    name: ClassVar[str]
    #: Variable names and coordinate descriptors used by the analysis pipeline.
    #: Validated against :py:attr:`.WorkflowConfig.sim`.
    variables: dict[VariablePath, Any] = field(init=False)

    # file I/O
    # ~~~~~~~~~~~~~~~~~ #
    #: File path for pipeline result data (Zarr ZIP file).
    result_file: pathlib.Path = field(init=False)
    #: File path for plots of pipeline results (e.g., SVG file).
    figure_file: pathlib.Path = field(init=False)
    #: Store the pipeline result data, instead of post-processing it.
    save_data: bool = False
    #: Load result data, instead of executing the pipeline.
    load_data: bool = False

    # compute resources
    # ~~~~~~~~~~~~~~~~~ #
    #: Number of :py:mod:`multiprocessing` processes.
    n_procs: int
    #: Number of Zarr threads per :py:mod:`multiprocessing` process.
    n_threads: int

    # ~~~~~~~~~~~~~~~~~ #
    #: Print debug information and reduce the plot size.
    debug: bool = False
    #: Execute pipeline inside `cProfile`_ (``n_procs == 1``).
    prof_time: bool = False
    #: Execute pipeline inside `memray`_.
    prof_mem: bool = False
    #: File path for profiling output (relative to ``./``).
    prof_file: pathlib.Path | None = None

    # ~~~~~~~~~~~~~~~~~ #

    def __post_init__(self) -> None:
        assert isinstance(self.save_data, bool)
        assert isinstance(self.load_data, bool)
        assert not (self.save_data and self.load_data)

        assert isinstance(self.n_procs, int) and self.n_procs > 0
        assert isinstance(self.n_threads, int) and self.n_threads > 0
        if self.load_data:
            if self.n_procs > 1:
                raise ValueError("Set `--procs 1` for `--load`.")
            if self.n_threads > 1:
                raise ValueError("Set `--threads 1` for `--load`.")
            if self.prof_file is not None:
                raise ValueError("No profiling support for `--load`.")

        assert isinstance(self.prof_time, bool)
        assert isinstance(self.prof_mem, bool)
        assert isinstance(self.prof_file, Path | None)
        match (self.prof_time, self.prof_mem, self.prof_file):
            case (True, True, _):
                raise TypeError("Choose one profiler at a time.")
            case (True, False, _) if self.n_procs > 1:
                raise ValueError("Set `--procs 1` for `--prof-time`.")
            case (True, False, f) | (False, True, f) if f == Path():
                raise ValueError("Set `--prof-file` when profiling.")
            case (False, False, f) if f is not None:
                raise ValueError("Only set `--prof-file` when profiling.")

        assert isinstance(self.debug, bool)

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def make_cli_parser(description: str, /) -> ArgumentParser:
        """
        This :py:class:`~argparse.ArgumentParser` may be further customised by a
        concrete pipeline.
        """
        parser = ArgumentParser(
            prog=Path(__file__).name, description=description,
            formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument(
            "config", type=str,
            help="simulation configuration JSON (relative to `configs/`)")
        parser.add_argument(
            "-p", "--procs", type=int, metavar="P", required=True,
            help="number of `multiprocessing` processes")
        parser.add_argument(
            "-t", "--threads", type=int, metavar="T", required=True,
            help="number of `zarr` threads per `multiprocessing` process")
        parser.add_argument(
            "--save", action="store_true",
            help="save analysis result to ZIP without post-processing")
        parser.add_argument(
            "--load", action="store_true",
            help="load analysis result from ZIP and post-process (--procs 1)")
        parser.add_argument(
            "--debug", action="store_true",
            help="print debug information and reduce plot size")
        parser.add_argument(
            "--prof-time", action="store_true",
            help="run analysis inside `cProfile` (--procs 1)")
        parser.add_argument(
            "--prof-mem", action="store_true",
            help="run analysis inside `memray`")
        parser.add_argument(
            "--prof-file", type=str, metavar="F",
            help="profiling output (relative to `./`)")
        return parser

    @classmethod
    def from_cli_parser(cls, args: Namespace, /, **kwargs) -> Self:
        """
        Interpret the CLI arguments returned by the parser from
        :py:meth:`.make_cli_parser`.
        """
        assert isinstance(args, Namespace)
        return cls(
            save_data=args.save, load_data=args.load, debug=args.debug,
            n_procs=args.procs, n_threads=args.threads,
            prof_time=args.prof_time, prof_mem=args.prof_mem,
            prof_file=args.prof_file,
            **kwargs)

    # ~~~~~~~~~~~~~~~~~ #

    def cfg_error(self, msg: str, path: str, /) -> str:
        return (f"{msg}:\n  "
                f"\"analysis_options.zarr_mapreduce.{self.name}{path}\"")

    def validate(self, workflow: WorkflowConfig, /) -> None:
        """
        Parse and validate the analysis configuration.

        Called by: :py:meth:`ZarrMapReduce.__init__`.

        Calls: :py:meth:`.validate_generic`, :py:meth:`.validate_specific`.
        """
        # load analysis config from workflow JSON
        assert isinstance(workflow, WorkflowConfig)
        analysis = (workflow.sim
                    .get("analysis_options", {}).get("zarr_mapreduce", {})
                    .get(self.name, {}))
        if not (isinstance(analysis, dict) and analysis):
            raise KeyError(self.cfg_error("Missing analysis configuration", ""))
        for k in ["paths", "variables", "parameters"]:
            if not analysis.get(k, {}):
                raise KeyError(self.cfg_error("Missing analysis configuration",
                                              f".{k}"))
        for k in ["result", "figure"]:
            if not (isinstance(p := analysis["paths"].get(k), str) and p):
                raise KeyError(self.cfg_error("Missing analysis configuration",
                                              f".paths.{k}"))
            if k == "result" and not p.endswith(".zip"):
                raise ValueError(self.cfg_error("Invalid analysis configuration",
                                                f".paths.{k}"))
            setattr(self, f"{k}_file", Path(p))
        variables = analysis["variables"]
        if not all(p == str(Path(p)) for p in variables):
            raise ValueError(self.cfg_error("Invalid analysis configuration",
                                            ".variables"))
        self.variables = variables

        # validate against simulation config
        self.validate_generic(workflow)
        self.validate_specific(workflow)

    def validate_generic(self, workflow: WorkflowConfig, /) -> None:
        """
        Analysis-agnostic validation checks of the *analysis configuration*
        against the *simulation configuration*.
        """
        view = ForestView.from_dict(workflow.sim["emitter_arg"]["view"])
        leaves = {str(lf.path) for lf in view.leaves.values()}
        if not set(self.variables.keys()).issubset(leaves):
            raise KeyError("Analysis requires variables missing from "
                           "the emitter configuration.")

    @abstractmethod
    def validate_specific(self, workflow: WorkflowConfig, /) -> None:
        """
        Analysis-specific validation checks of the *analysis configuration*
        against the *simulation configuration*.
        """
        ...

    def zarr_config(self) -> None:
        """
        Apply configuration options related to the Zarr library.
        """
        zarr.config.update({
            "async": {"concurrency": self.n_threads},
            "threading": {"max_workers": self.n_threads}})
        assert zarr.config.get("async.concurrency") == self.n_threads
        assert zarr.config.get("threading.max_workers") == self.n_threads


# ==============================================================================
# analysis result
# ==============================================================================


@dataclass(kw_only=True, frozen=True)
class ZarrMapReduceResult[AnalysisConfigT: ZarrMapReduceConfig](ABC):
    """
    Output type of a :py:class:`ZarrMapReduce` pipeline, with dedicated
    serialisation/deserialisation methods.

    .. note::

      This object should contain the full information required by
      post-processing tasks, including metadata.
    """

    @staticmethod
    @abstractmethod
    def zarr_config(cfg: AnalysisConfigT, /) -> None:
        """
        Apply configuration options related to the Zarr library.
        """
        ...

    @abstractmethod
    def to_zarr(self, cfg: AnalysisConfigT, /) -> None:
        """
        Serialise the pipeline results into a Zarr ZIP file.

        Calls: :py:meth:`.zarr_config`.
        """
        ...

    @classmethod
    @abstractmethod
    def from_zarr(self, cfg: AnalysisConfigT, /) -> Self:
        """
        Deserialise the pipeline results from a Zarr ZIP file.

        Calls: :py:meth:`.zarr_config`.
        """
        ...

    @abstractmethod
    def to_pandas(
        self, workflow_cfg: WorkflowConfig,
        /, max_rows_per_panel: int
    ) -> Any:
        r"""
        Export the pipeline results into a *grid* of `long-form`_
        :py:class:`~pandas.DataFrame`\ s suitable for plotting with
        `Vega-Altair`_.

        Called by: :py:meth:`.ZarrMapReducePlot.import_data`.

        .. note::
          The issue of `dataset size`_ is controlled by the following measures:

          - Each panel in the grid is exported as a separate
            :py:class:`~pandas.DataFrame`, rather than using `repeated`_ or
            `faceted`_ charts.
          - The size of each output :py:class:`~pandas.DataFrame` is limited by
            :py:attr:`.ZarrMapReducePlotConfig.max_rows_per_panel`, and
            `Vega-Altair`_ is configured accordingly.

        .. _long-form: https://altair-viz.github.io/user_guide/data.html#long-form-vs-wide-form-data
        .. _Vega-Altair: https://altair-viz.github.io/index.html
        .. _dataset size: https://altair-viz.github.io/user_guide/large_datasets.html#large-datasets
        .. _repeated: https://altair-viz.github.io/user_guide/compound_charts.html#repeated-charts
        .. _faceted: https://altair-viz.github.io/user_guide/compound_charts.html#faceted-charts
        """
        ...


# ==============================================================================
# analysis pipeline
# ==============================================================================


@dataclass(kw_only=True, slots=True, frozen=True)
class CoordDescriptor:
    """
    Analysis-related metadata and subarray `selector`_ for a single simulation
    variable.

    Constructed by: :py:meth:`.ZarrMapReduce.select_variable_coordinate` and
    :py:meth:`.ZarrMapReduce.select_time_coordinate`.

    .. _selector: https://zarr.readthedocs.io/en/stable/user-guide/arrays/#advanced-indexing
    """

    #: Labels of coordinate dimensions. Used for plotting.
    dim_names: list[str]
    #: A `selector`_ applicable to the Zarr array representing a variable.
    selector: Selector
    #: Unit annotation.
    unit: str | None

    def __post_init__(self) -> None:
        assert isinstance(self.dim_names, list)
        assert all(isinstance(d, str) for d in self.dim_names)
        assert isinstance(self.unit, str | None)

    def __eq__(self, other, /) -> bool:
        return (self.unit, self.dim_names) == (other.unit, other.dim_names)


# ------------------------------------------------------------------------------


class ZarrMapReduce[
    AnalysisConfigT: ZarrMapReduceConfig,
    ResultT: ZarrMapReduceResult,
    PlotConfigT: ZarrMapReducePlotConfig,
](ABC):
    """
    Driver class for map-reduce pipelines on Zarr stores produced by the
    :py:class:`.XarrayEmitter`.

    This abstract class defines a specialised data flow, which takes into
    account the full (in-)dependence assumptions underlying the :ref:`storage
    <storage_layout>` and :ref:`variable <variable_layout>` layouts of the
    :py:class:`.XarrayEmitter`, and which leverages both
    :py:mod:`multiprocessing` parallelism and the
    :py:mod:`~zarr.api.asynchronous` Zarr API. The data flow is organised into a
    method call hierarchy as follows::

      .compute()
      ├─ .mapreduce_workflow()
      │  ├─ [parallel] .sync_mapreduce_substore()
      │  │  └─ [async] .mapreduce_substore()
      │  │     ├─ [async] .load_substore()
      │  │     │  └─ [async] zarr.open_group()
      │  │     ├─ .select_partitions()                        ;  (abstract)
      │  │     ├─ [async] .select_substore_coordinates()
      │  │     │  ├─ [async] zarr.getitem()
      │  │     │  └─ [async] .select_variable_coordinate()    ;  (abstract)
      │  │     ├─ [async] .map_substore()
      │  │     │  └─ [async] .mapreduce_partition()
      │  │     │     ├─ [async] zarr.getitem()
      │  │     │     ├─ [async] .select_time_coordinate()     ;  (abstract)
      │  │     │     ├─ [async] .map_partition()
      │  │     │     │  └─ [async] .mapreduce_variable()
      │  │     │     │     ├─ [async] zarr.getitem()
      │  │     │     │     ├─ [async] .get_array_selection()  ;  (abstract)
      │  │     │     │     └─ [async] .reduce_variable()      ;  (abstract)
      │  │     │     └─ [async] .reduce_partition()           ;  (abstract)
      │  │     └─ .reduce_substore()                          ;  (abstract)
      │  └─ .reduce_workflow()                                ;  (abstract)
      └─ .post_process()                                      ;  (abstract)

    Concrete pipelines are defined by subclasses overloading the abstract
    methods, which are provided with full access to the locally relevant
    configuration options and Zarr objects at each level of the pipeline. For a
    complete example, see :py:class:`.MovingAvgPipeline`.

    .. note::

      This design is motivated by the following considerations:

      - Within a *workflow*, ``mapreduce`` operations on independent substores
        can be processed in parallel, up to the final top level ``reduce``
        operation.
      - Within each *independent substore* (lineage), the relevant partitions
        (generations) and variable coordinate subsets can be selected, in an
        analysis-specific way, at the top level without accessing all child
        arrays. Such selections help avoid unnecessary data transfers and
        decompressions of child arrays.
      - Similarly, within each *partition*, the relevant time coordinate subset
        can be selected, in an analysis-specific way, at the top level.
      - Once these selections are made, the retrieval of relevant child array
        subsets and the evaluation of their local ``reduce`` operations can be
        performed asynchronously, thus mitigating the latency cost of child
        arrays.
      - By performing intermediate ``reduce`` operations at the partition level
        and at the substore level, data marshalling can be further reduced.
    """

    def __init__(
        self, workflow_cfg: WorkflowConfig, analysis_cfg: AnalysisConfigT,
        plot_cfg: PlotConfigT
    ) -> None:
        assert isinstance(workflow_cfg, WorkflowConfig)
        assert issubclass(self.config_type, ZarrMapReduceConfig)
        assert isinstance(analysis_cfg, self.config_type)
        assert isinstance(plot_cfg, ZarrMapReducePlotConfig)
        self.workflow_cfg = workflow_cfg
        self.analysis_cfg = analysis_cfg
        self.analysis_cfg.validate(self.workflow_cfg)
        self.analysis_cfg.zarr_config()
        self.plot_cfg = plot_cfg
        self.store = WorkflowPaths.locate(self.workflow_cfg)

    @property
    @abstractmethod
    def config_type(self) -> type[AnalysisConfigT]:
        """
        Expected type of configuration data class.
        """
        ...

    @property
    @abstractmethod
    def result_type(self) -> type[ResultT]:
        """
        Expected type of pipeline result.
        """
        ...

    # ~~~~~~~~~~~~~~~~~ #
    # workflow level
    # ~~~~~~~~~~~~~~~~~ #

    def compute(self) -> ResultT:
        """
        Driver method.

        If a profiler is set in the :py:attr:`ZarrMapReduceConfig` constructor
        argument, then the map-reduce pipeline is wrapped inside a profiler, and
        post-processing is skipped. Otherwise, the map-reduce pipeline is
        executed and post-processed.

        In addition, :py:attr:`.ZarrMapReduceConfig.save_data` and
        :py:attr:`.ZarrMapReduceConfig.load_data` can be used to skip repeated
        executions of the map-reduce pipeline, e.g., while the post-processing
        code is under active development.

        Calls: :py:meth:`.mapreduce_workflow` and :py:meth:`.post_process`.
        """
        if (config := self.analysis_cfg).prof_time:
            # execute analysis pipeline inside time profiler
            from cProfile import Profile
            with Profile() as prof:
                result = self.mapreduce_workflow()
                prof.dump_stats(cast(Path, config.prof_file))

        elif config.prof_mem:
            # execute analysis pipeline inside memory profiler
            try:
                from memray import Tracker
            except ImportError:
                raise ImportError("Install the [prof] extra for profiling.")
            with Tracker(file_name=cast(Path, config.prof_file),
                         native_traces=True, follow_fork=True):
                result = self.mapreduce_workflow()

        else:
            # execute analysis pipeline or load its result
            result = (self.result_type.from_zarr(self.analysis_cfg)
                      if self.analysis_cfg.load_data else
                      self.mapreduce_workflow())
            # store or post-process pipeline result
            if self.analysis_cfg.save_data:
                result.to_zarr(self.analysis_cfg)
            else:
                self.post_process(self.workflow_cfg, self.analysis_cfg, result,
                                  self.plot_cfg)

        assert isinstance(result, self.result_type)
        return result

    @classmethod
    @abstractmethod
    def post_process(
        cls, workflow_cfg: WorkflowConfig, analysis_cfg: AnalysisConfigT,
        result: ResultT, plot_cfg: PlotConfigT
    ) -> None:
        """
        Downstream uses of map-reduce pipeline results, e.g., plot rendering.

        Called by: :py:meth:`.compute`.
        """
        ...

    def mapreduce_workflow(self) -> ResultT:
        """
        Entry point for the global ``mapreduce`` data flow.

        This method spawns a pool of parallel processes for executing
        ``mapreduce`` operations at the level of independent substores, and then
        performs the final global ``reduce`` operation.

        Called by: :py:meth:`.compute`.

        Calls: :py:meth:`.sync_mapreduce_substore`, :py:meth:`.reduce_workflow`.
        """
        worker = PoolDictItem(
            Substore.identity,
            partial(self.sync_mapreduce_substore,
                    self.workflow_cfg, self.analysis_cfg, self.store.root))
        return self.reduce_workflow(pool_dict(
            worker, iter(self.store),
            n_items=len(self.store), n_procs=self.analysis_cfg.n_procs))

    @abstractmethod
    def reduce_workflow(
        self, workflow_map: dict[Substore, Reducer], /
    ) -> ResultT:
        """
        Execute the final, global ``reduce`` operation.

        Called by: :py:meth:`.mapreduce_workflow`.
        """
        ...

    # ~~~~~~~~~~~~~~~~~ #
    # substore level
    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    def sync_mapreduce_substore(
        cls, workflow_cfg: WorkflowConfig, analysis_cfg: AnalysisConfigT,
        root: str, substore: Substore, /
    ) -> Reducer:
        """
        Entry point for the ``mapreduce`` data flow at the level of an
        independent substore.

        This method is merely a synchronisation barrier for
        :py:meth:`.mapreduce_substore`.

        Called by: :py:meth:`.mapreduce_workflow`.

        Calls: :py:meth:`.mapreduce_substore`.
        """
        return sync(
            cls.mapreduce_substore(workflow_cfg, analysis_cfg, root, substore))

    @classmethod
    async def mapreduce_substore(
        cls, workflow_cfg: WorkflowConfig, analysis_cfg: AnalysisConfigT,
        root: str, substore: Substore, /
    ) -> Reducer:
        """
        Access the root of an independent substore, select relevant partition
        subtrees, coordinate variables and dimensions, and execute ``mapreduce``
        operations up to the level of the independent substore.

        Called by: :py:meth:`.sync_mapreduce_substore`.

        Calls: :py:meth:`.load_substore`, :py:meth:`.select_partitions`,
        :py:meth:`.select_substore_coordinates`, :py:meth:`.map_substore`,
        :py:meth:`.reduce_substore`.
        """
        group, partitions = await cls.load_substore(workflow_cfg, root, substore)
        var_dscrs = await cls.select_substore_coordinates(analysis_cfg, group)
        return cls.reduce_substore(
            analysis_cfg, var_dscrs,
            await cls.map_substore(
                analysis_cfg, group,
                cls.select_partitions(analysis_cfg, substore, partitions),
                {v: d.selector for (v, d) in var_dscrs.items()}))

    @classmethod
    async def load_substore(
        cls, config: WorkflowConfig, root: str, substore: Substore, /
    ) -> tuple[zarr.AsyncGroup, list[XarrayStoragePartition]]:
        """
        Access the root of an independent substore, and detect its partitions.

        Called by: :py:meth:`.mapreduce_substore`.
        """
        group: AsyncGroup = await open_group(
            root, path=join(*astuple(substore)),
            mode="r", use_consolidated=True)
        t_coords: set[str] = {
            key async for key in group.array_keys()
            if XarrayStoragePartition.is_time_coo_name(key)}
        if (n_gens := len(t_coords)) > config.sim["generations"]:
            raise ValueError(
                f"Unexpected number of generations in \"{substore}\": {n_gens}")
        partitions = [
            XarrayStoragePartition.from_substore(config, substore, g)
            for g in range(1, 1 + n_gens)]
        if t_coords != {p.time_coo_name for p in partitions}:
            raise ValueError(
                f"Unexpected generations in \"{substore}\": {t_coords}")
        return (group, partitions)

    @staticmethod
    @abstractmethod
    def select_partitions(
        cfg: AnalysisConfigT, substore: Substore,
        partitions: list[XarrayStoragePartition], /
    ) -> list[XarrayStoragePartition]:
        """
        Select relevant partitions within an independent substore.

        Called by: :py:meth:`.mapreduce_substore`.
        """
        ...

    @classmethod
    async def select_substore_coordinates(
        cls, cfg: AnalysisConfigT, group: zarr.AsyncGroup, /
    ) -> dict[VariablePath, CoordDescriptor]:
        """
        Select relevant coordinate variables and dimensions within an
        independent substore.

        Called by: :py:meth:`.mapreduce_substore`.

        Calls: :py:meth:`.select_variable_coordinate`.
        """
        return await async_dict(
            AsyncDictItem(path, path,
                          cls.select_variable_coordinate(
                              cfg, path,
                              ((await get_async_group(group, path))
                               .attrs.get(var_name(path))),
                              await get_async_array(group, coo_path(path)),
                              match_coo))
            for (path, match_coo) in cfg.variables.items())

    @staticmethod
    @abstractmethod
    async def select_variable_coordinate(
        cfg: AnalysisConfigT, path: VariablePath, attr: Any,
        coo: zarr.AsyncArray, match_coo: str, /
    ) -> CoordDescriptor:
        """
        Select relevant coordinate dimensions within a selected variable.

        Called by: :py:meth:`.select_substore_coordinates`.
        """
        ...

    @classmethod
    async def map_substore(
        cls, cfg: AnalysisConfigT,
        group: zarr.AsyncGroup, partitions: list[XarrayStoragePartition],
        var_ixs: dict[VariablePath, Selector], /
    ) -> dict[XarrayStoragePartition, Reducer]:
        """
        Execute ``mapreduce`` operations for all relevant partitions within an
        independent substore.

        Called by: :py:meth:`.mapreduce_substore`.

        Calls: :py:meth:`.mapreduce_partition`.
        """
        return await async_dict(
            AsyncDictItem(p, str(p.generation),
                          cls.mapreduce_partition(cfg, group, p, var_ixs))
            for p in partitions)

    @staticmethod
    @abstractmethod
    def reduce_substore(
        cfg: AnalysisConfigT,
        var_dscrs: dict[VariablePath, CoordDescriptor],
        substore_map: dict[XarrayStoragePartition, Reducer], /
    ) -> Reducer:
        """
        Execute the ``reduce`` operation at the level of an independent
        substore.

        Called by: :py:meth:`.mapreduce_substore`.
        """
        ...

    # ~~~~~~~~~~~~~~~~~ #
    # partition level
    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    async def mapreduce_partition(
        cls, cfg: AnalysisConfigT, group: zarr.AsyncGroup,
        p: XarrayStoragePartition, var_ixs: dict[VariablePath, Selector], /
    ) -> Reducer:
        """
        Select the relevant time coordinate dimensions within a relevant
        partition, and execute ``mapreduce`` operations up to the level of the
        relevant partition.

        Called by: :py:meth:`.map_substore`.

        Calls: :py:meth:`.select_time_coordinate`, :py:meth:`.map_partition`,
        :py:meth:`.reduce_partition`.
        """
        time_coo = await get_async_array(group, p.time_coo_name)
        time_var = await get_async_array(group, p.time_var_name)
        time_dscr = await cls.select_time_coordinate(
            cfg, p, cast(str, group.attrs[p.time_var_name]),
            time_coo, time_var)
        return await cls.reduce_partition(
            cfg, time_dscr, time_coo, time_var,
            await cls.map_partition(cfg, group, p, time_dscr.selector, var_ixs))

    @staticmethod
    @abstractmethod
    async def select_time_coordinate(
        cfg: AnalysisConfigT, p: XarrayStoragePartition, attr: str,
        time_coo: zarr.AsyncArray, time_var: zarr.AsyncArray, /
    ) -> CoordDescriptor:
        """
        Select relevant time coordinate dimensions within a relevant partition.

        Called by: :py:meth:`.mapreduce_partition`.
        """
        ...

    @classmethod
    async def map_partition(
        cls, cfg: AnalysisConfigT, group: zarr.AsyncGroup,
        p: XarrayStoragePartition,
        time_ix: Selector, var_ixs: dict[VariablePath, Selector], /
    ) -> dict[VariablePath, Reducer]:
        """
        Execute ``mapreduce`` operations for all relevant variables within a
        relevant partition.

        Called by: :py:meth:`.mapreduce_partition`.

        Calls: :py:meth:`.mapreduce_variable`.
        """
        return await async_dict(
            AsyncDictItem(path, path,
                          cls.mapreduce_variable(
                              cfg, group, p, path, time_ix, var_ix))
            for (path, var_ix) in var_ixs.items())

    @staticmethod
    @abstractmethod
    async def reduce_partition(
        cfg: AnalysisConfigT, time_dscr: CoordDescriptor,
        time_coo: zarr.AsyncArray, time_var: zarr.AsyncArray,
        partition_map: dict[VariablePath, Reducer], /
    ) -> Reducer:
        """
        Execute the ``reduce`` operation at the level of a relevant partition,
        using both the time coordinate and the inner reduced variables.

        Called by: :py:meth:`.mapreduce_partition`.
        """
        ...

    # ~~~~~~~~~~~~~~~~~ #
    # variable level
    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    async def mapreduce_variable(
        cls, cfg: AnalysisConfigT, group: zarr.AsyncGroup,
        p: XarrayStoragePartition, path: VariablePath,
        time_ix: Selector, var_ix: Selector, /
    ) -> Reducer:
        """
        Retrieve the relevant subset of a relevant simulation variable, and
        execute the local ``reduce`` operation for it.

        Called by: :py:meth:`.map_partition`.

        Calls: :py:meth:`.get_array_selection`, :py:meth:`.reduce_variable`.
        """
        return await cls.reduce_variable(
            cfg, path,
            await cls.get_array_selection(
                cfg, path, time_ix, var_ix,
                await get_async_array(group, join(path, p.dynamic_suffix))))

    @staticmethod
    @abstractmethod
    async def get_array_selection(
        cfg: AnalysisConfigT, path: VariablePath,
        time_ix: Selector, var_ix: Selector, var: zarr.AsyncArray, /
    ) -> NDArrayLikeOrScalar:
        """
        Load and decompress the relevant subset of a relevant variable.

        Called by: :py:meth:`.mapreduce_variable`.

        .. note::

          What fraction of the underlying chunked array will actually be loaded
          into memory and decompressed depends on the ordering of dimensions, on
          chunk sizes, on the filter and compression codecs, as well as on Zarr
          library internals that may change over time.

          When preparing large simulation and analysis workflows, in particular
          for analyses that require only small fractions of stored arrays, users
          may benefit from empirically testing the configuration-specific
          resource usage by profiling sample analyses, see :py:meth:`.compute`.
        """
        ...

    @staticmethod
    @abstractmethod
    async def reduce_variable(
        cfg: AnalysisConfigT, path: VariablePath, data: NDArrayLikeOrScalar, /
    ) -> Reducer:
        """
        Execute the ``reduce`` operation at the level of a single relevant
        variable, without using the time coordinate yet.

        Called by: :py:meth:`.mapreduce_variable`.
        """
        ...


# ==============================================================================
# analysis plot
# ==============================================================================


@dataclass(kw_only=True, slots=True)
class ZarrMapReducePlotConfig(ABC):
    """
    Configuration of a :py:class:`ZarrMapReducePlot`. This object is expected to
    mostly contain analysis-specific rendering options.
    """

    #: Used by :py:meth:`.ZarrMapReduceResult.to_pandas`.
    max_rows_per_panel: int = 20_000


# ------------------------------------------------------------------------------


@dataclass(slots=True)
class ZarrMapReducePlot[
    AnalysisConfigT: ZarrMapReduceConfig,
    ResultT: ZarrMapReduceResult,
    PlotConfigT: ZarrMapReducePlotConfig
](ABC):
    """
    Container for the full information required to produce analysis-specific
    plots.
    """

    workflow_cfg: WorkflowConfig
    analysis_cfg: AnalysisConfigT
    result: ResultT
    plot_cfg: PlotConfigT
    plot_data: Any = None

    # ~~~~~~~~~~~~~~~~~ #

    def __post_init__(self) -> None:
        assert isinstance(self.workflow_cfg, WorkflowConfig)
        assert isinstance(self.analysis_cfg, ZarrMapReduceConfig)
        assert isinstance(self.result, ZarrMapReduceResult)
        assert isinstance(self.plot_cfg, ZarrMapReducePlotConfig)
        assert isinstance(self.plot_cfg.max_rows_per_panel, int)
        self.import_data()

    def import_data(self) -> None:
        """
        Load plot data in a format suitable for the rendering engine.

        Called by: ``.__post_init__()``.

        Calls: :py:meth:`.ZarrMapReduceResult.to_pandas`.
        """
        self.plot_data = self.result.to_pandas(self.workflow_cfg,
                                               self.plot_cfg.max_rows_per_panel)

    @abstractmethod
    def render(self) -> Any:
        """
        Driver method. Renders and stores plots.
        """
        ...
