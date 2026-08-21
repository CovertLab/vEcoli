
"""
Analysis pipeline for a `vEcoli` simulation workflow emitted to a Zarr store:

  - selects variables and their coordinate indices
  - computes min/max values across the entire workflow
  - computes moving average timeseries per cell lineage
  - groups moving average timeseries by generation and variant
  - plots a grid of confidence intervals over moving average timeseries
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from functools import cached_property
from math import ceil
from pathlib import Path
from textwrap import wrap
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast
from zipfile import ZIP_DEFLATED

import altair
import numpy as np
import pandas
import zarr
from altair import (
    Chart,
    Color,
    ColorScheme,
    Title,
    X,
    Y,
    data_transformers,
    hconcat,
    value,
    vconcat,
)
from bottleneck import move_mean, nanmax, nanmin
from numpy import array, array_equal, flatnonzero, ndarray, stack, str_, strings
from numpy.typing import NDArray
from pandas import DataFrame, RangeIndex, Series
from zarr import create_array, create_group, open_group
from zarr.storage import ZipStore

from ecoli.analysis.xarray_emitter.mapreduce import (
    Chain,
    MinMax,
    ReducerDistribute,
    ReducerMapping,
    Unique,
)
from ecoli.analysis.xarray_emitter.zarr_mapreduce import (
    CoordDescriptor,
    ZarrMapReduce,
    ZarrMapReduceConfig,
    ZarrMapReducePlot,
    ZarrMapReducePlotConfig,
    ZarrMapReduceResult,
)
from ecoli.library.xarray_emitter.emit_predicate import (
    ConjunctiveEmitPredicate,
    SubsampleSteps,
)
from ecoli.library.xarray_emitter.storage import (
    VARIANT_PREFIX,
    Substore,
    VariablePath,
    WorkflowConfig,
)
from ecoli.library.xarray_emitter.utils import filter_warnings
from ecoli.library.xarray_emitter.zarr_utils import (
    get_ndarray,
    get_rectilinear_ndarray,
    parse_codecs,
    zarr_warnings,
)

if TYPE_CHECKING:

    from zarr.core.buffer import NDArrayLikeOrScalar
    from zarr.core.indexing import Selector

    from ecoli.analysis.xarray_emitter.mapreduce import Reducer
    from ecoli.library.xarray_emitter.storage import XarrayStoragePartition

# mypy: disable-error-code="union-attr"

# ==============================================================================
# constants
# ==============================================================================


sec_unit = "[s]"
min_unit = "[min]"
sec_per_min = 60


# ==============================================================================
# analysis configuration
# ==============================================================================


@dataclass(kw_only=True, slots=True)
class MovingAvgConfig(ZarrMapReduceConfig):

    name: ClassVar[str] = "moving_avg"
    t_var: ClassVar[str] = "time"

    #: Width of the emitted time grid. Read from
    #: :py:attr:`.SubsampleSteps.interval` in the simulation configuration.
    interval: int = field(init=False)
    #: Rolling window size for the moving average. Read from the analysis
    #: configuration, and rescaled by :py:attr:`.interval`.
    window: int = field(init=False)
    #: Threshold window size for the moving average. Read from the analysis
    #: configuration, and rescaled by :py:attr:`.interval`.
    min_window: int | None = field(init=False)

    def validate_specific(self, workflow: WorkflowConfig) -> None:

        # check variable names
        assert self.t_var not in self.variables

        # check emit predicate
        pred = ConjunctiveEmitPredicate.build(
            workflow.sim["emitter_arg"]["transducer"]["predicate"]
        )
        if not (
            sum(map(len, pred)) == 1
            and isinstance(subsample := next(iter(next(iter(pred)))),
                           SubsampleSteps)
        ):
            raise TypeError(
                "Analysis expects the following simulation configuration:\n"
                "  \"emitter_arg.transducer.predicate\": "
                "[[{\"subsample\": {\"interval\": INT}}]]")
        self.interval = subsample.interval

        # check moving window size
        params = (workflow.sim["analysis_options"]["zarr_mapreduce"]
                  [self.name]["parameters"])
        if not (
            isinstance(window := params.get("window"), int)
            and self.interval <= window
        ):
            raise ValueError(self.cfg_error("Invalid analysis configuration:\n",
                                            ".parameters.window"))
        self.window = window // self.interval

        # check moving window threshold
        if not (
            (min_window := params.get("min_window")) is None
            or (isinstance(min_window, int)
                and self.interval < min_window <= window)
        ):
            raise ValueError(self.cfg_error("Invalid analysis configuration:\n",
                                            ".parameters.min_window"))
        self.min_window = (None if (min_window is None)
                           else min_window // self.interval)


# ==============================================================================
# analysis result
# ==============================================================================


@dataclass(kw_only=True, frozen=True)
class MovingAvgResult(ZarrMapReduceResult[MovingAvgConfig]):

    # internal representation
    # ~~~~~~~~~~~~~~~~~ #
    #: Local time bounds (by ``generation``).
    t_minmax: dict[int, NDArray]
    #: Variable coordinate labels (by ``variable/dimension``).
    y_names: dict[VariablePath, list[str]]
    #: Variable units (by ``variable``).
    y_units: dict[VariablePath, str | None]
    #: Global variable bounds (by ``variable/dimension``).
    y_minmax: dict[VariablePath, NDArray]
    #: Variable time series
    #: (by ``generation/variant/variable/lineage/dimension``).
    trajs: dict[int, dict[str, dict[VariablePath, list[tuple[NDArray, NDArray]]]]]

    # plotting interface
    # ~~~~~~~~~~~~~~~~~ #
    #: Column names for the "long-form" output of :py:meth:`.to_pandas`.
    #:
    #: Used by: :py:meth:`.subsample_concat_lineages`,
    #: :py:meth:`.concat_variants`.
    panel_levels: ClassVar[list[str]] = ["variant", "lineage", "t", "y"]

    # ~~~~~~~~~~~~~~~~~ #

    def __post_init__(self) -> None:
        """
        Check consistency of keys and array dimensions.
        """
        assert list(self.trajs.keys()) == self.generations
        assert all(vnt.startswith(VARIANT_PREFIX) for vnt in self.variants)
        assert all(isinstance(vbl, str) for vbl in self.variables)
        assert list(self.y_units.keys()) == self.variables
        assert list(self.y_minmax.keys()) == self.variables
        assert all(isinstance(d, str)
                   for dims in self.y_names.values() for d in dims)
        assert all(isinstance(u, str | None) for u in self.y_units.values())
        assert set(self.variables) == set.union(*(
            set(vnt.keys())
            for gen in self.trajs.values() for vnt in gen.values()))
        assert all(isinstance(t, ndarray) and t.shape == (2,)
                   for t in self.t_minmax.values())
        assert all(isinstance(y, ndarray) and y.ndim == 2 and y.shape[1] == 2
                   for y in self.y_minmax.values())
        assert all(
            isinstance(t, ndarray) and t.ndim == 1
            and lin.shape == (len(t), self.dimensions[vbl])
            for g in self.generations for (vnt, vbls) in self.trajs[g].items()
            for (vbl, lins) in vbls.items() for (t, lin) in lins)

    # ~~~~~~~~~~~~~~~~~ #

    @cached_property
    def generations(self) -> list[int]:
        gens = list(self.t_minmax.keys())
        assert gens == list(range(1, 1 + len(gens)))
        return gens

    @cached_property
    def variants(self) -> list[str]:
        vrns = {tuple(sorted(g.keys())) for g in self.trajs.values()}
        assert 1 == len(vrns)
        return list(next(iter(vrns)))

    @cached_property
    def variables(self) -> list[VariablePath]:
        return list(self.y_names.keys())

    @cached_property
    def dimensions(self) -> dict[str, int]:
        y_shapes = {v: y.shape for (v, y) in self.y_minmax.items()}
        assert all(len(sh) == 2 and sh[1] == 2 for sh in y_shapes.values())
        return {v: sh[0] for (v, sh) in y_shapes.items()}

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def zarr_config(_: MovingAvgConfig) -> None:
        zarr.config.set({"array.rectilinear_chunks": True})

    def to_zarr(self, cfg: MovingAvgConfig) -> None:
        assert isinstance(cfg, MovingAvgConfig)
        self.zarr_config(cfg)
        with ZipStore(cfg.result_file, mode="w", compression=ZIP_DEFLATED) as store:
            root = create_group(
                store, path="", zarr_format=3, attributes={
                    "generations": self.generations,
                    "variants": self.variants,
                    "variables": self.variables,
                    "dimensions": self.dimensions,
                    "y_units": self.y_units
            })
            root.create_array(
                "t_minmax", data=array(list(self.t_minmax.values())))
            yns = create_group(store, path="y_names")
            with filter_warnings([zarr_warnings["string"]]):
                for (v, y_n) in self.y_names.items():
                    yns[v] = array(y_n)
            yms = create_group(store, path="y_minmax")
            for (v, y_m) in self.y_minmax.items():
                yms[v] = y_m
            lin0 = self.trajs[self.generations[0]][
                self.variants[0]][self.variables[0]][0]
            t_codecs = parse_codecs(
                3, category="delta", dtype=lin0[0].dtype.str)
            y_codecs = parse_codecs(
                3, category="num", dtype=lin0[1].dtype.str)
            for (gen, vnts) in self.trajs.items():
                for (vnt, vbls) in vnts.items():
                    for (vbl, lins) in vbls.items():
                        pfx = f"trajs/{gen}/{vnt}/{vbl}"
                        create_array(
                            store=store, name=f"{pfx}/t",
                            dtype=lins[0][0].dtype,
                            shape=(sum(t_lens := [len(t) for (t, _) in lins]),),
                            chunks=[t_lens],
                            **t_codecs
                        )[:] = np.concat([t for (t, _) in lins])
                        create_array(
                            store=store, name=f"{pfx}/y",
                            dtype=lins[0][1].dtype,
                            shape=(sum(y_lens := [len(y) for (_, y) in lins]),
                                    self.dimensions[vbl]),
                            chunks=[y_lens, [self.dimensions[vbl]]],
                            **y_codecs
                        )[:] = np.concat([y for (_, y) in lins])
            if cfg.debug:
                print()
                print(root.info_complete())
                print()
                print(root.tree())

    @classmethod
    def from_zarr(cls, cfg: MovingAvgConfig) -> Self:
        assert isinstance(cfg, MovingAvgConfig)
        cls.zarr_config(cfg)
        with ZipStore(cfg.result_file, mode="r") as store:
            root = open_group(store, mode="r")
            if cfg.debug:
                print()
                print(root.info_complete())
                print()
                print(root.tree())
            generations = cast(list[int], root.attrs["generations"])
            variants = cast(list[str], root.attrs["variants"])
            variables = cast(list[VariablePath], root.attrs["variables"])
            units = cast(dict[str, str], root.attrs["y_units"])
            return cls(
                t_minmax={g: get_ndarray(root, "/t_minmax")[g - 1]
                          for g in generations},
                y_names={v: get_ndarray(root, f"/y_names/{v}").tolist()
                         for v in variables},
                y_units={v: y if y else None
                         for (v, y) in units.items()},
                y_minmax={v: get_ndarray(root, f"/y_minmax/{v}")
                          for v in variables},
                trajs={
                    gen: {
                        vnt: {
                            vbl: list(zip(
                                get_rectilinear_ndarray(
                                    root, f"trajs/{gen}/{vnt}/{vbl}/t"),
                                (chk[0] for chk in
                                 get_rectilinear_ndarray(
                                     root, f"trajs/{gen}/{vnt}/{vbl}/y"))
                            ))
                            for vbl in variables
                        }
                        for vnt in variants
                    }
                    for gen in generations
                }
            )

    # ~~~~~~~~~~~~~~~~~ #

    def to_pandas(
        self, workflow_cfg: WorkflowConfig, max_rows_per_panel: int
    ) -> tuple[list[str],
               dict[int, dict[VariablePath, dict[int, pandas.DataFrame]]]]:
        """
        Calls: :py:meth:`.variant_label`, :py:meth:`.subsample_concat_lineages`,
        :py:meth:`.concat_variants`.

        Returns:
          - variant labels
          - grid (by ``generation/variable/dimension``) of `long-form`_ panel
            data (by :py:attr:`.panel_levels`).

        .. _long-form: https://altair-viz.github.io/user_guide/data.html#long-form-vs-wide-form-data
        """
        vnt_labels = {vnt: self.variant_label(workflow_cfg, vnt)
                      for vnt in
                      sorted(self.variants, key=WorkflowConfig.variant_index)}
        return (
            list(vnt_labels.values()),
            {g: self.concat_variants(
                vnt_labels,
                self.subsample_concat_lineages(g, max_rows_per_panel))
             for g in self.generations})

    def variant_label(self, workflow_cfg: WorkflowConfig, variant: str) -> str:
        """
        Determine the label for a variant configuration in the plot legend.

        Called by: :py:meth:`.to_pandas`.

        .. note::
          To allow for custom variant configurations, this method needs to be
          generalised or overloaded.
        """
        params = workflow_cfg.variant_params(variant)
        assert all(isinstance(p, str) for p in params)
        if not params:
            return "baseline"
        elif (cond := params.get("condition")) and isinstance(cond, str):
            return cond
        else:
            raise NotImplementedError(
                "variant legend labels require generalization")

    def subsample_concat_lineages(
        self, gen: int, max_rows_per_panel: int
    ) -> dict[VariablePath, dict[int, dict[str, Series]]]:
        """
        Fairly subsample trajectory ensembles along the time dimension, in order
        to control the total dataset size per Vega-Altair plot panel -- see
        :py:meth:`.ZarrMapReduceResult.to_pandas`.

        Called by: :py:meth:`.to_pandas`.
        """
        # calculate temporal subsampling step
        n_lins = max(max(map(len, vbls.values()))
                     for vbls in self.trajs[gen].values())
        sz_vnt = max_rows_per_panel / len(self.variants)
        sz_lin = sz_vnt / n_lins
        max_len_lin = max(max(dim[1].shape[0] for dim in dims)
                          for vbls in self.trajs[gen].values()
                          for dims in vbls.values())
        step_lin = ceil(max_len_lin / sz_lin)

        # subsample & gather
        srs_vbl: dict[VariablePath, dict[int, dict[str, Series]]] = {}
        for (vnt, vbls) in self.trajs[gen].items():
            for (vbl, lins) in vbls.items():
                for dim in range(self.dimensions[vbl]):
                    # aggregate timeseries across lineages
                    sr_dim = pandas.concat(
                        [Series(
                            # apply temporal subsampling
                            y[::step_lin, dim],
                            # time coordinate
                            index=Series(t[::step_lin],
                                         name=self.panel_levels[-2]),
                            # variable coordinate
                            name=self.panel_levels[-1])
                        # iterate over pipeline results
                        for (t, y) in lins],
                        # concatenation coordinate
                        axis=0,
                        keys=range(len(lins)),
                        names=[self.panel_levels[-3]])
                    assert sr_dim.index.names == self.panel_levels[-3:-1]
                    # check temporal subsampling
                    assert len(sr_dim.index) <= sz_vnt
                    # collect
                    srs_vbl.setdefault(vbl, {}).setdefault(dim, {})[vnt] = sr_dim
        return srs_vbl

    def concat_variants(
        self, vnt_labels: dict[str, str],
        srs_vbl: dict[VariablePath, dict[int, dict[str, Series]]]
    ) -> dict[VariablePath, dict[int, pandas.DataFrame]]:
        """
        For each Vega-Altair plot panel, concatenate the variant
        :py:class:`~pandas.Series` into a single :py:class:`~pandas.DataFrame`,
        and apply the variant label transformations determined by
        :py:meth:`.variant_label`.

        Called by: :py:meth:`.to_pandas`.
        """
        dfs_panel: dict[VariablePath, dict[int, DataFrame]] = {}
        for (vbl, sr_dims) in srs_vbl.items():
            for (dim, srs_vnt) in sr_dims.items():
                df_dim = pandas.concat(
                    # re-label variants for plot legend
                    {lbl: srs_vnt[vnt] for (vnt, lbl) in vnt_labels.items()},
                    # concatenation coordinate
                    axis=0, names=[self.panel_levels[0]]
                ).reset_index(level=self.panel_levels[:-1])
                assert isinstance(df_dim.index, RangeIndex)
                assert array_equal(df_dim.columns, self.panel_levels)
                dfs_panel.setdefault(vbl, {})[dim] = df_dim
        return dfs_panel


# ==============================================================================
# analysis plot config
# ==============================================================================


@dataclass(kw_only=True, slots=True)
class MovingAvgPlotConfig(ZarrMapReducePlotConfig):

    # data
    # ~~~~~~~~~~~~~~~~~ #
    y_rescale: tuple[tuple[str, Any],...] = (("type", "symlog"), ("constant", .5))
    band_extent: str = "ci"
    debug_num_dims: int = 2

    # grid
    # ~~~~~~~~~~~~~~~~~ #
    grid_spacing: int = 0
    panel_offset: int = 10
    legend_offset: int = 30

    # panel
    # ~~~~~~~~~~~~~~~~~ #
    panel_height: int = 120
    min_panel_width: int = 230
    max_avg_panel_width: int = 250

    # axes
    # ~~~~~~~~~~~~~~~~~ #
    y_label_width: int = 40
    title_shift_x: int = 10
    title_shift_y: int = 20
    tick_format: str = "0<6.1e"

    # fonts
    # ~~~~~~~~~~~~~~~~~ #
    title_fontsize: int = 16
    panel_fontsize: int = 14
    axis_fontsize: int = 12
    tick_fontsize: int = 10
    title_textwidth: int = 30

    # style
    # ~~~~~~~~~~~~~~~~~ #
    mean_line_size: int = 4
    color_scheme: str = "set2"
    grid_opacity: float = .5

    # ~~~~~~~~~~~~~~~~~ #

    def __post_init__(self) -> None:
        assert isinstance(self.band_extent, str)
        assert isinstance(self.debug_num_dims, int)

        assert isinstance(self.grid_spacing, int)
        assert isinstance(self.panel_offset, int)
        assert isinstance(self.legend_offset, int)

        assert isinstance(self.panel_height, int)
        assert isinstance(self.min_panel_width, int)
        assert isinstance(self.max_avg_panel_width, int)
        assert self.min_panel_width <= self.max_avg_panel_width

        assert isinstance(self.y_label_width, int)
        assert isinstance(self.title_shift_x, int)
        assert isinstance(self.title_shift_y, int)
        assert isinstance(self.tick_format, str)

        assert isinstance(self.title_fontsize, int)
        assert isinstance(self.panel_fontsize, int)
        assert isinstance(self.axis_fontsize, int)
        assert isinstance(self.tick_fontsize, int)
        assert isinstance(self.title_textwidth, int)

        assert isinstance(self.mean_line_size, int)
        assert isinstance(self.color_scheme, str)
        assert 0 <= self.grid_opacity <= 1


# ==============================================================================
# analysis pipeline
# ==============================================================================


class MovingAvgPipeline(ZarrMapReduce[
    MovingAvgConfig, MovingAvgResult, MovingAvgPlotConfig
]):

    @property
    def config_type(self) -> type[MovingAvgConfig]:
        return MovingAvgConfig

    @property
    def result_type(self) -> type[MovingAvgResult]:
        return MovingAvgResult

    # ~~~~~~~~~~~~~~~~~ #

    @classmethod
    def post_process(
        cls, workflow_cfg: WorkflowConfig, analysis_cfg: MovingAvgConfig,
        result: MovingAvgResult, plot_cfg: MovingAvgPlotConfig
    ) -> None:
        """
        Calls: :py:meth:`.MovingAvgPlot.render`.
        """
        MovingAvgPlot(workflow_cfg, analysis_cfg, result, plot_cfg).render()

    # ~~~~~~~~~~~~~~~~~ #

    def reduce_workflow(
        self, workflow_map: dict[Substore, Reducer], /
    ) -> MovingAvgResult:
        """
        Transpose index dimensions, check consistency of coordinate descriptors,
        aggregate variable coordinate bounds and time coordinate bounds, and
        group time series by generation and variant.
        """
        wf_map = cast(dict[Substore, ReducerMapping[str]], workflow_map)

        # swap index dimensions: substore/variable -> variable/substore
        by_vbl: ReducerDistribute[VariablePath, Substore] = ReducerDistribute(
            cast(dict[Substore, ReducerMapping[VariablePath]], {
                substore: wf.reducers["variables"]
                for (substore, wf) in wf_map.items()
            })
        ).distribute()
        variables = {
            vbl: Unique[CoordDescriptor].reduce(list(cast(
                dict[Substore, Unique[CoordDescriptor]],
                substores.reducers
            ).values())).extract()[0]
            for (vbl, substores) in by_vbl.reducers.items()
        }

        # swap index dimensions: substore/generation -> generation/substore
        by_gen: ReducerDistribute[int, Substore] = ReducerDistribute(
            cast(dict[Substore, ReducerMapping[int]], {
                substore: wf.reducers["generations"]
                for (substore, wf) in wf_map.items()
            })
        ).distribute()

        # perform global aggregations
        return MovingAvgResult(

            # aggregate time bounds (across substores)
            t_minmax={
                gen: np.concat(MinMax.reduce([
                    st.reducers["t_minmax"]  # type: ignore[attr-defined]
                    for st in substores.reducers.values()
                ]).extract())
                for (gen, substores) in by_gen.reducers.items()
            },

            # extract variable descriptors
            y_units={vbl: dscr.unit for (vbl, dscr) in variables.items()},
            y_names={vbl: dscr.dim_names for (vbl, dscr) in variables.items()},

            # aggregate variable bounds (across generations & substores)
            y_minmax={
                vbl: stack(ys, axis=-1)
                for (vbl, ys) in
                ReducerMapping[VariablePath].reduce([
                    st.reducers["y_minmax"]  # type: ignore[attr-defined]
                    for (gen, substores) in by_gen.reducers.items()
                    for st in substores.reducers.values()
                ]).extract().items()
            },

            # aggregate trajectory samples (across lineages)
            trajs={
                gen: {
                    variant: ReducerMapping[VariablePath].reduce([
                        lin.reducers["trajs"]  # type: ignore[attr-defined]
                        for lin in lins.values()
                    ]).extract()
                    # split index dimension: generation/variant/lineage
                    for (variant, lins)
                    in Substore.groupby_variant(substores.reducers).items()
                }
                for (gen, substores) in by_gen.reducers.items()
            }
        )

    @staticmethod
    def reduce_substore(
        cfg: MovingAvgConfig,
        var_dscrs: dict[VariablePath, CoordDescriptor],
        substore_map: dict[XarrayStoragePartition, Reducer], /
    ) -> ReducerMapping[str]:
        """
        Extract per-generation variable coordinate bounds and time coordinate
        bounds, and pair the variable trajectories and time coordinates into
        per-generation time series.
        """
        # verify ordering of partitions within substore results
        generations = [p.generation for p in substore_map]
        assert generations == list(range(1, 1 + len(generations)))

        # perform aggregations across partitions
        results = ReducerMapping[str].reduce(
            cast(list[ReducerMapping[str]], list(substore_map.values()))
        ).extract()

        # prepare aggregations across substores
        return ReducerMapping[str]({
            # collect variable descriptors
            "variables": ReducerMapping[VariablePath]({
                var: Unique[CoordDescriptor]([dscr])
                for (var, dscr) in var_dscrs.items()
            }),
            "generations": ReducerMapping[int]({
                # compute separate statistics for each generation
                gen: ReducerMapping[str]({

                    # extract interval bounds for local time
                    "t_minmax": MinMax(
                        mins=(t := results[cfg.t_var][gen - 1])[[0]],
                        maxs=t[[-1]]
                    ),

                    # extract interval bounds for each variable
                    "y_minmax": ReducerMapping[VariablePath]({
                        var: MinMax(
                            mins=(y := res["y_minmax"])[0],
                            maxs=y[1]
                        )
                        for (var, res) in results.items()
                        if var != cfg.t_var
                    }),

                    # pair local time with smoothed trajectories
                    "trajs": ReducerMapping[VariablePath]({
                        var: Chain[tuple[NDArray, NDArray]]([
                            (t, res["moving_avg"][gen - 1])
                        ])
                        for (var, res) in results.items()
                        if var != cfg.t_var
                    })
                })
                for gen in generations
            })
        })

    @staticmethod
    async def reduce_partition(
        cfg: MovingAvgConfig, time_dscr: CoordDescriptor,
        time_coo: zarr.AsyncArray, time_var: zarr.AsyncArray,
        partition_map: dict[VariablePath, Reducer], /
    ) -> ReducerMapping[str]:
        """
        Convert simulation-global times into generation-local times, and pass
        through the inner reduced variables.
        """
        # compute local time within generation
        t = cast(NDArray, await time_var.get_orthogonal_selection(slice(None)))
        t -= t[0]

        # convert units
        assert time_dscr.unit == sec_unit
        t /= sec_per_min

        # aggregation purpose: Monte Carlo estimates over trajectories
        partition_time = {cfg.t_var: Chain[NDArray]([t])}

        # add to aggregation variables
        return ReducerMapping(partition_time | partition_map)

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    def select_partitions(
        _: MovingAvgConfig,
        substore: Substore, partitions: list[XarrayStoragePartition], /
    ) -> list[XarrayStoragePartition]:
        """
        Apply this pipeline to all generations.
        """
        return partitions

    @staticmethod
    async def select_time_coordinate(
        _: MovingAvgConfig, p: XarrayStoragePartition, attr: str,
        time_coo: zarr.AsyncArray, time_var: zarr.AsyncArray, /
    ) -> CoordDescriptor:
        """
        Apply this pipeline to the full time dimension within each generation.
        """
        return CoordDescriptor(dim_names=[], selector=slice(None), unit=attr)

    @staticmethod
    async def select_variable_coordinate(
        _: MovingAvgConfig, path: VariablePath, attr: Any,
        coo: zarr.AsyncArray, match_coo: str, /
    ) -> CoordDescriptor:
        """
        Select only variable dimensions whose coordinate labels match the string
        ``match_coo``.
        """
        data = cast(NDArray, await coo.get_orthogonal_selection(slice(None)))
        assert data.ndim == 1
        assert data.dtype.type == str_
        ix = flatnonzero(strings.find(data, match_coo) >= 0)
        return CoordDescriptor(dim_names=data[ix].tolist(), selector=ix,
                               unit=attr)

    # ~~~~~~~~~~~~~~~~~ #

    @staticmethod
    async def get_array_selection(
        _: MovingAvgConfig, path: VariablePath,
        time_ix: Selector, var_ix: Selector, var: zarr.AsyncArray, /
    ) -> NDArrayLikeOrScalar:
        """
        Retrieve the target variable at the Cartesian product of the time
        coordinate selection and the variable coordinate selection.
        """
        return await var.get_orthogonal_selection((time_ix, var_ix))

    @staticmethod
    async def reduce_variable(
        cfg: MovingAvgConfig, path: VariablePath, data: NDArrayLikeOrScalar, /
    ) -> ReducerMapping[str]:
        """
        Compute the variable bounds and the moving average within a generation.
        """
        assert cast(NDArray, data).ndim == 2
        return ReducerMapping[str]({

            # aggregation purpose: plotting boundaries
            "y_minmax": MinMax(
                mins=nanmin(data, axis=0),
                maxs=nanmax(data, axis=0)
            ),

            # aggregation purpose: Monte Carlo estimates over trajectories
            "moving_avg": Chain[NDArray]([
                move_mean(data, axis=0,
                          window=cfg.window, min_count=cfg.min_window)
            ])
        })


# ==============================================================================
# analysis plot
# ==============================================================================


@dataclass(slots=True)
class MovingAvgPlot(ZarrMapReducePlot[
    MovingAvgConfig, MovingAvgResult, MovingAvgPlotConfig
]):

    chart: None | altair.vegalite.v5.api.VConcatChart = None

    # ~~~~~~~~~~~~~~~~~ #

    @property
    def num_generations(self) -> int:
        return self.result.generations[-1]

    def num_dims(self, variable: str, /) -> int:
        return (self.plot_cfg.debug_num_dims
                if self.analysis_cfg.debug else
                self.result.dimensions[variable])

    # ~~~~~~~~~~~~~~~~~ #

    def render(self) -> None:
        """
        Calls: :py:meth:`.assemble`.
        """
        data_transformers.options["max_rows"] = self.plot_cfg.max_rows_per_panel
        self.chart = self.assemble()
        self.chart.save(self.analysis_cfg.figure_file)

    def assemble(self) -> altair.VConcatChart:
        """
        Assemble the full Vega-Altair plot specification, without rendering it.

        Called by: :py:meth:`.render`.

        Calls: :py:meth:`.assemble_grid`.
        """
        assert self.plot_data is not None
        assert self.chart is None
        return self.assemble_grid(*self.plot_data)

    def assemble_grid(
        self, vnt_keys: list[str],
        df_grid: dict[int, dict[VariablePath, dict[int, pandas.DataFrame]]], /
    ) -> altair.VConcatChart:
        """
        Called by: :py:meth:`.assemble`.

        Calls: :py:meth:`.assemble_panel`.
        """
        cfg = self.plot_cfg
        panel_times = [t[1] for t in self.result.t_minmax.values()]
        panel_width_rate = max(
            cfg.min_panel_width / min(panel_times),
            cfg.max_avg_panel_width * self.num_generations / sum(panel_times)
        )
        return vconcat(
            *(
                vconcat(
                    *(
                        hconcat(
                            *(
                                self.assemble_panel(
                                    vnt_keys,
                                    df_grid[gen][vbl][dim],
                                    vbl=vbl, gen=gen, dim=dim,
                                    left=(gen == 1),
                                    right=(gen == self.num_generations),
                                    top=(dim == 0),
                                    bot=(dim == self.num_dims(vbl) - 1),
                                    width_rate=panel_width_rate
                                )
                                for gen in self.result.generations
                            ),
                            spacing=cfg.grid_spacing
                        )
                        for dim in range(self.num_dims(vbl))
                    ),
                    title=Title(vbl, fontSize=cfg.title_fontsize),
                    spacing=cfg.grid_spacing
                )
                for vbl in self.result.variables
            ),
            resolve={"scale": {"color": "independent"}}
        )

    def assemble_panel(
        self, vnt_keys: list[str], df: pandas.DataFrame,
        *,
        vbl: str, gen: int, dim: int,
        left: bool, right: bool, top: bool, bot: bool,
        width_rate: float
    ) -> altair.LayerChart:
        """
        Called by: :py:meth:`.assemble_grid`.

        .. note::
          `VegaFusion`_ is *not* used here, since it does not support the
          bootstrapped confidence interval option for the `error band`_
          visualization.

        .. _VegaFusion: https://vegafusion.io/
        .. _error band: https://altair-viz.github.io/user_guide/marks/errorband.html#error-band-mark-properties
        """
        cfg = self.plot_cfg

        # chart & encoding spec
        inner_axis: dict[str, Any] = {
            "title": None, "labels": False, "domain": False, "ticks": False
        }
        base = Chart(df, **(
            {  # type: ignore[arg-type]
                "title": Title(
                    f"Generation {gen}",
                    fontSize=cfg.panel_fontsize, offset=cfg.panel_offset)
            }
            if top else
            {}
        )).properties(
            height=cfg.panel_height,
            width=int(width_rate * self.result.t_minmax[gen][1])
        ).encode(
            x=X("t:Q")
                .scale(domain=tuple(self.result.t_minmax[gen]),
                       nice=False)
                .axis(gridOpacity=cfg.grid_opacity, **(
                    {
                        "title": f"time {min_unit}",
                        "titleFontSize": cfg.axis_fontsize,
                        "labelFontSize": cfg.tick_fontsize,
                        "orient": "top" if top else "bottom", "zindex": 2
                    }
                    if (top or bot) else
                    inner_axis)),
            y=Y("y:Q")
                .scale(domain=self.result.y_minmax[vbl][dim],
                       zero=False, nice=False, **dict(cfg.y_rescale))
                .axis(gridOpacity=cfg.grid_opacity, **(
                    {
                        "title": wrap(f"{self.result.y_names[vbl][dim]} "
                                      f"{self.result.y_units[vbl]}",
                                      width=cfg.title_textwidth),
                        "titleFontSize": cfg.axis_fontsize,
                        "labelFontSize": cfg.tick_fontsize,
                        "format": cfg.tick_format,
                        "orient": "left" if left else "right", "zindex": 2,
                        "titleAlign": "left" if left else "right",
                        "titleAngle": 0,
                        "titleX": cfg.title_shift_x if left else -cfg.title_shift_x,
                        "titleY": cfg.title_shift_y,
                        "minExtent": cfg.y_label_width,
                        "maxExtent": cfg.y_label_width
                    }
                    if (left or right) else
                    inner_axis
                )),
            color=Color("variant:N", title="Variant", sort=vnt_keys)
                .legend(orient="left", offset=cfg.legend_offset,
                        titleFontSize=cfg.title_fontsize,
                        labelFontSize=cfg.panel_fontsize)
                .scale(scheme=ColorScheme(cfg.color_scheme)),
            tooltip=value(None))

        # data spec
        band = base.mark_errorband(extent=cfg.band_extent, borders=False)
        mean = base.mark_line(size=cfg.mean_line_size).encode(y="mean(y)")

        # panel spec
        return band + mean


# ==============================================================================
# entry point
# ==============================================================================


def main(args: Namespace | None = None) -> None:
    """
    Module entry point.

    Parses the CLI arguments, loads the :py:class:`.WorkflowConfig`, and
    constructs and executes the :py:class:`.MovingAvgPipeline`.
    """

    # receive driver configuration
    if args is None:
        parser = MovingAvgConfig.make_cli_parser(__doc__)
        args = parser.parse_args()

    # define analysis pipeline
    from configs import CONFIG_DIR_PATH
    sim_cfg = WorkflowConfig.load(Path(CONFIG_DIR_PATH) / args.config)
    analysis_cfg = MovingAvgConfig.from_cli_parser(args)
    plot_cfg = MovingAvgPlotConfig()

    # run driver
    analysis = MovingAvgPipeline(sim_cfg, analysis_cfg, plot_cfg)
    analysis.compute()


if __name__ == "__main__":
    main()
