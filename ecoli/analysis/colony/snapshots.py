import os
import math
import random
import itertools
import shutil
import concurrent.futures
from typing import Any

import imageio.v3 as iio
from duckdb import DuckDBPyConnection
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from mpl_toolkits.axes_grid1 import make_axes_locatable, anchored_artists

from vivarium.library.dict_utils import get_value_from_path
from vivarium.library.units import Quantity, units
from vivarium.plots.agents_multigen import plot_agents_multigen

DEFAULT_BOUNDS = [10, 10]
DEFAULT_HIGHLIGHT_COLOR = [0, 1, 1]
PLOT_WIDTH = 7

# constants
PI = math.pi

# colors for phylogeny initial agents
HUES = [hue / 360 for hue in np.linspace(0, 360, 30)]
DEFAULT_HUE = 45 / 360
DEFAULT_SV = [100.0 / 100.0, 70.0 / 100.0]
BASELINE_TAG_COLOR = [0, 0, 1]  # HSV
FLOURESCENT_SV = [0.75, 1.0]  # SV for fluorescent colors


class LineWidthData(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72.0 / self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data)) - trans((0, 0))) * ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def init_axes(
    fig,
    edge_length_x,
    edge_length_y,
    grid,
    row_idx,
    col_idx,
    time,
    molecule,
    ylabel_size=20,
    title_size=12,
):
    ax = fig.add_subplot(grid[row_idx, col_idx])
    if row_idx == 0:
        plot_title = "time: {:.4f} s".format(float(time))
        plt.title(plot_title, y=1.08, fontsize=title_size)
    if col_idx == 0:
        ax.set_ylabel(
            molecule,
            fontsize=ylabel_size,
            rotation="horizontal",
            horizontalalignment="right",
        )
    ax.set(xlim=[0, edge_length_x], ylim=[0, edge_length_y], aspect=1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return ax


def add_time_axis(
    fig, grid, n_rows, n_cols, n_snapshots, snapshot_times, time_unit="s"
):
    # Add time axis across subplots
    super_spec = matplotlib.gridspec.SubplotSpec(
        grid,
        (n_rows - 1) * n_cols,
        (n_rows - 1) * n_cols + n_snapshots - 1,
    )
    grid_params = grid.get_subplot_params()
    if n_snapshots > 1:
        time_per_snapshot = (snapshot_times[-1] - snapshot_times[0]) / (
            (n_snapshots - 1) * (grid_params.wspace + 1)
        )
    else:
        time_per_snapshot = 1  # Arbitrary
    super_ax = fig.add_subplot(  # type: ignore
        super_spec,
        xticks=snapshot_times,
        xlim=(
            snapshot_times[0] - time_per_snapshot / 2,
            snapshot_times[-1] + time_per_snapshot / 2,
        ),
        yticks=[],
    )
    super_ax.set_xlabel(f"Time ({time_unit})", labelpad=50)  # type: ignore
    super_ax.xaxis.set_tick_params(width=2, length=8)
    for spine_name in ("top", "right", "left"):
        super_ax.spines[spine_name].set_visible(False)
    super_ax.spines["bottom"].set_linewidth(2)


def mutate_color(baseline_hsv):
    mutation = 0.1
    new_hsv = [(n + np.random.uniform(-mutation, mutation)) for n in baseline_hsv]
    # wrap hue around
    new_hsv[0] = new_hsv[0] % 1
    # reflect saturation and value
    if new_hsv[1] > 1:
        new_hsv[1] = 2 - new_hsv[1]
    if new_hsv[2] > 1:
        new_hsv[2] = 2 - new_hsv[2]
    return new_hsv


def plot_agent(
    ax,
    data,
    color,
    agent_shape,
    membrane_width=0.1,
    membrane_color=[0, 0, 0],
    alpha=1,
):
    """Plot an agent

    Args:
        ax: The axes to draw on.
        data (dict): The agent data dictionary.
        color (list): HSV color of agent body.
        agent_shape (str): One of ``rectangle``, ``segment``, and ``circle``.
        membrane_width (float): Width of drawn agent boundary.
        membrane_color (list): RGB color of drawn agent boundary.
    """
    if not data or not data.get("boundary"):
        return
    x_center = data["boundary"]["location"][0]
    y_center = data["boundary"]["location"][1]

    # get color, convert to rgb. Strings are already RGB
    if isinstance(color, str):
        rgb = color
    else:
        rgb = hsv_to_rgb(color)

    if agent_shape == "rectangle":
        theta = (
            data["boundary"]["angle"] / PI * 180 + 90
        )  # rotate 90 degrees to match field
        length = data["boundary"]["length"]
        width = data["boundary"]["width"]

        # get bottom left position
        x_offset = width / 2
        y_offset = length / 2
        theta_rad = math.radians(theta)
        dx = x_offset * math.cos(theta_rad) - y_offset * math.sin(theta_rad)
        dy = x_offset * math.sin(theta_rad) + y_offset * math.cos(theta_rad)

        x = x_center - dx
        y = y_center - dy

        # Create a rectangle
        shape = patches.Rectangle(
            (x, y),
            width,
            length,
            angle=theta,
            linewidth=membrane_width,
            edgecolor=membrane_color,
            alpha=alpha,
            facecolor=rgb,
        )
        ax.add_patch(shape)

    elif agent_shape == "segment":
        theta = (
            data["boundary"]["angle"] / PI * 180 + 90
        )  # rotate 90 degrees to match field
        length = data["boundary"]["length"]
        width = data["boundary"]["width"]

        radius = width / 2

        # get the two ends
        length_offset = (length / 2) - radius
        theta_rad = math.radians(theta)
        dx = -length_offset * math.sin(theta_rad)
        dy = length_offset * math.cos(theta_rad)

        x1 = x_center - dx
        y1 = y_center - dy
        x2 = x_center + dx
        y2 = y_center + dy

        # segment plot
        membrane = LineWidthData(
            [x1, x2],
            [y1, y2],
            color=membrane_color,
            linewidth=width,
            alpha=alpha,
            solid_capstyle="round",
        )
        line = LineWidthData(
            [x1, x2],
            [y1, y2],
            color=rgb,
            alpha=alpha,
            linewidth=width - membrane_width,
            solid_capstyle="round",
        )
        ax.add_line(membrane)
        ax.add_line(line)

    elif agent_shape == "circle":
        diameter = data["boundary"]["diameter"]

        # get bottom left position
        radius = diameter / 2
        x = x_center - radius
        y = y_center - radius

        # Create a circle
        circle = patches.Circle(
            (x, y),
            radius,
            linewidth=membrane_width,
            edgecolor=membrane_color,
            alpha=alpha,
        )
        ax.add_patch(circle)


def plot_agents(
    ax,
    agents,
    agent_colors=None,
    agent_shape="segment",
    dead_color=None,
    membrane_width=0.1,
    membrane_color=[1, 1, 1],
    alpha=1,
    highlight_agent=None,
):
    """Plot agents.

    Args:
        ax: the axis for plot
        agents (dict): a mapping from agent ID to that agent's data,
            which should have keys ``location``, ``angle``, ``length``,
            and ``width``.
        agent_colors (dict): Mapping from agent ID to HSV color.
        dead_color (list): List of 3 floats that define HSV color to use
            for dead cells. Dead cells only get treated differently if
            this is set.
        membrane_width (float): Width of agent outline to draw.
        membrane_color (list): List of 3 floats that define the RGB
            color to use for agent outlines.
        alpha: Alpha value for agents.
        highlight_agent: Mapping of agent ID to `membrane_color` and
            `membrane_width`. Useful for highlighting specific agents,
            with rest using default `membrane_width` and `membrane_color`
    """

    if not agent_colors:
        agent_colors = dict()
    if not highlight_agent:
        highlight_agent = []
    for agent_id, agent_data in agents.items():
        color = agent_colors.get(agent_id, [DEFAULT_HUE] + DEFAULT_SV)
        if dead_color and agent_data.get("boundary"):
            if agent_data["boundary"].get("dead"):
                color = dead_color
        if agent_data:
            if agent_id in highlight_agent:
                agent_membrane_width = highlight_agent[agent_id]["membrane_width"]
                agent_membrane_color = highlight_agent[agent_id]["membrane_color"]
            else:
                agent_membrane_width = membrane_width
                agent_membrane_color = membrane_color
            plot_agent(
                ax,
                agent_data,
                color,
                agent_shape,
                agent_membrane_width,
                agent_membrane_color,
                alpha,
            )

    if len(agents) == 1:
        ax.set_title("1 agent", y=1.1)
    else:
        ax.set_title(f"{len(agents)} agents", y=1.1)


def color_phylogeny(ancestor_id, phylogeny, baseline_hsv, phylogeny_colors=None):
    """
    get colors for all descendants of the ancestor
    through recursive calls to each generation
    """
    if not phylogeny_colors:
        phylogeny_colors = {}
    phylogeny_colors.update({ancestor_id: baseline_hsv})
    daughter_ids = phylogeny.get(ancestor_id)
    if daughter_ids:
        for daughter_id in daughter_ids:
            daughter_color = mutate_color(baseline_hsv)
            color_phylogeny(daughter_id, phylogeny, daughter_color, phylogeny_colors)
    return phylogeny_colors


def get_phylogeny_colors_from_names(agent_ids):
    """Get agent colors using phlogeny saved in agent_ids
    This assumes the names use daughter_phylogeny_id() from meta_division
    """

    # make phylogeny with {mother_id: [daughter_1_id, daughter_2_id]}
    phylogeny = {agent_id: [] for agent_id in agent_ids}
    for agent1, agent2 in itertools.combinations(agent_ids, 2):
        if agent1 == agent2[0:-1]:
            phylogeny[agent1].append(agent2)
        elif agent2 == agent1[0:-1]:
            phylogeny[agent2].append(agent1)

    # get initial ancestors
    daughters = list(phylogeny.values())
    daughters = set([item for sublist in daughters for item in sublist])
    mothers = set(list(phylogeny.keys()))
    ancestors = list(mothers - daughters)

    # agent colors based on phylogeny
    agent_colors = {agent_id: [] for agent_id in agent_ids}
    for agent_id in ancestors:
        hue = random.choice(HUES)  # select random initial hue
        initial_color = [hue] + DEFAULT_SV
        agent_colors.update(color_phylogeny(agent_id, phylogeny, initial_color))

    return agent_colors


def format_snapshot_data(data):
    agents = {}
    fields = {}
    for time, time_data in data.items():
        agents[time] = time_data.get("agents", {})
        fields[time] = time_data.get("fields", {})
    return agents, fields


def get_field_range(
    fields,
    time_vec,
    include_fields=None,
    skip_fields=None,
    min_pct=0.95,
    max_pct=1,
):
    if not skip_fields:
        skip_fields = []
    field_range = {}
    if fields:
        if include_fields is None:
            field_ids = set(fields[time_vec[0]].keys())
        else:
            field_ids = set(include_fields)

        field_ids -= set(skip_fields)
        for field_id in field_ids:
            field_min = min(
                [
                    min(min(field_data[field_id])) * min_pct
                    for t, field_data in fields.items()
                ]
            )
            field_max = max(
                [
                    max(max(field_data[field_id])) * max_pct
                    for t, field_data in fields.items()
                ]
            )
            field_range[field_id] = [field_min, field_max]
    return field_range


def get_agent_ids(agents):
    agent_ids = set()
    for time, time_data in agents.items():
        current_agents = list(time_data.keys())
        agent_ids.update(current_agents)
    return list(agent_ids)


def get_agent_colors(
    agents,
    phylogeny_names=True,
    agent_fill_color=None,
):
    agent_ids = get_agent_ids(agents)
    agent_colors = {}
    if agents:
        # set agent colors
        if agent_fill_color:
            agent_colors = {agent_id: agent_fill_color for agent_id in agent_ids}
        elif phylogeny_names:
            agent_colors = get_phylogeny_colors_from_names(agent_ids)
        else:
            agent_colors = {}
            for agent_id in agent_ids:
                hue = random.choice(HUES)
                color = [hue] + DEFAULT_SV
                agent_colors[agent_id] = color
    return agent_colors


def plot_snapshots(
    bounds,
    agents=None,
    fields=None,
    n_snapshots=5,
    snapshot_times=None,
    agent_fill_color=None,
    agent_colors=None,
    phylogeny_names=True,
    skip_fields=None,
    include_fields=None,
    out_dir=None,
    filename="snapshots",
    min_pct=0.95,
    max_pct=1,
    **kwargs,
):
    """Plot snapshots of the simulation over time

    The snapshots depict the agents and environmental molecule
    concentrations.

    Arguments:
        data (dict): A dictionary with the following keys:

            * **bounds** (:py:class:`tuple`): The dimensions of the
              environment.
            * **agents** (:py:class:`dict`): A mapping from times to
              dictionaries of agent data at that timepoint. Agent data
              dictionaries should have the same form as the hierarchy
              tree rooted at ``agents``.
            * **fields** (:py:class:`dict`): A mapping from times to
              dictionaries of environmental field data at that
              timepoint.  Field data dictionaries should have the same
              form as the hierarchy tree rooted at ``fields``.
            * **n_snapshots** (:py:class:`int`): Number of snapshots to
              show per row (i.e. for each molecule). Defaults to 6.
            * **phylogeny_names** (:py:class:`bool`): This selects agent
              colors based on phylogenies seved in their names using
              meta_division.py daughter_phylogeny_id()
            * **skip_fields** (:py:class:`Iterable`): Keys of fields to
              exclude from the plot. This takes priority over
              ``include_fields``.
            * **include_fields** (:py:class:`Iterable`): Keys of fields
              to plot.
            * **snapshot_times** (:py:class:`Iterable`): Times to plot
              snapshots for. Defaults to None, in which case n_snapshots
              is used.
            * **out_dir** (:py:class:`str`): Output directory, which is
              ``out`` by default.
            * **filename** (:py:class:`str`): Base name of output file.
              ``snapshots`` by default.
            * **min_pct** (:py:class:`float`): Percent of minimum value
              to use as minimum in colorbar (1 = 100%)
            * **max_pct** (:py:class:`float`): Percent of maximum value
              to use as maximum in colorbar (1 = 100%)
    """
    if not agents:
        agents = {}
    if not fields:
        fields = {}
    if not skip_fields:
        skip_fields = []
    # Strip units from bounds if present.
    if isinstance(bounds[0], Quantity):
        bounds = tuple(bound.to(units.um).magnitude for bound in bounds)
    # time steps that will be used
    if agents and fields:
        assert set(list(agents.keys())) == set(list(fields.keys())), (
            "agent and field times are different"
        )
        time_vec = list(agents.keys())
    elif agents:
        time_vec = list(agents.keys())
    elif fields:
        time_vec = list(fields.keys())
        agents = {t: {} for t in time_vec}
    else:
        raise Exception("No agents or field data")

    # get fields id and range
    field_range = get_field_range(
        fields, time_vec, include_fields, skip_fields, min_pct, max_pct
    )

    # get agent ids
    if not agent_colors:
        agent_colors = get_agent_colors(agents, phylogeny_names, agent_fill_color)

    # get time data
    if snapshot_times:
        n_snapshots = len(snapshot_times)
        time_indices = [time_vec.index(time) for time in snapshot_times]
    else:
        time_indices = np.round(np.linspace(0, len(time_vec) - 1, n_snapshots)).astype(
            int
        )
        snapshot_times = [time_vec[i] for i in time_indices]

    return make_snapshots_figure(
        agents=agents,
        agent_colors=agent_colors,
        fields=fields,
        field_range=field_range,
        n_snapshots=n_snapshots,
        time_indices=time_indices,
        snapshot_times=snapshot_times,
        bounds=bounds,
        out_dir=out_dir,
        filename=filename,
        **kwargs,
    )


def make_snapshots_figure(
    agents,
    fields,
    bounds,
    n_snapshots,
    time_indices,
    snapshot_times,
    time_unit="s",
    plot_width=12,
    field_range=None,
    agent_colors=None,
    dead_color=[0, 0, 0],
    membrane_width=0.1,
    membrane_color=[1, 1, 1],
    default_font_size=36,
    field_label_size=32,
    agent_shape="segment",
    agent_alpha=1,
    colorbar_decimals=3,
    show_timeline=True,
    scale_bar_length=1,
    scale_bar_color="black",
    xlim=None,
    ylim=None,
    min_color="white",
    max_color="gray",
    out_dir=None,
    filename="snapshots",
    figsize=None,
):
    """
    Args:
        bounds (:py:class:`tuple`): The dimensions of the
            environment.
        field_label_size (:py:class:`float`): Font size of the
            field label.
        dead_color (:py:class:`list` of 3 :py:class:`float`):
            Color for dead cells in HSV. Defaults to [0, 0, 0], which
            is black.
        default_font_size (:py:class:`float`): Font size for
            titles and axis labels.
        agent_shape (:py:class:`str`): the shape of the agents.
            select from ``rectangle``, ``segment``
        agent_alpha (:py:class:`float`): Alpha for agent
            plots.
        colorbar_decimals (:py:class:`int`): number of decimals in
            colorbar.
        scale_bar_length (:py:class:`float`): Length of scale
            bar.  Defaults to 1 (in units of micrometers). If 0, no
            bar plotted.
        scale_bar_color (:py:class:`str`): Color of scale bar
        xlim (:py:class:`tuple` of :py:class:`float`): Tuple
            of lower and upper x-axis limits.
        ylim (:py:class:`tuple` of :py:class:`float`): Tuple
            of lower and upper y-axis limits.
        min_color (any valid matplotlib color): Color for
            minimum field values.
        max_color (any valid matplotlib color): Color for
            maximum field values.
        out_dir (:py:class:`str`): Output directory, which is
            ``out`` by default.
        filename (:py:class:`str`): Base name of output file.
            ``snapshots`` by default.
        figsize (:py:class:`tuple`): Dimensions of figure in
            inches (takes precedence over `plot_width`)
    """
    edge_length_x = bounds[0]
    edge_length_y = bounds[1]

    # make the figure
    field_ids = list(field_range.keys())
    n_rows = max(len(field_ids), 1)
    n_cols = n_snapshots + 1  # one column for the colorbar
    if not figsize:
        figsize = (plot_width * n_cols, plot_width * n_rows)
    max_dpi = min([2**16 // dim for dim in figsize]) - 1
    fig = plt.figure(figsize=figsize, dpi=min(max_dpi, 100))
    grid = plt.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2)
    original_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": default_font_size})

    # Add time axis across subplots
    if show_timeline:
        add_time_axis(fig, grid, n_rows, n_cols, n_snapshots, snapshot_times, time_unit)

    # Make the colormap
    min_rgb = matplotlib.colors.to_rgb(min_color)
    max_rgb = matplotlib.colors.to_rgb(max_color)
    colors_dict = {
        "red": [
            [0, min_rgb[0], min_rgb[0]],
            [1, max_rgb[0], max_rgb[0]],
        ],
        "green": [
            [0, min_rgb[1], min_rgb[1]],
            [1, max_rgb[1], max_rgb[1]],
        ],
        "blue": [
            [0, min_rgb[2], min_rgb[2]],
            [1, max_rgb[2], max_rgb[2]],
        ],
    }
    cmap = matplotlib.colors.LinearSegmentedColormap(
        "field", segmentdata=colors_dict, N=512
    )

    stats = {
        "agents": {},
    }
    if field_ids:
        stats["fields"] = {field_id: {} for field_id in field_ids}
    # plot snapshot data in each subsequent column
    for col_idx, (time_idx, time) in enumerate(zip(time_indices, snapshot_times)):
        stats["agents"][time] = len(agents[time])
        if field_ids:
            for row_idx, field_id in enumerate(field_ids):
                ax = init_axes(
                    fig,
                    edge_length_x,
                    edge_length_y,
                    grid,
                    row_idx,
                    col_idx,
                    time,
                    field_id,
                    field_label_size,
                )
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                )

                # transpose field to align with agents
                field = np.transpose(np.array(fields[time][field_id]))
                vmin, vmax = field_range[field_id]
                q1, q2, q3 = np.percentile(field, [25, 50, 75])
                stats["fields"][field_id][time] = (field.min(), q1, q2, q3, field.max())
                im = plt.imshow(
                    field.tolist(),
                    origin="lower",
                    extent=[0, edge_length_x, 0, edge_length_y],
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                if agents:
                    agents_now = agents[time]
                    plot_agents(
                        ax,
                        agents_now,
                        agent_colors,
                        agent_shape=agent_shape,
                        dead_color=dead_color,
                        membrane_width=membrane_width,
                        membrane_color=membrane_color,
                        alpha=agent_alpha,
                    )
                if xlim:
                    ax.set_xlim(*xlim)
                if ylim:
                    ax.set_ylim(*ylim)

                # colorbar in new column after final snapshot
                if col_idx == n_snapshots - 1:
                    cbar_col = col_idx + 1
                    ax = fig.add_subplot(grid[row_idx, cbar_col])
                    if row_idx == 0:
                        ax.set_title("Concentration\n(mM)", y=1.08)
                    ax.axis("off")
                    if vmin == vmax:
                        continue
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("left", size="5%", pad=0.0)
                    fig.colorbar(im, cax=cax, format=f"%.{colorbar_decimals}f")
                    ax.axis("off")
                # Scale bar in first snapshot of each row
                if col_idx == 0 and scale_bar_length:
                    scale_bar = anchored_artists.AnchoredSizeBar(
                        ax.transData,
                        scale_bar_length,
                        f"{scale_bar_length} μm",
                        "lower left",
                        color=scale_bar_color,
                        frameon=False,
                        sep=scale_bar_length,
                        size_vertical=scale_bar_length / 20,
                    )
                    ax.add_artist(scale_bar)
        else:
            row_idx = 0
            ax = init_axes(fig, bounds[0], bounds[1], grid, row_idx, col_idx, time, "")

            if agents:
                agents_now = agents[time]
                plot_agents(
                    ax,
                    agents_now,
                    agent_colors,
                    agent_shape=agent_shape,
                    dead_color=dead_color,
                    membrane_width=membrane_width,
                    membrane_color=membrane_color,
                    alpha=agent_alpha,
                )
            if xlim:
                ax.set_xlim(*xlim)
            if ylim:
                ax.set_ylim(*ylim)
            # Scale bar in first snapshot of each row
            if col_idx == 0 and scale_bar_length:
                scale_bar = anchored_artists.AnchoredSizeBar(
                    ax.transData,
                    scale_bar_length,
                    f"{scale_bar_length} μm",
                    "lower left",
                    color=scale_bar_color,
                    frameon=False,
                    sep=scale_bar_length,
                    size_vertical=scale_bar_length / 20,
                )
                ax.add_artist(scale_bar)

    plt.rcParams.update({"font.size": original_fontsize})
    if out_dir:
        fig_path = os.path.join(out_dir, filename)
        fig.subplots_adjust(wspace=0.7, hspace=0.1)
        fig.savefig(fig_path, bbox_inches="tight")
    return fig


def plot_tags(
    data,
    bounds,
    snapshot_times=None,
    n_snapshots=5,
    **kwargs,
):
    agents, fields = format_snapshot_data(data)
    time_vec = list(agents.keys())

    # get time data
    if snapshot_times:
        n_snapshots = len(snapshot_times)
        time_indices = [time_vec.index(time) for time in snapshot_times]
    else:
        time_indices = np.round(np.linspace(0, len(time_vec) - 1, n_snapshots)).astype(
            int
        )
        snapshot_times = [time_vec[i] for i in time_indices]

    return make_tags_figure(
        agents=agents,
        bounds=bounds,
        n_snapshots=n_snapshots,
        time_indices=time_indices,
        snapshot_times=snapshot_times,
        **kwargs,
    )


def get_tag_ranges(
    agents, tagged_molecules, time_indices, convert_to_concs, tag_colors
):
    # get tag ids and range
    tag_ranges = {}
    for time_idx, (time, time_data) in enumerate(agents.items()):
        if time_idx in time_indices:
            for agent_id, agent_data in time_data.items():
                volume = agent_data.get("boundary", {}).get("volume", 0)
                for tag_id in tagged_molecules:
                    level = get_value_from_path(agent_data, tag_id)
                    if level is None:
                        continue
                    if convert_to_concs:
                        level = level / volume if volume else 0
                    if tag_id in tag_ranges:
                        tag_ranges[tag_id] = [
                            min(tag_ranges[tag_id][0], level),
                            max(tag_ranges[tag_id][1], level),
                        ]
                    else:
                        # add new tag
                        tag_ranges[tag_id] = [level, level]

                        # select random initial hue
                        if tag_id not in tag_colors:
                            hue = random.choice(HUES)
                            tag_color = [hue] + FLOURESCENT_SV
                            tag_colors[tag_id] = tag_color
    return tag_ranges, tag_colors


def make_tags_figure(
    agents,
    bounds,
    time_indices,
    snapshot_times,
    tag_ranges=None,
    tag_colors=None,
    min_color="black",
    agent_colors=None,
    n_snapshots=6,
    scale_bar_length=1,
    scale_bar_color="black",
    show_timeline=True,
    show_colorbar=True,
    time_unit="s",
    tagged_molecules=None,
    out_dir=False,
    filename="tags",
    agent_shape="segment",
    background_color="black",
    colorbar_decimals=1,
    tag_path_name_map=None,
    tag_label_size=20,
    plot_width=12,
    default_font_size=36,
    convert_to_concs=True,
    membrane_width=0.1,
    membrane_color=None,
    xlim=None,
    ylim=None,
    figsize=None,
    highlight_agent=None,
):
    """Plot snapshots of the simulation over time

    The snapshots depict the agents and the levels of tagged molecules
    in each agent by agent color intensity.

    Arguments:
        data (dict): A dictionary with the following keys:

            * **agents** (:py:class:`dict`): A mapping from times to
              dictionaries of agent data at that timepoint. Agent data
              dictionaries should have the same form as the hierarchy
              tree rooted at ``agents``.
            * **n_snapshots** (:py:class:`int`): Number of snapshots to
              show per row (i.e. for each molecule). Defaults to 6.
            * **out_dir** (:py:class:`str`): Output directory, which is
              ``out`` by default.
            * **filename** (:py:class:`str`): Base name of output file.
              ``tags`` by default.
            * **tagged_molecules** (:py:class:`typing.Iterable`): The
              tagged molecules whose concentrations will be indicated by
              agent color. Each molecule should be specified as a
              :py:class:`tuple` of the path in the agent compartment
              to where the molecule's count can be found, with the last
              value being the molecule's count variable.
            * **convert_to_concs** (:py:class:`bool`): if True, convert counts
              to concentrations.
            * **background_color** (:py:class:`str`): use matplotlib colors,
              ``black`` by default
            * **colorbar_decimals** (:py:class:`int`): number of decimals in
              colorbar.
            * **tag_label_size** (:py:class:`float`): The font size for
              the tag name label
            * **default_font_size** (:py:class:`float`): Font size for
              titles and axis labels.
            * **membrane_width** (:py:class:`float`): Width to use for
                drawing agent edges.
            * **membrane_color** (:py:class:`list`): RGB color to use
                for drawing agent edges.
            * **tag_colors** (:py:class:`dict`): Mapping from tag ID to
                the HSV color to use for that tag as a list. Alternatively,
                each tag ID is mapped to a dictionary containing the `cmp`
                and `norm` keys with the :py:class:`matplotlib.colors.Colormap`
                and the :py:class:`matplotlib.colors.Normalize` instance to use
                for that tag. Using dictionaries will override ``min_color``
            * **figsize** (:py:class:`tuple`): Dimensions of figure in
                inches (takes precedence over `plot_width`)
            * **highlight_agent** (:py:class:`dict`): Mapping of agent IDs to
                `membrane_color` and `membrane_width`. Useful for highlighting
                specific agents, with rest using default color / width
    """

    membrane_color = membrane_color or [1, 1, 1]
    agent_colors = agent_colors or {}
    tag_colors = tag_colors or {}
    tag_path_name_map = tag_path_name_map or {}
    tagged_molecules = tagged_molecules or []
    if tagged_molecules == []:
        raise ValueError("At least one molecule must be tagged.")
    if not tag_ranges:
        tag_ranges, tag_colors = get_tag_ranges(
            agents, tagged_molecules, time_indices, convert_to_concs, tag_colors
        )

    # get data
    edge_length_x, edge_length_y = bounds

    # make the figure
    n_rows = len(tagged_molecules)
    n_cols = n_snapshots + 1  # one column for the colorbar
    if not figsize:
        figsize = (plot_width * n_cols, plot_width * n_rows)
    max_dpi = min([2**16 // dim for dim in figsize]) - 1
    fig = plt.figure(figsize=figsize, dpi=min(max_dpi, 100))
    grid = plt.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2)
    original_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": default_font_size})

    # Add time axis across subplots
    if show_timeline:
        add_time_axis(fig, grid, n_rows, n_cols, n_snapshots, snapshot_times, time_unit)

    # plot tags
    for row_idx, tag_id in enumerate(tag_ranges.keys()):
        for col_idx, (time_idx, time) in enumerate(zip(time_indices, snapshot_times)):
            tag_name = tag_path_name_map.get(tag_id, tag_id)
            ax = init_axes(
                fig,
                edge_length_x,
                edge_length_y,
                grid,
                row_idx,
                col_idx,
                time,
                tag_name,
                tag_label_size,
                title_size=default_font_size,
            )
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
            )
            ax.set_facecolor(background_color)

            # update agent colors based on tag_level
            min_tag, max_tag = tag_ranges[tag_id]
            agent_tag_colors = {}
            if isinstance(tag_colors[tag_id], dict):
                cmap = tag_colors[tag_id]["cmp"]
                norm = tag_colors[tag_id]["norm"]
                mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            else:
                tag_h, tag_s, tag_v = tag_colors[tag_id]
                tag_color_rgb = np.array(
                    matplotlib.colors.hsv_to_rgb([tag_h, tag_s, tag_v])
                )
                min_color = np.array(matplotlib.colors.to_rgb(min_color))
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    name=f"row_{row_idx}", colors=[min_color, tag_color_rgb]
                )
                norm = matplotlib.colors.Normalize(vmin=min_tag, vmax=max_tag)
                mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            for agent_id, agent_data in agents[time].items():
                # get current tag concentration, and determine color
                level = get_value_from_path(agent_data, tag_id)
                if convert_to_concs:
                    volume = agent_data.get("boundary", {}).get("volume", 0)
                    level = level / volume if volume else 0
                agent_tag_colors[agent_id] = rgb_to_hsv(mappable.to_rgba(level)[:3])

            agent_tag_colors.update(agent_colors)
            plot_agents(
                ax,
                agents[time],
                agent_tag_colors,
                agent_shape,
                None,
                membrane_width,
                membrane_color,
                highlight_agent=highlight_agent,
            )
            if xlim:
                ax.set_xlim(*xlim)
            if ylim:
                ax.set_ylim(*ylim)

            # Scale bar in first snapshot of each row
            if col_idx == 0 and scale_bar_length:
                scale_bar = anchored_artists.AnchoredSizeBar(
                    ax.transData,
                    scale_bar_length,
                    f"{scale_bar_length} μm",
                    "lower left",
                    color=scale_bar_color,
                    frameon=False,
                    sep=scale_bar_length,
                    size_vertical=scale_bar_length / 20,
                )
                ax.add_artist(scale_bar)

            # colorbar in new column after final snapshot
            if col_idx == n_snapshots - 1 and show_colorbar:
                cbar_col = col_idx + 1
                ax = fig.add_subplot(grid[row_idx, cbar_col])
                if row_idx == 0:
                    if convert_to_concs:
                        ax.set_title("Concentration\n(counts/fL)", y=1.08)
                ax.axis("off")
                if min_tag == max_tag:
                    continue
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("left", size="5%", pad=0.0)
                fig.colorbar(mappable, cax=cax, format=f"%.{colorbar_decimals}f")

    plt.rcParams.update({"font.size": original_fontsize})
    if out_dir:
        fig_path = os.path.join(out_dir, filename)
        fig.subplots_adjust(wspace=0.7, hspace=0.1)
        fig.savefig(fig_path, bbox_inches="tight")
    return fig


def save_snapshot_figure(data_at_time, kwargs):
    time = data_at_time[0]
    fig_path = os.path.join(kwargs.pop("images_dir", ""), f"img{time}.jpg")
    multibody_agents, multibody_fields = format_snapshot_data({time: data_at_time[1]})
    fig = make_snapshots_figure(
        agents=multibody_agents,
        fields=multibody_fields,
        n_snapshots=1,
        time_indices=[time],
        snapshot_times=[time],
        plot_width=PLOT_WIDTH,
        scale_bar_length=0,
        **kwargs,
    )
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def save_tags_figure(data_at_time, kwargs):
    time = data_at_time[0]
    fig_path = os.path.join(kwargs.pop("images_dir", ""), f"img{time}.jpg")
    agents = {time: data_at_time[1].get("agents", {})}
    fig = make_tags_figure(
        time_indices=[time],
        snapshot_times=[time],
        agents=agents,
        n_snapshots=1,
        plot_width=PLOT_WIDTH,
        **kwargs,
    )
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()
    return fig_path


def save_timeseries_figure(t_index, kwargs):
    time_vec = list(kwargs["data"].keys())

    plot_settings = {
        "column_width": 6,
        "row_height": 2,
        "stack_column": True,
        "tick_label_size": 10,
        "linewidth": 2,
        "title_size": 10,
    }

    if kwargs["show_timeseries"]:
        plot_settings.update({"include_paths": kwargs["show_timeseries"]})

    # remove agents not included in highlight_agents
    if kwargs["highlight_agents"]:
        for time, state in kwargs["data"].items():
            agents = state["agents"]
            for agent_id, agent_state in agents.items():
                if agent_id not in kwargs["highlight_agents"]:
                    del kwargs["data"][time]["agents"][agent_id]
        agent_colors = {
            agent_id: kwargs["highlight_color"]
            for agent_id in kwargs["highlight_agents"]
        }
        plot_settings.update({"agent_colors": agent_colors})

    fig_path = os.path.join(kwargs["images_dir"], f"timeseries{t_index}.jpg")
    time_indices = np.array(range(0, t_index + 1))
    current_data = {
        time_vec[index]: kwargs["data"][time_vec[index]] for index in time_indices
    }
    fig = plot_agents_multigen(current_data, dict(plot_settings))
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return fig_path


def video_from_images(img_paths, out_file, fps=15):
    """
    Create a video from a list of image paths using imageio.

    Args:
        img_paths (list): List of file paths to images.
        out_file (str): Output file path for the video.
        fps (int): Frames per second for the video.
    """
    # Read all images and determine the smallest dimensions
    images = [iio.imread(img_file) for img_file in img_paths]
    min_height = min(img.shape[0] for img in images)
    min_width = min(img.shape[1] for img in images)

    # Crop all images to the smallest dimensions
    cropped_images = [img[:min_height, :min_width] for img in images]

    # Write the video
    iio.imwrite(out_file, cropped_images, fps=fps, plugin="ffmpeg")


def make_video(
    data,
    bounds,
    plot_type="fields",
    step=1,
    highlight_agents=None,
    show_timeseries=None,
    highlight_color=DEFAULT_HIGHLIGHT_COLOR,
    out_dir="out",
    filename="snapshot_vid",
    cpus=1,
    **kwargs,
):
    """Make a video with snapshots across time

    Args:
        plot_type: (str) select either 'fields' or 'tags'. 'fields' is the default
    """
    # Remove last timestep since data may be empty
    data = dict(list(data.items())[:-1])
    highlight_agents = highlight_agents or []
    show_timeseries = show_timeseries or []

    # Strip units from bounds if present.
    if isinstance(bounds[0], Quantity):
        bounds = tuple(bound.to(units.um).magnitude for bound in bounds)

    # make images directory, remove if existing
    out_file = os.path.join(out_dir, f"{filename}.mp4")
    out_file2 = os.path.join(out_dir, f"{filename}_timeseries.mp4")
    images_dir = os.path.join(out_dir, f"_images_{plot_type}")
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)

    agent_colors = {}
    if highlight_agents:
        agent_colors = {agent_id: highlight_color for agent_id in highlight_agents}

    # get the single snapshots function
    multibody_agents, multibody_fields = format_snapshot_data(data)
    time_vec = list(multibody_agents.keys())
    if plot_type == "fields":
        multibody_field_range = get_field_range(multibody_fields, time_vec)
        multibody_agent_colors = get_agent_colors(multibody_agents)
        multibody_agent_colors.update(agent_colors)

        do_plot = save_snapshot_figure
        plot_kwargs = {
            "multibody_agent_colors": multibody_agent_colors,
            "multibody_field_range": multibody_field_range,
            "images_dir": images_dir,
            "bounds": bounds,
            **kwargs,
        }

    elif plot_type == "tags":
        time_indices = np.array(range(0, len(time_vec)))
        tag_ranges, tag_colors = get_tag_ranges(
            agents=multibody_agents,
            tagged_molecules=kwargs.get("tagged_molecules", None),
            time_indices=time_indices,
            convert_to_concs=kwargs.get("convert_to_concs", False),
            tag_colors=kwargs.pop("tag_colors", {}),
        )

        do_plot = save_tags_figure
        plot_kwargs = {
            "tag_ranges": tag_ranges,
            "tag_colors": tag_colors,
            "images_dir": images_dir,
            "agent_colors": agent_colors,
            "bounds": bounds,
            **kwargs,
        }

    # Only plot data for every `step` timepoints
    if step != 1:
        filtered_data = {}
        time_counter = 0
        for timepoint in time_vec:
            if time_counter % step == 0:
                filtered_data[timepoint] = data[timepoint]
            time_counter += 1
        data = filtered_data

    with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
        img_paths = list(
            tqdm(
                executor.map(do_plot, data.items(), itertools.repeat(plot_kwargs)),
                total=len(data),
            )
        )

    img_paths_2 = []
    if show_timeseries:
        plot_kwargs = {
            "show_timeseries": show_timeseries,
            "highlight_agents": highlight_agents,
            "highlight_color": highlight_color,
            "images_dir": images_dir,
            "data": data,
            **kwargs,
        }
        time_indices = list(range(0, len(time_vec)))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            img_paths_2 = list(
                tqdm(
                    executor.map(
                        save_timeseries_figure,
                        time_indices,
                        itertools.repeat(plot_kwargs),
                    ),
                    total=len(time_indices),
                )
            )

    # make the video
    video_from_images(img_paths, out_file)
    video_from_images(img_paths_2, out_file2)

    # delete image folder
    shutil.rmtree(images_dir)


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_name: str,
):
    # TODO: Write analysis script using DuckDB
    raise NotImplementedError("Still need to write analysis script using DuckDB!")
