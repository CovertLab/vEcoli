"""
Doubling time per generation across all seeds of one experiment and variant, to
show whether doubling time drifts over generations.

Top panel: one line per seed, with a red x where a lineage ended in a cell that
failed and a purple square where a cell reached ``max_duration``. Run this after
the workflow finishes, because a cell that is still running also has no success
marker.

Middle panel: mean and standard deviation across seeds for each generation.

Bottom panel: histogram of doubling times after skipping initial generations.
Failed cells are excluded, except cells that reached ``max_duration``. Those are
counted at their doubling time even if ``fail_at_max_duration`` marked them as
failed, because they were still alive at the time limit.

All panels mark the variant's target doubling time from its sim_data.

Optional params:
    skip_n_gens: leave this many initial generations out of the histogram and of
        the printed and titled average. All generations are still shown in the
        first two panels. Default: 2.
    max_doubling_time: largest doubling time shown in the first two panels, in
        minutes. Cells above this are drawn as triangles at the top of the first
        panel. Default: fits the data, up to 3 x target.
"""

import os
import pickle
from typing import Any, cast

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ecoli.library.parquet_emitter import read_stacked_columns
from wholecell.utils import units

DEATH_COLOR = "#d03030"
MAX_DURATION_COLOR = "#7b2fbe"
MEAN_COLOR = "#2a78d6"
TARGET_COLOR = "#52514e"
GRID_COLOR = "#e6e5e1"


def get_target_doubling_time(sim_data_dict: dict[str, dict[int, str]]) -> float | None:
    """Return the variant's target doubling time in minutes, if available."""
    for variants in sim_data_dict.values():
        for sim_data_path in variants.values():
            with open(sim_data_path, "rb") as f:
                sim_data = pickle.load(f)
            return float(sim_data.doubling_time.asNumber(units.min))
    return None


def get_max_duration_minutes(conn: DuckDBPyConnection, config_sql: str) -> float | None:
    """
    Return the shortest doubling time in minutes that counts as reaching
    ``max_duration``, allowing for the final time step not being emitted.
    """
    row = conn.sql(
        f"SELECT max(max_duration), max(time_step) FROM ({config_sql})"
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return (row[0] - 2 * (row[1] or 0)) / 60


def load_doubling_times(
    conn: DuckDBPyConnection, history_sql: str, success_sql: str
) -> pl.DataFrame:
    """
    Return one row per cell with its doubling time in minutes and whether it
    has a success marker.
    """
    times_sql = cast(
        str, read_stacked_columns(history_sql, ["time"], order_results=False)
    )
    return conn.sql(f"""
        WITH cells AS (
            SELECT experiment_id, variant, lineage_seed, generation, agent_id,
                (max(time) - min(time)) / 60 AS doubling_time
            FROM ({times_sql})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        ),
        succeeded AS (
            SELECT DISTINCT experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({success_sql})
        )
        SELECT cells.lineage_seed, cells.generation, cells.agent_id,
            cells.doubling_time, succeeded.generation IS NOT NULL AS success
        FROM cells
        LEFT JOIN succeeded
            ON cells.experiment_id = succeeded.experiment_id
            AND cells.variant = succeeded.variant
            AND cells.lineage_seed = succeeded.lineage_seed
            AND cells.generation = succeeded.generation
            AND cells.agent_id = succeeded.agent_id
        ORDER BY cells.lineage_seed, cells.generation, cells.agent_id
    """).pl()


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """Plot doubling time trajectories for each seed and their mean and SD."""
    cells = load_doubling_times(conn, history_sql, success_sql)
    if cells.is_empty():
        raise ValueError("No cells found in the data.")

    completed = cells.filter(pl.col("success"))
    if completed.is_empty():
        raise ValueError("No successfully divided cells found in the data.")

    # Cells of one seed and generation are averaged if single_daughters is off
    per_seed = (
        completed.group_by(["lineage_seed", "generation"])
        .agg(pl.col("doubling_time").mean())
        .sort(["lineage_seed", "generation"])
    )
    per_generation = (
        per_seed.group_by("generation")
        .agg(
            pl.col("doubling_time").mean().alias("mean"),
            pl.col("doubling_time").std().fill_null(0).alias("std"),
            pl.len().alias("n"),
        )
        .sort("generation")
    )

    skip_n_gens = params.get("skip_n_gens", 2)
    first_generation = cast(int, per_seed["generation"].min())
    averaged = per_seed.filter(pl.col("generation") >= first_generation + skip_n_gens)
    summary = (
        f"{averaged['doubling_time'].mean():.1f} ± "
        f"{averaged['doubling_time'].std():.1f} min after skipping "
        f"{skip_n_gens} generations (n={averaged.height})"
        if averaged.height > 1
        else "too few cells for an average"
    )

    target = get_target_doubling_time(sim_data_dict)
    observed_min = cast(float, per_seed["doubling_time"].min())
    observed_max = cast(float, per_seed["doubling_time"].max())
    y_max = params.get("max_doubling_time")
    if y_max is None:
        y_max = 1.05 * max(observed_max, target or 0)
        if target:
            y_max = min(y_max, 3 * target)
    y_min = 0.9 * min(observed_min, target or observed_min)

    seeds = per_seed["lineage_seed"].unique().sort().to_list()
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(10, 11.5))
    grid = fig.add_gridspec(3, 1)
    ax_seeds = fig.add_subplot(grid[0])
    ax_mean = fig.add_subplot(grid[1], sharex=ax_seeds)
    ax_hist = fig.add_subplot(grid[2])

    for i, seed in enumerate(seeds):
        seed_data = per_seed.filter(pl.col("lineage_seed") == seed)
        generations = seed_data["generation"].to_numpy()
        doubling_times = seed_data["doubling_time"].to_numpy()
        clipped = np.minimum(doubling_times, y_max)
        color = cmap(i / max(1, len(seeds) - 1))
        ax_seeds.plot(
            generations,
            clipped,
            color=color,
            linewidth=0.9,
            alpha=0.65,
            marker=".",
            markersize=3,
        )
        above = doubling_times > y_max
        ax_seeds.plot(
            generations[above], clipped[above], "^", color=color, markersize=7
        )

    max_duration = get_max_duration_minutes(conn, config_sql)
    reached_limit = (
        pl.col("doubling_time") >= max_duration
        if max_duration is not None
        else pl.lit(False)
    )
    at_limit = cells.filter(~pl.col("success") & reached_limit)
    failed = cells.filter(~pl.col("success") & ~reached_limit)

    # Mark each cell at the time limit at its own doubling time, connected to
    # the last completed generation of its seed
    for row in at_limit.iter_rows(named=True):
        earlier = per_seed.filter(
            (pl.col("lineage_seed") == row["lineage_seed"])
            & (pl.col("generation") < row["generation"])
        )
        if earlier.height and row["lineage_seed"] in seeds:
            seed_index = seeds.index(row["lineage_seed"])
            ax_seeds.plot(
                [earlier["generation"][-1], row["generation"]],
                [
                    min(earlier["doubling_time"][-1], y_max),
                    min(row["doubling_time"], y_max),
                ],
                color=cmap(seed_index / max(1, len(seeds) - 1)),
                linewidth=0.9,
                alpha=0.65,
            )
        ax_seeds.plot(
            row["generation"],
            min(row["doubling_time"], y_max),
            "s",
            color=MAX_DURATION_COLOR,
            markersize=7,
            zorder=5,
        )

    # Mark each failed cell at its lineage's last completed doubling time
    for row in failed.iter_rows(named=True):
        earlier = per_seed.filter(
            (pl.col("lineage_seed") == row["lineage_seed"])
            & (pl.col("generation") < row["generation"])
        )
        y_value = earlier["doubling_time"][-1] if earlier.height else y_min
        ax_seeds.plot(
            row["generation"],
            min(y_value, y_max),
            "x",
            color=DEATH_COLOR,
            markersize=9,
            markeredgewidth=2.2,
            zorder=5,
        )

    n_failed_seeds = failed["lineage_seed"].n_unique()
    ax_seeds.plot(
        [],
        [],
        "x",
        color=DEATH_COLOR,
        markersize=9,
        markeredgewidth=2.2,
        label=f"lineage failed ({n_failed_seeds} of {len(seeds)} seeds)",
    )
    if at_limit.height:
        ax_seeds.plot(
            [],
            [],
            "s",
            color=MAX_DURATION_COLOR,
            markersize=7,
            label=(
                f"reached max duration ({at_limit.height} "
                f"{'cell' if at_limit.height == 1 else 'cells'})"
            ),
        )

    generations = per_generation["generation"].to_numpy()
    means = per_generation["mean"].to_numpy()
    stds = per_generation["std"].to_numpy()
    ax_mean.plot(generations, means, color=MEAN_COLOR, linewidth=2, label="mean")
    ax_mean.fill_between(
        generations,
        means - stds,
        means + stds,
        color=MEAN_COLOR,
        alpha=0.18,
        linewidth=0,
        label="± SD across seeds",
    )

    for ax in (ax_seeds, ax_mean):
        if target:
            ax.axhline(
                target,
                color=TARGET_COLOR,
                linestyle="--",
                linewidth=1.2,
                label=f"target ({target:.0f} min)",
            )
        ax.set_ylim(y_min, y_max * 1.03)
        ax.set_ylabel("doubling time (min)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax_seeds.set_title(f"Doubling time, one line per seed: {summary}", loc="left")
    ax_mean.set_title("Mean and SD across seeds", loc="left")
    ax_mean.set_xlabel("generation")
    ax_mean.xaxis.get_major_locator().set_params(integer=True)

    # Histogram of completed cells and cells alive at the time limit
    after_skip = cells.filter(pl.col("generation") >= first_generation + skip_n_gens)
    included = after_skip.filter(pl.col("success") | reached_limit)
    n_at_limit = after_skip.filter(~pl.col("success") & reached_limit).height
    n_failed = after_skip.height - included.height
    if included.is_empty():
        ax_hist.text(0.5, 0.5, "no cells to show", transform=ax_hist.transAxes)
    else:
        values = included["doubling_time"].to_numpy()
        failed_word = "generation" if n_failed == 1 else "generations"
        label = (
            f"{included.height} cells included "
            f"({n_failed} failed {failed_word} excluded"
        )
        label += f", {n_at_limit} at max duration kept)" if n_at_limit else ")"
        ax_hist.hist(values, bins=40, color=MEAN_COLOR, alpha=0.75, label=label)
        ax_hist.axvline(
            values.mean(),
            color=DEATH_COLOR,
            linewidth=1.5,
            label=f"mean {values.mean():.1f} ± {values.std(ddof=1) if values.size > 1 else 0:.1f} min",
        )
    if target:
        ax_hist.axvline(
            target,
            color=TARGET_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=f"target ({target:.0f} min)",
        )
    ax_hist.set_xlabel("doubling time (min)")
    ax_hist.set_ylabel("cells")
    ax_hist.yaxis.get_major_locator().set_params(integer=True)
    ax_hist.set_title(
        f"Doubling times, skipping the first {skip_n_gens} generations", loc="left"
    )
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax_hist.set_axisbelow(True)
    ax_hist.legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "doubling_time_trajectories.pdf"))
    fig.savefig(os.path.join(outdir, "doubling_time_trajectories.png"), dpi=150)
    plt.close(fig)
    print(f"Doubling time: {summary}")
