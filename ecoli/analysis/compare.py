import argparse
import math

from matplotlib import pyplot as plt
from vivarium.core.emitter import path_timeseries_from_data
from vivarium.library.topology import convert_path_style

from ecoli.analysis.analyze_db_experiment import access, OUT_DIR


COL_WIDTH = 3
ROW_HEIGHT = 3
MAX_COLS = 5
LEGEND_WIDTH = 5
ALPHA = 0.5


def make_plot(datasets):
    paths = list(datasets.values())[0].keys() - {"time"}
    n_variables = len(paths)
    n_cols = min(n_variables, MAX_COLS) + 1  # 1 col for legend
    n_rows = math.ceil(n_variables / MAX_COLS)
    fig = plt.figure(figsize=(n_cols * COL_WIDTH + LEGEND_WIDTH, n_rows * ROW_HEIGHT))
    grid = plt.GridSpec(n_rows, n_cols)

    row_idx = 0
    col_idx = 0

    for path_idx, path in enumerate(paths):
        ax = fig.add_subplot(grid[row_idx, col_idx])
        plotted = False
        for key, ts in datasets.items():
            times = ts["time"]
            try:
                data = ts[path]
            except KeyError:
                # This dataset does not contain path, so skip it.
                continue

            try:
                iter(data[0])
            except TypeError:
                pass
            else:
                # The data values are iterable, so skip.
                continue

            plotted = True
            if path_idx == len(paths) - 1:
                ax.plot(times, data, label=key, alpha=ALPHA)
            else:
                ax.plot(times, data, alpha=ALPHA)
        if not plotted:
            # If nothing was plotted, skip this path entirely.
            continue
        ax.set_title("\n".join(path))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if row_idx == n_rows - 1 or path_idx == len(paths) - 1:
            ax.set_xlabel("Time (s)")
            row_idx = 0
            col_idx += 1
        else:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)
            ax.tick_params(labelbottom=False)
            row_idx += 1
        if path_idx == len(paths) - 1:
            fig.legend()
    fig.tight_layout()
    return fig


def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Compare data from multiple experiments"
    )
    parser.add_argument(
        "--experiment_id",
        "-e",
        nargs="+",
        help="Experiment ID to include in comparison.",
    )
    parser.add_argument(
        "--path",
        "-p",
        nargs="+",
        help='Path to variable to compare, e.g. "agents>0>mass"',
    )
    parser.add_argument(
        "--out", "-o", default=f"{OUT_DIR}comparison.png", help="Output path"
    )

    args = parser.parse_args(cli_args)

    paths = [convert_path_style(path) for path in args.path]
    datasets = {
        experiment_id: path_timeseries_from_data(access(experiment_id, paths)[0])
        for experiment_id in args.experiment_id
    }

    fig = make_plot(datasets)
    fig.savefig(args.out)


if __name__ == "__main__":
    main()
