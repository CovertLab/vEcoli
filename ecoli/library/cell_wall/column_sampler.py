import argparse
import os
from collections import Counter
from itertools import accumulate, chain, takewhile, tee
from re import S

import matplotlib.pyplot as plt
import numpy as np
from sympy import arg


def poisson_sampler(rng, rate):
    def sampler():
        while True:
            yield rng.poisson(rate)

    return sampler


def geom_sampler(rng, p):
    def sampler():
        while True:
            yield rng.geometric(p)

    return sampler


def sample_column(rows, murein, strand_sampler, rng, shift=True):
    result = np.zeros(rows, dtype=int)

    # Don't try to assign more murein than can fit in the column
    murein = int(min(murein, rows))

    # Create iterator for strand lengths, total accumulated length
    strand_length, total_length = tee(strand_sampler())
    total_length = accumulate(total_length)

    # Sample strand lengths
    strands = [
        s
        for s, _ in takewhile(lambda s: s[1] < murein, zip(strand_length, total_length))
        if s > 0
    ]
    remaining = murein - sum(strands)
    if remaining:
        strands.append(remaining)

    # Get probability for initiating a strand
    total_gap = rows - sum(strands)
    strand_starts = list(rng.integers(0, total_gap + 1, size=len(strands)))
    strand_starts.sort(reverse=True)

    result_i = 0
    next_start = strand_starts.pop()
    for gap_i in range(total_gap + 1):
        while next_start == gap_i:
            strand = strands.pop()
            result[result_i:(result_i + strand)] = 1
            result_i += strand
            next_start = strand_starts.pop() if len(strands) > 0 else -1
        result_i += 1

    if shift:
        result = np.roll(result, rng.integers(len(result)))

    return result


def sample_lattice(rows, columns, murein, strand_sampler, rng):
    result = np.zeros((rows, columns), dtype=int)

    # Get murein in each column, distributing extra murein uniformly at random
    murein_per_column = np.repeat(murein // columns, repeats=columns)
    extra_murein = murein % columns
    for col in rng.integers(0, columns, size=extra_murein):
        murein_per_column[col] += 1

    for c, murein in enumerate(murein_per_column):
        result[:, c] = sample_column(rows, murein, strand_sampler, rng)

    return result


def length_distributions(column):
    strand_lengths = []
    gap_lengths = []

    in_strand = bool(column[0])
    current_length = 0
    for x in column:
        if x == int(in_strand):
            current_length += 1
        else:
            (strand_lengths if in_strand else gap_lengths).append(current_length)
            in_strand = not in_strand
            current_length = 1

    return strand_lengths, gap_lengths


def plot_length_distributions(strand_lengths, gap_lengths):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    for c, data in enumerate([strand_lengths, gap_lengths]):
        flat_data = list(chain(*data))
        N = len(flat_data)

        # Create histogram of each run in the data
        histograms = np.zeros((len(data), max(flat_data) + 1))
        for run, run_data in enumerate(data):
            hist = Counter(run_data)
            for k, v in hist.items():
                histograms[run, k] = v

        # Normalize as probability distributions
        distributions = histograms / histograms.sum(axis=1)[:, np.newaxis]

        # Calculate means, error bars
        y = distributions.mean(axis=0)
        err = distributions.std(axis=0) / np.sqrt(N)
        bincenters = np.arange(y.size)
        axs[c].bar(bincenters, y, yerr=err)
        axs[c].set_xticks(bincenters)
        axs[c].set_xticklabels(axs[c].get_xticks(), rotation=45)
        axs[c].set_title(f"{('Strand', 'Gap')[c]} Length Distribution")

    return fig, axs


def plot_locational(columns):
    columns = np.array(columns)
    runs, positions = columns.shape

    strands = columns.sum(axis=0)
    gaps = runs - strands

    fig, ax = plt.subplots()
    x = np.arange(positions)
    ax.bar(x, strands, width=1, label="Strand")
    ax.bar(x, gaps, width=1, bottom=strands, label="Gap")
    ax.set_title(f"Count of Strand/Gap By Position, over {runs} runs")
    ax.legend()

    return fig, ax


def test_column_sampler(outdir="out/murein_sampling"):
    rng = np.random.default_rng(0)

    columns = []
    strand_lengths = []
    gap_lengths = []

    rows = 3050
    cols = 700
    murein = 401898
    for i in range(cols):
        col = sample_column(rows, int(murein // cols), geom_sampler(rng, 0.058), rng)
        s, g = length_distributions(col)
        columns.append(col)
        strand_lengths.append(s)
        gap_lengths.append(g)

    # Diagnostic plots ====================================================
    os.makedirs(outdir, exist_ok=True)

    fig, _ = plot_length_distributions(strand_lengths, gap_lengths)
    fig.set_size_inches((10, 6))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "length_distributions.png"))

    fig, _ = plot_locational(columns)
    fig.set_size_inches((8, 6))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "locational.png"))


def main():
    parser = argparse.ArgumentParser(description="Sample lattice and run tests.")

    parser.add_argument("--test", "-t", action="store_true", help="Run tests.")
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for sampling (not fixed if None)",
    )

    parser.add_argument("rows", type=int, help="Number of rows to sample.")
    parser.add_argument("cols", type=int, help="Number of cols to sample.")
    parser.add_argument("murein", type=int, help="Murein limitation.")
    parser.add_argument("rate", type=float, help="Rate for poisson distribution")
    parser.add_argument("--file", "-f", type=str, default=None, help="File to output to.")

    args = parser.parse_args()

    if args.test:
        test_column_sampler()
    else:
        rng = np.random.default_rng(args.seed)
        lat = sample_lattice(
            args.rows, args.cols, args.murein, poisson_sampler(rng, args.rate), rng
        )
        if args.file:
            with open(args.file, "w") as f:
                f.write("[")
                for row in lat:
                    f.write(f'{list(row)},\n')
                f.write("]\n")


if __name__ == "__main__":
    main()
