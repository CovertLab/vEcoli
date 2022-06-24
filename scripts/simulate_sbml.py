import argparse

from bioscrape.sbmlutil import import_sbml
from bioscrape.simulator import py_simulate_model, ModelCSimInterface
import numpy as np
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Simulate an SBML model')
    parser.add_argument(
        'sbml_file', type=str,
        help='Path to SBML file containing the model to simulate.')
    parser.add_argument(
        '-s', '--start', type=float, default=0,
        help='Time to start simulation. 0 by default.')
    parser.add_argument(
        '-e', '--end', type=float, default=10,
        help='Time to end simulation. 10 by default.')
    parser.add_argument(
        '-t', '--step', type=float, default=1,
        help='Time step for output. 1 by default.')
    parser.add_argument(
        '-m', '--hmax', type=float, default=0,
        help='Maximum step size for numerical integrator.')
    parser.add_argument(
        '-o', '--out', type=str, default='simulation.png',
        help='Output path. simulation.png by default.')
    parser.add_argument(
        '-k', '--skip', type=str, nargs='+', default=[],
        help='Variables to exclude from plot.')
    parser.add_argument(
        '-d', '--debugger', action='store_true',
        help='Open a Pdb shell after generating the figure.')

    args = parser.parse_args()

    model = import_sbml(args.sbml_file)
    timepoints = np.arange(args.start, args.end, args.step)
    result = py_simulate_model(timepoints, Model=model, hmax=args.hmax)

    fig, ax = plt.subplots()

    for species in model.get_species_list():
        if species in args.skip:
            continue
        ax.plot(timepoints, result[species], label=species)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amounts')
    ax.set_title(f'Simulating {args.sbml_file}')
    ax.legend()
    fig.savefig(args.out)

    if args.debugger:
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
