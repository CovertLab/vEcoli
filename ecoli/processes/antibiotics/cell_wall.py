"""
TODO: references for parameters
"""

import os
from random import choice
import numpy as np
from matplotlib import pyplot as plt

from vivarium.library.units import units

from vivarium.core.engine import Engine
from vivarium.core.composition import simulate_process, simulate_composite
from vivarium.core.process import Process
from vivarium.plots.simulation_output import plot_variables

from ecoli.processes.registries import topology_registry
from ecoli.library.schema import bulk_schema


# Register default topology for this process, associating it with process name
NAME = 'ecoli-cell-wall'
TOPOLOGY = {
    'shape': ('shape',),
    'bulk_murein': ("bulk",),
}
topology_registry.register(NAME, TOPOLOGY)


class CellWall(Process):

    name = NAME
    topology = TOPOLOGY
    defaults = {
        # Molecules
        'murein': 'CPD-12261[p]',  # two crosslinked peptidoglycan units
        'SLT': '',  # TBD
        'PBP': {  # penicillin-binding proteins
            'PBP1A': 'CPLX0-7717[i]',  # transglycosylase-transpeptidase ~100
            'PBP1B': 'CPLX0-3951[i]'  # transglycosylase-transpeptidase ~100
        },
        'cephaloridine': '',

        # Physical parameters
        'critical_radius': 20 * units.nm,
        'cell_radius': 0.5 * units.um,
        # 4.1 in maximally stretched configuration,
        'disaccharide_length': 1.03 * units.nm,
        # divided by 3 because the sacculus can be stretched threefold
        'crossbridge_length': 	4.1 * units.nm / 3,

        # Simulation parameters
        'seed': 0
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Get murein id and keep track of murein from last timestep
        self.murein = self.parameters['murein']

        self.critical_radius = 20 * units.nm
        self.critical_area = np.pi * self.critical_radius**2

        # Create pseudorandom number generator
        self.rng = np.random.default_rng(self.parameters['seed'])

    def ports_schema(self):
        schema = {
            'bulk_murein': bulk_schema([self.parameters['murein']]),
            'murein_state': {
                'free_murein': {'_default': 0, '_updater': 'set'},
                'incorporated_murein': {'_default': 0, '_updater': 'set'}
            },
            'PBP': bulk_schema(self.parameters['PBP'].values()),

            'shape': {
                "length": {
                    '_default': 0 * units.um
                }
            },

            'wall_state': {
                'lattice': {
                    '_default': list()  # A sparse, set-of-defects representation of the cell wall
                },
                'lattice_rows': {
                    '_default': 0
                },
                'lattice_cols': {
                    '_default': 0
                },
                'cracked': {
                    '_default': False
                }
            }
        }

        return schema

    def next_update(self, timestep, states):
        DEBUG = True

        # Unpack states
        length = states['shape']['length']
        lattice = states['wall_state']['lattice']
        lattice_rows = states['wall_state']['lattice_rows']
        lattice_cols = states['wall_state']['lattice_cols']

        update = {}

        if DEBUG:
            #states['bulk_murein'][self.murein] = 3000000
            update['shape'] = {"length": length + 0.1 * units.um}
            assert states['bulk_murein'][self.murein] == states['murein_state']['free_murein'] + \
                states['murein_state']['incorporated_murein']

        # Expand lattice size if necessary, depending on cell size
        print("resizing lattice")
        lattice, rows, columns = self.resize_lattice(length, self.parameters['cell_radius'],
                                                     lattice, lattice_rows, lattice_cols)

        # Cell wall construction/destruction
        print("assigning murein")
        lattice, new_free_murein, new_incorporated_murein = self.assign_murein(states['murein_state']['free_murein'],
                                                                               states['murein_state']['incorporated_murein'],
                                                                               lattice, rows, columns)
        print(len(lattice))

        update['wall_state'] = {
            'lattice': lattice,
            'lattice_rows': rows,
            'lattice_cols': columns
        }

        update['murein_state'] = {
            'free_murein': new_free_murein,
            'incorporated_murein': new_incorporated_murein
        }

        # Crack detection (cracking is irreversible)
        print("crack detection")
        if (not states['wall_state']['cracked']
                and self.get_largest_defect_area(lattice) > self.critical_area):
            update['wall_state']['cracked'] = True

        return update

    def resize_lattice(self, cell_length, cell_radius,
                       lattice, prev_rows=0, prev_cols=0):

        # Calculate new lattice size
        columns = int(cell_length / self.parameters['crossbridge_length'])  # TODO: account for sugar
        self.circumference = 2 * np.pi * cell_radius
        rows = int(self.circumference / self.parameters['disaccharide_length'])

        # Fill in new positions with defects initially
        if columns > prev_cols:
            for c in range(prev_cols, columns):
                for r in range(rows):
                    lattice.append((r, c))

        if rows > prev_rows:
            for r in range(prev_rows, rows):
                for c in range(columns):
                    lattice.append((r, c))

        return lattice, rows, columns

    def assign_murein(self, free_murein, incorporated_murein, lattice, rows, columns):
        n_holes = len(lattice)
        n_incorporated = rows * columns - n_holes

        # fill holes
        fill_n = min(free_murein, n_holes)
        fill_idx = self.rng.choice(
            np.arange(len(lattice)), size=fill_n, replace=False)
        lattice = [lattice[i]
                   for i in range(len(lattice)) if i not in fill_idx]

        # add holes
        new_holes = n_incorporated - incorporated_murein
        # if new_holes > 0:
        #     for _ in range(new_holes):
        #         candidate = (self.rng.integers(rows),
        #                      self.rng.integers(columns))
        #         while candidate in lattice:
        #             candidate = (self.rng.integers(rows),
        #                          self.rng.integers(columns))
        #         lattice.append(candidate)

        while new_holes > 0:
            candidates = (self.rng.integers(rows, size=new_holes),
                          self.rng.integers(columns, size=new_holes))
            redo = 0
            for c in range(len(candidates)):
                cand = (candidates[0][c], candidates[1][c])
                if c in lattice:
                    redo += 1
                else:
                    lattice.append(cand)
            new_holes = redo

        total_murein = free_murein + incorporated_murein
        new_incorporated = (rows*columns - len(lattice))
        new_free = total_murein - new_incorporated
        return lattice, new_free, new_incorporated

    def get_largest_defect_area(self, lattice):
        # TODO: generate hole-view from defects
        max_size = 0

        # TODO: replace with actual area from literature
        return max_size * 4 * units.nm**2


def get_full_lattice(sparse_lattice, rows, cols):
    result = np.ones((rows, cols))

    for r, c in sparse_lattice:
        result[r, c] = 0

    return result


def plot_lattice(lattice):
    fig, axs = plt.subplot()
    fig.imshow(lattice, interpolation='nearest')
    return fig, axs


def main():
    from vivarium.processes.timeline import TimelineProcess
    from vivarium.core.composer import Composite

    # Stub for rest of cell (increasing murein)
    cell_stub = TimelineProcess({
        'time_step': 2.0,
        'timeline': [(time, {('bulk_murein', 'CPD-12261[p]'): int(3e6 + 1000 * time)}) for time in range(0, 10, 2)]
    })

    # Cell wall process
    params = {

    }
    cell_wall = CellWall(params)

    settings = {
        'total_time': 10,
        'initial_state': {
            'bulk_murein': {
                'CPD-12261[p]': int(3e6)
            },
            'shape': {
                'length': 1 * units.um
            },
            'murein_state': {
                'incorporated_murein': 3000000
            },
            'wall_state': {
                'lattice_rows': 6100,
                'lattice_cols': 1463
            }
        }
    }

    # sim = Engine(
    #     processes={
    #         # 'cell_stub': cell_stub,
    #         "cell_wall": cell_wall
    #     },
    #     topology={
    #         'cell_wall': {
    #             'bulk_murein': ("bulk", 'CPD-12261[p]'),
    #             'shape' : ("shape",)
    #         },
    #         # 'cell_stub': {
    #         #     'global': ('global',),
    #         #     'bulk_murein': ("bulk", 'CPD-12261[p]'),
    #         # }
    #     },
    #     initial_state={
    #         'bulk': {
    #             'CPD-12261[p]': int(3e6),
    #         },
    #         'shape' : {
    #             'length' : 2 * units.um,
    #         }})
    #
    # sim.run_for(10)
    # data = sim.emitter.get_data()

    data = simulate_process(cell_wall, settings)
    #data = simulate_process(cell_stub, settings)
    #data = simulate_composite(test_composite, settings)
    fig = plot_variables(
        data,
        variables=[
            ("murein_state", 'free_murein'),
            ("murein_state", 'incorporated_murein'),
            ("shape", "length"),
            ("wall_state", "rows"),
            ("wall_state", "columns")
        ],
    )
    fig.tight_layout()

    os.makedirs("out/processes/cell_wall/", exist_ok=True)
    fig.savefig("out/processes/cell_wall/test.png")

    for t, lattice in enumerate([]):
        fig = plot_lattice(get_full_lattice(lattice))
        fig.tight_layout()
        fig.save_fig(f"out/processes/cell_wall_t{t}.png")


if __name__ == '__main__':
    main()
