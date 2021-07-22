"""
Two component system

Two component system sub-model

"""

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process

from ecoli.library.schema import (
    array_from, array_to, arrays_from,
    arrays_to, listener_schema, bulk_schema)

from wholecell.utils import units


class TwoComponentSystem(Process):
    name = 'ecoli-two-component-system'

    defaults = {
        'jit': False,
        'n_avogadro': 0.0,
        'cell_density': 0.0,
        'moleculesToNextTimeStep': lambda counts, volume, avogadro, timestep, random, method, min_step, jit: (
            [], []),
        'moleculeNames': [],
        'seed': 0}

    # Constructor
    def __init__(self, initial_parameters):
        super().__init__(initial_parameters)

        # Simulation options
        self.jit = self.parameters['jit']

        # Get constants
        self.n_avogadro = self.parameters['n_avogadro']
        self.cell_density = self.parameters['cell_density']

        # Create method
        self.moleculesToNextTimeStep = self.parameters['moleculesToNextTimeStep']

        # Build views
        self.moleculeNames = self.parameters['moleculeNames']

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)

    def ports_schema(self):
        return {
            'molecules': bulk_schema(self.moleculeNames),
            'listeners': {
                'mass': {
                    'cell_mass': {'_default': 0}}}}

    def next_update(self, timestep, states):
        # Get molecule counts
        moleculeCounts = array_from(states['molecules'])

        # Get cell mass and volume
        cellMass = (states['listeners']['mass']['cell_mass'] * units.fg).asNumber(units.g)
        self.cellVolume = cellMass / self.cell_density

        # Solve ODEs to next time step using the BDF solver through solve_ivp.
        # Note: the BDF solver has been empirically tested to be the fastest
        # solver for this setting among the list of solvers that can be used
        # by the scipy ODE suite.
        self.molecules_required, self.all_molecule_changes = self.moleculesToNextTimeStep(
            moleculeCounts, self.cellVolume, self.n_avogadro,
            timestep, self.random_state, method="BDF", jit=self.jit)

        # Increment changes in molecule counts
        update = {
            'molecules': array_to(self.moleculeNames, self.all_molecule_changes.astype(int))}

        return update
