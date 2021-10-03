"""
====================
Two Component System
====================

This process models the phosphotransfer reactions of signal transduction pathways.

Specifically, phosphate groups are transferred from histidine kinases to response regulators
and back in response to counts of ligand stimulants.
"""

import numpy as np

from vivarium.core.process import Process
from vivarium.library.dict_utils import deep_merge

from ecoli.library.schema import (
    array_from, array_to, bulk_schema)

from wholecell.utils import units
from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = 'ecoli-two-component-system'
TOPOLOGY = {
        "listeners": ("listeners",),
        "molecules": ("bulk",)
}
topology_registry.register(NAME, TOPOLOGY)


class TwoComponentSystem(PartitionedProcess):
    """ Two Component System PartitionedProcess """

    name = NAME
    topology = TOPOLOGY
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
        
        
    def calculate_request(self, timestep, states):
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
            timestep, self.random_state, method="BDF", jit=self.jit,
            )
        requests = {'molecules': array_to(states['molecules'], self.molecules_required)}
        return requests
    
    def evolve_state(self, timestep, states):
        moleculeCounts = array_from(states['molecules'])
        # Check if any molecules were allocated fewer counts than requested
        if (self.molecules_required > moleculeCounts).any():
            _, self.all_molecule_changes = self.moleculesToNextTimeStep(
                moleculeCounts, self.cellVolume, self.n_avogadro,
                10000, self.random_state, method="BDF", min_time_step=timestep,
                jit=self.jit)
        # Increment changes in molecule counts
        update = {
            'molecules': array_to(self.moleculeNames, self.all_molecule_changes.astype(int))}
        
        return update
