"""
===========
Equilibrium
===========

Equilibrium binding sub-model.

This process models how ligands are bound to or unbound
from their transcription factor binding partners in a fashion
that maintains equilibrium.
"""

import numpy as np

from vivarium.core.process import Process
from vivarium.library.dict_utils import deep_merge

from ecoli.library.schema import array_from, array_to, bulk_schema

from wholecell.utils import units
from six.moves import range

from ecoli.processes.registries import topology_registry

# Register default topology for this process, associating it with process name
NAME = 'ecoli-equilibrium'
TOPOLOGY = {
        "listeners": ("listeners",),
        "molecules": ("bulk",)
}
topology_registry.register(NAME, TOPOLOGY)


class Equilibrium(Process):
    name = NAME
    topology = TOPOLOGY
    """
        molecule_names: list of molecules that are being iterated over size:94
    """
    defaults = {
        'jit': False,
        'n_avogadro': 0.0,
        'cell_density': 0.0,
        'stoichMatrix': [[]],
        'fluxesAndMoleculesToSS': lambda counts, volume, avogadro, random, jit: ([], []),
        'moleculeNames': [],
        'seed': 0}

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Simulation options
        self.jit = self.parameters['jit']  # utilized in the fluxes and molecules function

        # Get constants
        self.n_avogadro = self.parameters['n_avogadro']
        self.cell_density = self.parameters['cell_density']

        # Create matrix and method
        # stoichMatrix: (94, 33), molecule counts are (94,).
        self.stoichMatrix = self.parameters['stoichMatrix']

        # fluxesAndMoleculesToSS: solves ODES to get to steady state based off of cell density,
        # volumes and molecule counts
        self.fluxesAndMoleculesToSS = self.parameters['fluxesAndMoleculesToSS']

        self.product_indices = [idx for idx in np.where(np.any(self.stoichMatrix > 0, axis=1))[0]]

        # Build views
        # moleculeNames: list of molecules that are being iterated over size: 94
        self.moleculeNames = self.parameters['moleculeNames']

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)

    def ports_schema(self):
        return {
            'molecules': bulk_schema(self.moleculeNames),
            'listeners': {
                'mass': {
                    'cell_mass': {'_default': 0}},
                'equilibrium_listener': {
                    'reaction_rates': {'_default': [], '_updater': 'set', '_emit': True}}}}
        
    def calculate_request(self, timestep, states):
        # Get molecule counts
        moleculeCounts = array_from(states['molecules'])

        # Get cell mass and volume
        cellMass = (states['listeners']['mass']['cell_mass'] * units.fg).asNumber(units.g)
        cellVolume = cellMass / self.cell_density

        # Solve ODEs to steady state
        self.rxnFluxes, self.req = self.fluxesAndMoleculesToSS(
            moleculeCounts, cellVolume, self.n_avogadro, self.random_state,
            jit=self.jit,
            )

        # Request counts of molecules needed
        requests = {}
        requests['molecules'] = array_to(states['molecules'], self.req)
        return requests
        
    def evolve_state(self, timestep, states):
        # Get molecule counts
        moleculeCounts = array_from(states['molecules'])
        
        # Get counts of molecules allocated to this process
        rxnFluxes = self.rxnFluxes.copy()

        # If we didn't get allocated all the molecules we need, make do with
        # what we have (decrease reaction fluxes so that they make use of what
        # we have, but not more). Reduces at least one reaction every iteration
        # so the max number of iterations is the number of reactions that were
        # originally expected to occur + 1 to reach the break statement.
        max_iterations = int(np.abs(rxnFluxes).sum()) + 1
        for it in range(max_iterations):
            # Check if any metabolites will have negative counts with current reactions
            negative_metabolite_idxs = np.where(np.dot(self.stoichMatrix, rxnFluxes) + moleculeCounts < 0)[0]
            if len(negative_metabolite_idxs) == 0:
                break

            # Reduce reactions that consume metabolites with negative counts
            limited_rxn_stoich = self.stoichMatrix[negative_metabolite_idxs, :]
            fwd_rxn_idxs = np.where(np.logical_and(limited_rxn_stoich < 0, rxnFluxes > 0))[1]
            rev_rxn_idxs = np.where(np.logical_and(limited_rxn_stoich > 0, rxnFluxes < 0))[1]
            rxnFluxes[fwd_rxn_idxs] -= 1
            rxnFluxes[rev_rxn_idxs] += 1
            rxnFluxes[fwd_rxn_idxs] = np.fmax(0, rxnFluxes[fwd_rxn_idxs])
            rxnFluxes[rev_rxn_idxs] = np.fmin(0, rxnFluxes[rev_rxn_idxs])
        else:
            raise ValueError('Could not get positive counts in equilibrium with'
                ' allocated molecules.')

        # Increment changes in molecule counts
        deltaMolecules = np.dot(self.stoichMatrix, rxnFluxes).astype(int)

        update = {
            'molecules': array_to(self.moleculeNames, deltaMolecules),
            'listeners': {
                'equilibrium_listener': {
                    'reaction_rates': deltaMolecules[self.product_indices] / timestep}}}

        return update

    def next_update(self, timestep, states):
        requests = self.calculate_request(timestep, states)
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        return update


def test_equilibrium_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    sim = EcoliSim.from_file()
    sim.total_time = 2
    data = sim.run()
    assert(type(data['listeners']['equilibrium_listener']['reaction_rates'][0]) == list)
    assert(type(data['listeners']['equilibrium_listener']['reaction_rates'][1]) == list)


if __name__ == '__main__':
    test_equilibrium_listener()
