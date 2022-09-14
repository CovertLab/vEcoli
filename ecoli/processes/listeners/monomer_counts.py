"""
====================
Monomer Counts Listener
====================
"""

import numpy as np
from ecoli.library.schema import bulk_schema, array_from
from vivarium.core.process import Step

from ecoli.processes.registries import topology_registry


NAME = 'monomer_counts_listener'
TOPOLOGY = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "unique": ("unique",),
}
topology_registry.register(
    NAME, TOPOLOGY
)

class MonomerCounts(Step):
    """
    Listener for the counts of each protein monomer species.
    """
    name = NAME
    topology = TOPOLOGY

    defaults = {
        'bulk_molecule_ids': [],
        'unique_ids': [],
        'complexation_molecule_ids': [],
        'complexation_complex_ids': [],
        'equilibrium_molecule_ids': [],
        'equilibrium_complex_ids': [],
        'monomer_ids': [],
        'two_component_system_molecule_ids': [],
        'two_component_system_complex_ids': [],
        'ribosome_50s_subunits': [],
        'ribosome_30s_subunits': [],
        'rnap_subunits': [],
        'replisome_trimer_subunits': [],
        'replisome_monomer_subunits': [],
        'complexation_stoich': [],
        'equilibrium_stoich': [],
        'two_component_system_stoich': [],
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Get IDs of all bulk molecules
        self.bulk_molecule_ids = self.parameters['bulk_molecule_ids']

        # Get IDs of molecules involved in complexation and equilibrium
        complexation_molecule_ids = self.parameters['complexation_molecule_ids']
        complexation_complex_ids = self.parameters['complexation_complex_ids']
        equilibrium_molecule_ids = self.parameters['equilibrium_molecule_ids']
        equilibrium_complex_ids = self.parameters['equilibrium_complex_ids']
        self.monomer_ids = self.parameters['monomer_ids']

        # Get IDs of complexed molecules monomers involved in two component system
        two_component_system_molecule_ids = self.parameters['two_component_system_molecule_ids']
        two_component_system_complex_ids = self.parameters['two_component_system_complex_ids']

        # Get IDs of ribosome subunits
        ribosome_50s_subunits = self.parameters['ribosome_50s_subunits']
        ribosome_30s_subunits = self.parameters['ribosome_30s_subunits']
        ribosome_subunit_ids = (ribosome_50s_subunits["subunitIds"].tolist() +
                                ribosome_30s_subunits["subunitIds"].tolist())

        # Get IDs of RNA polymerase subunits
        rnap_subunits = self.parameters['rnap_subunits']
        rnap_subunit_ids = rnap_subunits["subunitIds"].tolist()

        # Get IDs of replisome subunits
        replisome_trimer_subunits = self.parameters['replisome_trimer_subunits']
        replisome_monomer_subunits = self.parameters['replisome_monomer_subunits']
        replisome_subunit_ids = replisome_trimer_subunits + replisome_monomer_subunits

        # Get stoichiometric matrices for complexation, equilibrium, two component system and the
        # assembly of unique molecules
        self.complexation_stoich = self.parameters['complexation_stoich']
        self.equilibrium_stoich = self.parameters['equilibrium_stoich']
        self.two_component_system_stoich = self.parameters['two_component_system_stoich']
        self.ribosome_stoich = np.hstack(
            (ribosome_50s_subunits["subunitStoich"],
             ribosome_30s_subunits["subunitStoich"]))
        self.rnap_stoich = rnap_subunits["subunitStoich"]
        self.replisome_stoich = np.hstack(
            (3 * np.ones(len(replisome_trimer_subunits)),
             np.ones(len(replisome_monomer_subunits))))

        # Construct dictionary to quickly find bulk molecule indexes from IDs
        molecule_dict = {mol: i for i, mol in enumerate(self.bulk_molecule_ids)}

        def get_molecule_indexes(keys):
            return np.array([molecule_dict[x] for x in keys])

        # Get indexes of all relevant bulk molecules
        self.monomer_idx = get_molecule_indexes(self.monomer_ids)
        self.complexation_molecule_idx = get_molecule_indexes(complexation_molecule_ids)
        self.complexation_complex_idx = get_molecule_indexes(complexation_complex_ids)
        self.equilibrium_molecule_idx = get_molecule_indexes(equilibrium_molecule_ids)
        self.equilibrium_complex_idx = get_molecule_indexes(equilibrium_complex_ids)
        self.two_component_system_molecule_idx = get_molecule_indexes(two_component_system_molecule_ids)
        self.two_component_system_complex_idx = get_molecule_indexes(two_component_system_complex_ids)
        self.ribosome_subunit_idx = get_molecule_indexes(ribosome_subunit_ids)
        self.rnap_subunit_idx = get_molecule_indexes(rnap_subunit_ids)
        self.replisome_subunit_idx = get_molecule_indexes(replisome_subunit_ids)

    def ports_schema(self):
        return {
            'listeners': {
                'monomer_counts': {
                    '_default': [],
                    '_updater': 'set',
                    '_emit': True}
            },
            'bulk': bulk_schema(self.bulk_molecule_ids),
            'unique': {
                str(unique_mol): {
                    '_default': {}
                } for unique_mol in self.parameters['unique_ids']
            },
        }

    def next_update(self, timestep, states):

        # Get current counts of bulk and unique molecules
        # uniqueMoleculeCounts = self.uniqueMolecules.container.counts()
        bulkMoleculeCounts = array_from(states['bulk'])
        n_active_ribosome = len(states['unique']['active_ribosome'])
        n_active_rnap = len(states['unique']['active_RNAP'])
        n_active_replisome = len(states['unique']['active_replisome'])

        # Account for monomers in bulk molecule complexes
        complex_monomer_counts = np.dot(self.complexation_stoich,
                                        np.negative(bulkMoleculeCounts[self.complexation_complex_idx]))
        equilibrium_monomer_counts = np.dot(self.equilibrium_stoich,
                                            np.negative(bulkMoleculeCounts[self.equilibrium_complex_idx]))
        two_component_monomer_counts = np.dot(self.two_component_system_stoich,
                                              np.negative(bulkMoleculeCounts[self.two_component_system_complex_idx]))

        bulkMoleculeCounts[self.complexation_molecule_idx] += complex_monomer_counts.astype(np.int)
        bulkMoleculeCounts[self.equilibrium_molecule_idx] += equilibrium_monomer_counts.astype(np.int)
        bulkMoleculeCounts[self.two_component_system_molecule_idx] += two_component_monomer_counts.astype(np.int)

        # Account for monomers in unique molecule complexes
        n_ribosome_subunit = n_active_ribosome * self.ribosome_stoich
        n_rnap_subunit = n_active_rnap * self.rnap_stoich
        n_replisome_subunit = n_active_replisome * self.replisome_stoich
        bulkMoleculeCounts[self.ribosome_subunit_idx] += n_ribosome_subunit.astype(np.int)
        bulkMoleculeCounts[self.rnap_subunit_idx] += n_rnap_subunit.astype(np.int)
        bulkMoleculeCounts[self.replisome_subunit_idx] += n_replisome_subunit.astype(np.int)

        # Update monomerCounts
        monomer_counts = bulkMoleculeCounts[self.monomer_idx]

        update = {
            'listeners': {
                'monomer_counts': monomer_counts
            }
        }
        return update


def test_monomer_counts_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    sim = EcoliSim.from_file()
    sim.total_time = 2
    sim.raw_output = False
    sim.run()
    data = sim.query()
    assert(type(data['listeners']['monomer_counts'][0]) == list)
    assert(type(data['listeners']['monomer_counts'][1]) == list)


# python ecoli/processes/listeners/monomer_counts.py
if __name__ == '__main__':
    test_monomer_counts_listener()
