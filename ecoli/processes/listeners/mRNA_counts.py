"""
====================
mRNA Counts Listener
====================
"""

import numpy as np
from ecoli.library.schema import arrays_from, dict_value_schema
from vivarium.core.process import Step

from ecoli.processes.registries import topology_registry


NAME = 'mRNA_counts_listener'
topology_registry.register(
    NAME,
    {
        "listeners": ("listeners",),
        "RNAs" : ("unique", "RNA"),
    }
)

class mRNACounts(Step):
    """
    Listener for the counts of each mRNA species.
    """
    name = NAME

    defaults = {
        'rna_ids': [],
        'mrna_indexes': [],
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Get IDs and indexes of all mRNAs
        self.all_RNA_ids = self.parameters['rna_ids']
        self.mRNA_indexes = self.parameters['mrna_indexes']

    def ports_schema(self):
        return {
            'listeners': {
                'mRNA_counts': {
                    '_default': [],
                    '_updater': 'set',
                    '_emit': True}
            },
            'RNAs': dict_value_schema('RNAs')
        }

    def next_update(self, timestep, states):
        # Get attributes of mRNAs
        tu_indexes, can_translate = arrays_from(
            states['RNAs'].values(), ['TU_index', 'can_translate'])

        # Get counts of all mRNAs
        mrna_counts = np.bincount(
            tu_indexes[can_translate],
            minlength=len(self.all_RNA_ids))[self.mRNA_indexes]

        return {
            'listeners': {
                'mRNA_counts': mrna_counts
            }
        }


def test_mrna_counts_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    sim = EcoliSim.from_file()
    sim.total_time = 2
    sim.raw_output = False
    data = sim.run()
    assert(type(data['listeners']['mRNA_counts'][0]) == list)
    assert(type(data['listeners']['mRNA_counts'][1]) == list)


if __name__ == '__main__':
    test_mrna_counts_listener()
