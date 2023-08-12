"""
=====================
RnaSynthProb Listener
=====================
"""

import numpy as np
from ecoli.library.schema import numpy_schema, listener_schema, attrs
from vivarium.core.process import Step

from ecoli.processes.registries import topology_registry


NAME = 'rna_synth_prob_listener'
TOPOLOGY = {
    "rna_synth_prob": ("listeners", "rna_synth_prob"),
    "promoters": ("unique", "promoter"),
    "genes": ("unique", "gene"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}
topology_registry.register(
    NAME, TOPOLOGY
)

class RnaSynthProb(Step):
    """
    Listener for additional RNA synthesis data.
    """
    name = NAME
    topology = TOPOLOGY

    defaults = {
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.n_TU = self.parameters['n_TU']
        self.n_TF = self.parameters['n_TF']
        self.n_cistron = self.parameters['n_cistron']
        self.cistron_tu_mapping_matrix = self.parameters[
            'cistron_tu_mapping_matrix']

    def ports_schema(self):
        return {
            'rna_synth_prob': listener_schema({
                'promoter_copy_number': [],
                'gene_copy_number': [],
                'target_rna_synth_prob': np.zeros(self.n_TU, np.float64),
                'actual_rna_synth_prob': np.zeros(self.n_TU, np.float64),
                'tu_is_overcrowded': [],
                'actual_rna_synth_prob_per_cistron': [],
                'target_rna_synth_prob_per_cistron': [],
                'expected_rna_init_per_cistron': [],
                'pPromoterBound': 'tf_ids',
                'nPromoterBound': 'tf_ids',
                'nActualBound': 'tf_ids',
                'n_available_promoters': 'tf_ids',
                'n_bound_TF_per_TU': np.zeros((self.n_TU, self.n_TF), np.int16),
                'n_bound_TF_per_cistron': [],
                'total_rna_init': 0
            }),
            'promoters': numpy_schema('promoters'),
            'genes': numpy_schema('genes'),
            'global_time': {'_default': 0},
            'timestep': {'_default': self.parameters['time_step']},
        }
    
    def update_condition(self, timestep, states):
        return (states['global_time'] % states['timestep']) == 0

    def next_update(self, timestep, states):
        TU_indexes, all_coordinates, all_domains, bound_TFs = attrs(
			states['promoters'],
            ['TU_index', 'coordinates', 'domain_index', 'bound_TF'])
        bound_promoter_indexes, TF_indexes = np.where(bound_TFs)
        cistron_indexes, = attrs(states['genes'], ['cistron_index'])

        actual_rna_synth_prob_per_cistron = self.cistron_tu_mapping_matrix.dot(
            states['rna_synth_prob']['actual_rna_synth_prob']
        )
        if actual_rna_synth_prob_per_cistron.sum() != 0:
            actual_rna_synth_prob_per_cistron = (
                actual_rna_synth_prob_per_cistron / 
                actual_rna_synth_prob_per_cistron.sum())
        target_rna_synth_prob_per_cistron = self.cistron_tu_mapping_matrix.dot(
            states['rna_synth_prob']['target_rna_synth_prob']
        )
        if target_rna_synth_prob_per_cistron.sum() != 0:
            target_rna_synth_prob_per_cistron = (
                target_rna_synth_prob_per_cistron /
                target_rna_synth_prob_per_cistron.sum()
            )
        
        return {
            'rna_synth_prob': {
                'promoter_copy_number': np.bincount(TU_indexes,
                                                    minlength=self.n_TU),
                'gene_copy_number': np.bincount(cistron_indexes,
                                                minlength=self.n_cistron),
                'bound_TF_indexes': TF_indexes,
                'bound_TF_coordinates': all_coordinates[bound_promoter_indexes],
                'bound_TF_domains': all_domains[bound_promoter_indexes],
                'expected_rna_init_per_cistron': (
                    actual_rna_synth_prob_per_cistron
                    * states['rna_synth_prob']['total_rna_init']),
                'actual_rna_synth_prob_per_cistron': \
                    actual_rna_synth_prob_per_cistron,
                'target_rna_synth_prob_per_cistron': \
                    target_rna_synth_prob_per_cistron,
                'n_bound_TF_per_cistron': self.cistron_tu_mapping_matrix.dot(
                    states['rna_synth_prob']['n_bound_TF_per_TU']
                    ).astype(np.int16).T
            }
        }
