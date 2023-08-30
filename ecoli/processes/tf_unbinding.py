"""
TfUnbinding
Unbind transcription factors from DNA to allow signaling processes before
binding back to DNA.
"""

import numpy as np

from vivarium.core.process import Step

from ecoli.processes.registries import topology_registry
from ecoli.library.schema import bulk_name_to_idx, attrs, numpy_schema

# Register default topology for this process, associating it with process name
NAME = 'ecoli-tf-unbinding'
TOPOLOGY = {
    "bulk": ("bulk",),
    "promoters": ("unique", "promoter",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "first_update": ("first_update", "tf_unbinding"),
}
topology_registry.register(NAME, TOPOLOGY)

class TfUnbinding(Step):
    """ TfUnbinding """

    name = NAME
    topology = TOPOLOGY

    defaults = {
        'time_step': 1,
        'emit_unique': False
    }

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.tf_ids = self.parameters['tf_ids']
        self.submass_indices = self.parameters['submass_indices']
        self.active_tf_masses = self.parameters['active_tf_masses']

        # Numpy indices for bulk molecules
        self.active_tf_idx = None

    def ports_schema(self):
        return {
            'bulk': numpy_schema('bulk'),
            'promoters': numpy_schema('promoters',
                emit=self.parameters['emit_unique']),
                
            'global_time': {'_default': 0},
            'timestep': {'_default': self.parameters['time_step']},

            'first_update': {
                '_default': True,
                '_updater': 'set',
                '_divider': {'divider': 'set_value',
                    'config': {'value': True}}},
        }
    
    def update_condition(self, timestep, states):
        return (states['global_time'] % states['timestep']) == 0

    def next_update(self, timestep, states):
        # At t=0, convert all strings to indices
        if self.active_tf_idx is None:
            self.active_tf_idx = bulk_name_to_idx(
                [tf + '[c]' for tf in self.tf_ids], states['bulk']['id'])
        
        if states['first_update']:
            return {'first_update': False}

        # Get attributes of all promoters
        bound_TF, = attrs(states['promoters'], ['bound_TF'])
        # If there are no promoters, return immediately
        if len(bound_TF) == 0:
            return {}

        # Calculate number of bound TFs for each TF prior to changes
        n_bound_TF = bound_TF.sum(axis=0)
        
        update = {
            # Free all DNA-bound TFs into free active TFs
            'bulk': [(self.active_tf_idx, n_bound_TF)],
            'promoters': {
                # Reset bound_TF attribute of promoters
                'set': {
                    'bound_TF': np.zeros_like(bound_TF)
                }
            }
        }

        # Add mass_diffs array to promoter submass
        mass_diffs = bound_TF @ -self.active_tf_masses
        for submass, idx in self.submass_indices.items():
            update['promoters']['set'][submass] = attrs(states['promoters'],
                [submass])[0] + mass_diffs[:, idx]
        
        return update
