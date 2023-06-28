"""
TfUnbinding
Unbind transcription factors from DNA to allow signaling processes before
binding back to DNA.
"""

import numpy as np

from vivarium.core.process import Step

from ecoli.processes.registries import topology_registry
from ecoli.library.schema import bulk_name_to_idx, attrs

# Register default topology for this process, associating it with process name
NAME = 'ecoli-tf-unbinding'
TOPOLOGY = {
    "bulk": ("bulk",),
    "bulk_total": ("bulk",),
    "listeners": ("listeners",)
}
topology_registry.register(NAME, TOPOLOGY)

class TfUnbinding(Step):
    """ TfUnbinding """

    name = NAME
    topology = TOPOLOGY

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.tf_ids = self.parameters['tf_ids']
        self.submass_indices = self.parameters['submass_indices']
        self.active_tf_masses = self.parameters['active_tf_masses']

        # Numpy indices for bulk molecules
        self.active_tf_idx = None

    def next_update(self, timestep, states):
        if self.active_tf_idx == None:
            self.active_tf_idx = bulk_name_to_idx(
                [tf + '[c]' for tf in self.tf_ids], states['bulk']['id'])

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
