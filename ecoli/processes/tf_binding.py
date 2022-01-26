"""
============================
Transcription Factor Binding
============================

This process models how transcription factors bind to promoters on the DNA sequence.
"""

import numpy as np

from vivarium.library.dict_utils import deep_merge

from ecoli.library.schema import arrays_from, bulk_schema, dict_value_schema, listener_schema

from wholecell.utils.random import stochasticRound
from wholecell.utils import units

from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = 'ecoli-tf-binding'
TOPOLOGY = {
        "promoters": ("unique", "promoter"),
        "active_tfs": ("bulk",),
        "active_tfs_total": ("bulk",),
        "inactive_tfs_total": ("bulk",),
        "listeners": ("listeners",)
}
topology_registry.register(NAME, TOPOLOGY)
topology_registry.register('ecoli-tf-binding-marA', TOPOLOGY)


class TfBinding(PartitionedProcess):
    """ Transcription Factor Binding PartitionedProcess """
    name = NAME
    topology = TOPOLOGY
    defaults = {
        'tf_ids': [],
        'delta_prob': {'deltaI': [], 'deltaJ': [], 'deltaV': []},
        'n_avogadro': 6.02214076e+23 / units.mol,
        'cell_density': 1100 * units.g / units.L,
        # Calculate promoter binding probability when not 0CS TF
        'p_promoter_bound_tf': lambda active, inactive: float(active) / 
                            (float(active) + float(inactive)),
        'tf_to_tf_type': {},
        'active_to_bound': {},
        'get_unbound': lambda tf: '',
        'active_to_inactive_tf': {},
        'bulk_molecule_ids': [],
        'bulk_mass_data': np.array([[]]) * units.g / units.mol,
        'seed': 0}

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Get IDs of transcription factors
        self.tf_ids = self.parameters['tf_ids']
        self.n_TF = len(self.tf_ids)

        # Build dict that maps TFs to transcription units they regulate
        self.delta_prob = self.parameters['delta_prob']
        self.TF_to_TU_idx = {}

        for i, tf in enumerate(self.tf_ids):
            self.TF_to_TU_idx[tf] = self.delta_prob['deltaI'][
                self.delta_prob['deltaJ'] == i]

        # Get total counts of transcription units
        self.n_TU = self.delta_prob['shape'][0]

        # Get constants
        self.n_avogadro = self.parameters['n_avogadro']
        self.cell_density = self.parameters['cell_density']

        # Create dictionaries and method
        self.p_promoter_bound_tf = self.parameters['p_promoter_bound_tf']
        self.tf_to_tf_type = self.parameters['tf_to_tf_type']

        self.active_to_bound = self.parameters['active_to_bound']
        self.get_unbound = self.parameters['get_unbound']
        self.active_to_inactive_tf = self.parameters['active_to_inactive_tf']

        self.active_tfs = {}
        self.inactive_tfs = {}

        for tf in self.tf_ids:
            self.active_tfs[tf] = tf + '[c]'

            if self.tf_to_tf_type[tf] == '1CS':
                if tf == self.active_to_bound[tf]:
                    self.inactive_tfs[tf] = self.get_unbound(tf + '[c]')
                else:
                    self.inactive_tfs[tf] = self.active_to_bound[tf] + '[c]'
            elif self.tf_to_tf_type[tf] == '2CS':
                self.inactive_tfs[tf] = self.active_to_inactive_tf[tf + '[c]']

        self.bulk_mass_data = self.parameters['bulk_mass_data']

        # Build array of active TF masses
        self.bulk_molecule_ids = self.parameters['bulk_molecule_ids']
        tf_indexes = [np.where(self.bulk_molecule_ids == tf_id + '[c]')[0][0] for tf_id in self.tf_ids]
        self.active_tf_masses = (self.bulk_mass_data[tf_indexes] / self.n_avogadro).asNumber(units.fg)

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)
        
    def ports_schema(self):
        return {
            'promoters': dict_value_schema('promoters'),

            'active_tfs': bulk_schema([
                self.active_tfs[tf]
                for tf in self.tf_ids]),
            
            'active_tfs_total': bulk_schema([
                self.active_tfs[tf]
                for tf in self.tf_ids],
                partition=False),

            'inactive_tfs_total': bulk_schema([
                self.inactive_tfs[tf]
                for tf in self.tf_ids
                if tf in self.inactive_tfs],
                partition=False),

            'listeners': {
                'rna_synth_prob': listener_schema({
                    'pPromoterBound': 0,
                    'nPromoterBound': 0,
                    'nActualBound': 0,
                    'n_available_promoters': 0,
                    'n_bound_TF_per_TU': 0,
                    'gene_copy_number': []})}
            }
        
    def calculate_request(self, timestep, states):
        # request all active tfs
        requests = {'active_tfs': {}}
        for tf_id in self.tf_ids:
            active_tf_key = self.active_tfs[tf_id]
            tf_count = states['active_tfs'][active_tf_key]
            requests['active_tfs'][active_tf_key] = tf_count

        return requests
        
    def evolve_state(self, timestep, states):
        # If there are no promoters, return immediately
        if not states['promoters']:
            return {}

        # Get attributes of all promoters
        TU_index, bound_TF = arrays_from(
            states['promoters'].values(),
            ['TU_index', 'bound_TF'])

        # Calculate number of bound TFs for each TF prior to changes
        n_bound_TF = bound_TF.sum(axis=0)

        # Initialize new bound_TF array
        bound_TF_new = np.zeros_like(bound_TF)

        # Create vectors for storing values
        pPromotersBound = np.zeros(self.n_TF, dtype=np.float64)
        nPromotersBound = np.zeros(self.n_TF, dtype=np.float64)
        nActualBound = np.zeros(self.n_TF, dtype=np.float64)
        n_promoters = np.zeros(self.n_TF, dtype=np.float64)
        n_bound_TF_per_TU = np.zeros((self.n_TU, self.n_TF), dtype=np.int16)

        update = {
            'active_tfs': {}}

        for tf_idx, tf_id in enumerate(self.tf_ids):
            # Free all DNA-bound transcription factors into free active
            # transcription factors
            active_tf_key = self.active_tfs[tf_id]
            tf_count = states['active_tfs'][active_tf_key]

            bound_tf_counts = n_bound_TF[tf_idx]
            update['active_tfs'][active_tf_key] = bound_tf_counts

            # Get counts of transcription factors
            active_tf_counts = (states['active_tfs_total'][active_tf_key]
                                + bound_tf_counts)
            n_available_active_tfs = tf_count + bound_tf_counts

            # Determine the number of available promoter sites
            available_promoters = np.isin(TU_index, self.TF_to_TU_idx[tf_id])
            n_available_promoters = np.count_nonzero(available_promoters)
            n_promoters[tf_idx] = n_available_promoters

            # If there are no active transcription factors to work with,
            # continue to the next transcription factor
            if n_available_active_tfs == 0:
                continue

            # Compute probability of binding the promoter
            if self.tf_to_tf_type[tf_id] == '0CS':
                pPromoterBound = 1.
            else:
                inactive_tf_counts = states['inactive_tfs_total'][self.inactive_tfs[tf_id]]
                pPromoterBound = self.p_promoter_bound_tf(
                    active_tf_counts, inactive_tf_counts)

            # Calculate the number of promoters that should be bound
            n_to_bind = int(min(stochasticRound(
                self.random_state, np.full(n_available_promoters, pPromoterBound)).sum(),
                n_available_active_tfs))

            bound_locs = np.zeros(n_available_promoters, dtype=bool)
            if n_to_bind > 0:
                # Determine randomly which DNA targets to bind based on which of
                # the following is more limiting:
                # number of promoter sites to bind, or number of active
                # transcription factors
                bound_locs[
                    self.random_state.choice(
                        n_available_promoters,
                        size=n_to_bind,
                        replace=False)
                    ] = True

                # Update count of free transcription factors
                update['active_tfs'][active_tf_key] -= bound_locs.sum()

                # Update bound_TF array
                bound_TF_new[available_promoters, tf_idx] = bound_locs

            n_bound_TF_per_TU[:, tf_idx] = np.bincount(
                TU_index[bound_TF_new[:, tf_idx]],
                minlength=self.n_TU)

            # Record values
            pPromotersBound[tf_idx] = pPromoterBound
            nPromotersBound[tf_idx] = n_to_bind
            nActualBound[tf_idx] = bound_locs.sum()

        delta_TF = bound_TF_new.astype(np.int8) - bound_TF.astype(np.int8)
        mass_diffs = delta_TF.dot(self.active_tf_masses)

        update['promoters'] = {
            key: {
                'bound_TF': bound_TF_new[index],
                'submass': states['promoters'][key]['submass'] + mass_diffs[index]}
            for index, key in enumerate(states['promoters'].keys())}

        update['listeners'] = {
            'rna_synth_prob': {
                'pPromoterBound': pPromotersBound,
                'nPromoterBound': nPromotersBound,
                'nActualBound': nActualBound,
                'n_available_promoters': n_promoters,
                'n_bound_TF_per_TU': n_bound_TF_per_TU,
                'gene_copy_number': np.bincount(TU_index, minlength=self.n_TU)},
        }

        return update
    
    def next_update(self, timestep, states):
        requests = self.calculate_request(timestep, states)
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        return update


def test_tf_binding_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    sim = EcoliSim.from_file()
    sim.total_time = 2
    sim.raw_output = False
    data = sim.run()
    assert(type(data['listeners']['rna_synth_prob']['gene_copy_number'][0]) == list)
    assert(type(data['listeners']['rna_synth_prob']['gene_copy_number'][1]) == list)


if __name__ == '__main__':
    test_tf_binding_listener()
