
"""
================
RNA Interference
================
Treats sRNA-mRNA binding as complexation events that create duplexes and release
bound ribosomes. Decreases ompF translation during micF overexpression.
"""
import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process
from vivarium.plots.simulation_output import plot_variables
from vivarium.core.emitter import timeseries_from_data

from ecoli.library.schema import (
    create_unique_indexes, arrays_from, arrays_to, dict_value_schema, 
    add_elements, bulk_schema)
from ecoli.processes.registries import topology_registry

# Register default topology for this process, associating it with process name
NAME = 'ecoli-rna-interference'
TOPOLOGY = {
    "subunits": ("bulk",),
    "RNAs": ("unique", "RNA"),
    "active_ribosome": ("unique", "active_ribosome"),
}
topology_registry.register(NAME, TOPOLOGY)

class RnaInterference(Process):
    name = NAME
    topology = TOPOLOGY
    defaults = {
        'srna_tu_ids': [],
        'target_tu_ids': [],
        'binding_probs': [],
        'ribosome30S': 'ribosome30S',
        'ribosome50S': 'ribosome50S',
        'duplex_tu_ids': [],
        'seed': 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        # Parameters are lists such that the nth element
        # of each list are grouped (e.g. the 1st sRNA ID
        # binds to the first sRNA target with the 1st
        # binding probability)
        self.srna_tu_ids = self.parameters['srna_tu_ids']
        self.target_tu_ids = self.parameters['target_tu_ids']
        self.binding_probs = self.parameters['binding_probs']
        self.ribosome30S = self.parameters['ribosome30S']
        self.ribosome50S = self.parameters['ribosome50S']
        self.duplex_tu_ids = self.parameters['duplex_tu_ids']
        self.duplex_lengths = self.parameters['duplex_lengths']
        self.random_state = np.random.RandomState(seed = self.parameters['seed'])
    
    def ports_schema(self):
        return {
            'subunits': bulk_schema([self.ribosome30S, self.ribosome50S]),
            'active_ribosome': dict_value_schema('active_ribosome'),
            'RNAs': dict_value_schema('RNAs'),
        }
        
    def next_update(self, timestep, states):
        update = {
            'RNAs': {
                '_delete': [],
                '_add': []},
            'active_ribosome': {
                '_delete': []},
            'subunits': {
                self.ribosome30S: 0,
                self.ribosome50S: 0
            }}
        
        TU_index, can_translate, is_full_transcript = arrays_from(
            states['RNAs'].values(),
            ['TU_index', 'can_translate', 'is_full_transcript'])
        rna_indexes = np.array(list(states['RNAs'].keys()))
        
        mRNA_index, = arrays_from(
            states['active_ribosome'].values(),
            ['mRNA_index'])
        ribosome_indexes = np.array(list(states['active_ribosome'].keys()))
        
        for srna_index, mrna_index, binding_prob, duplex_index, duplex_length in zip(
            self.srna_tu_ids, self.target_tu_ids, self.binding_probs,
            self.duplex_tu_ids, self.duplex_lengths):
            # Get mask for complete sRNAs
            srna_mask = np.logical_and(TU_index == srna_index, is_full_transcript)
            n_srna = srna_mask.sum()
            if n_srna == 0:
                continue
            
            # Get mask for translatable target mRNAs
            mrna_mask = np.logical_and(TU_index == mrna_index, can_translate)
            n_mrna = mrna_mask.sum()
            if n_mrna == 0:
                continue
            
            # Each sRNA has probability binding_prob of binding a target mRNA
            n_duplexed = np.min([self.random_state.binomial(n_srna, binding_prob),
                                 mrna_mask.sum()])
            
            # Choose n_duplexed mRNAs and sRNAs randomly to delete
            mrna_to_delete = self.random_state.choice(
                size=n_duplexed, a=np.where(mrna_mask)[0], replace=False)
            srna_to_delete = self.random_state.choice(
                size=n_duplexed, a=np.where(srna_mask)[0], replace=False)
            to_delete = list(mrna_to_delete) + list(srna_to_delete)
            update['RNAs']['_delete'] += list(rna_indexes[to_delete])
            
            # Dissociate ribosomes attached to new duplexes
            ribosomes_to_delete = list(ribosome_indexes[
                np.isin(mRNA_index.astype(str), rna_indexes[to_delete])
            ])
            update['active_ribosome']['_delete'] += ribosomes_to_delete
            update['subunits'][self.ribosome30S] += len(ribosomes_to_delete)
            update['subunits'][self.ribosome50S] += len(ribosomes_to_delete)
            
            # Ensure that additional sRNAs cannot bind to mRNAs that have
            # already been duplexed
            remainder_mask = np.ones(TU_index.size).astype('int')
            remainder_mask[to_delete] = False
            TU_index = TU_index[remainder_mask]
            can_translate = can_translate[remainder_mask]
            is_full_transcript = is_full_transcript[remainder_mask]
            rna_indexes = rna_indexes[remainder_mask]
            
            # Add new RNA duplexes
            rna_indices = create_unique_indexes(
                n_duplexed, self.random_state)
            new_RNAs = arrays_to(
                n_duplexed, {
                    'unique_index': rna_indices,
                    'TU_index': [duplex_index]*n_duplexed,
                    'transcript_length': [duplex_length]*n_duplexed,
                    'is_mRNA': [True]*n_duplexed,
                    'is_full_transcript': [True]*n_duplexed,
                    'can_translate': [False]*n_duplexed,
                    'RNAP_index': [-1]*n_duplexed})

            update['RNAs']['_add'] += add_elements(new_RNAs, 'unique_index')['_add']
        
        return update

def test_rna_interference():
    test_config = {
        'time_step': 2,
        'ribosome30S': 'CPLX0-3953[c]',
        'ribosome50S': 'CPLX0-3962[c]',
        'srna_ids': ['MICF-RNA[c]'],
        'target_ids': ['EG10671_RNA[c]'],
        'srna_tu_ids': [2493],
        'target_tu_ids': [661],
        'duplex_ids': ['micF-ompF[c]'],
        'duplex_deg_rates': [0.00135911],
        'duplex_lengths': [1182],
        'duplex_ACGU': [[306, 273, 280, 323]],
        'duplex_mw': [378762.459],
        'duplex_km': [0.00034204],
        'duplex_tu_ids': [4687],
        'binding_probs': [0.5]
        }

    rna_inter = RnaInterference(test_config)

    initial_state = {
        'subunits': {
            'CPLX0-3953[c]': 100,
            'CPLX0-3962[c]': 100
        },
        'active_ribosome': {
            '1': {'mRNA_index': 1},
            '2': {'mRNA_index': 2},
            '3': {'mRNA_index': 1}
        },
        'RNAs': {
            '1': {'TU_index': 661, 'can_translate': True, 'is_full_transcript': True},
            '2': {'TU_index': 661, 'can_translate': True, 'is_full_transcript': True},
            '3': {'TU_index': 661, 'can_translate': True, 'is_full_transcript': True},
            '4': {'TU_index': 661, 'can_translate': True, 'is_full_transcript': False},
            '5': {'TU_index': 2493, 'can_translate': False, 'is_full_transcript': True},
            '6': {'TU_index': 2493, 'can_translate': False, 'is_full_transcript': True},
            '7': {'TU_index': 2493, 'can_translate': False, 'is_full_transcript': True},
        }
    }

    settings = {
        'total_time': 4,
        'initial_state': initial_state,
        'return_raw_data': True}
    data = simulate_process(rna_inter, settings)

    return data, test_config

def validate(data, config):
    srna_counts = np.zeros((len(config['srna_tu_ids']), len(data)))
    target_counts = np.zeros((len(config['target_tu_ids']), len(data)))
    duplex_counts = np.zeros((len(config['duplex_tu_ids']), len(data)))
    for ticker, (timestep, state,) in enumerate(data.items()):
        TU_index, = arrays_from(
            state['RNAs'].values(), ['TU_index']) 
        for i, srna_tu_id in enumerate(config['srna_tu_ids']):
            srna_counts[i, ticker] += np.count_nonzero(TU_index == srna_tu_id)
        for i, target_tu_id in enumerate(config['target_tu_ids']):
            target_counts[i, ticker] += np.count_nonzero(TU_index == target_tu_id)
        for i, duplex_tu_id in enumerate(config['duplex_tu_ids']):
            duplex_counts[i, ticker] += np.count_nonzero(TU_index == duplex_tu_id)
    print(dict(zip(config['srna_ids'], srna_counts)))
    print(dict(zip(config['target_ids'], target_counts)))
    print(dict(zip(config['duplex_ids'], duplex_counts)))

def main():
    data, config = test_rna_interference()
    validate(data, config)

if __name__ == '__main__':
    main()
