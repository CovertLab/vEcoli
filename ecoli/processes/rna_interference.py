
"""
================
RNA Interference
================
Treats sRNA-mRNA binding as complexation events that create duplexes and release
bound ribosomes. Decreases ompF translation during micF overexpression.
"""
import numpy as np

from vivarium.core.process import Step
from vivarium.core.composition import simulate_process

from ecoli.library.schema import arrays_from, dict_value_schema, bulk_schema
from ecoli.processes.registries import topology_registry

# Register default topology for this process, associating it with process name
NAME = 'ecoli-rna-interference'
TOPOLOGY = {
    "subunits": ("bulk",),
    "bulk_RNAs": ("bulk",),
    "RNAs": ("unique", "RNA"),
    "active_ribosome": ("unique", "active_ribosome"),
}
topology_registry.register(NAME, TOPOLOGY)

class RnaInterference(Step):
    name = NAME
    topology = TOPOLOGY
    defaults = {
        'srna_ids': [],
        'target_tu_ids': [],
        'binding_probs': [],
        'ribosome30S': 'ribosome30S',
        'ribosome50S': 'ribosome50S',
        'duplex_ids': [],
        'seed': 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        # Parameters are lists such that the nth element
        # of each list are grouped (e.g. the 1st sRNA ID
        # binds to the first sRNA target with the 1st
        # binding probability)
        self.srna_ids = self.parameters['srna_ids']
        self.target_tu_ids = self.parameters['target_tu_ids']
        self.binding_probs = self.parameters['binding_probs']
        self.ribosome30S = self.parameters['ribosome30S']
        self.ribosome50S = self.parameters['ribosome50S']
        self.duplex_ids = list(self.parameters['duplex_ids'])
        self.bulk_rna_ids = self.srna_ids + self.duplex_ids
        self.random_state = np.random.RandomState(seed = self.parameters['seed'])
    
    def ports_schema(self):
        return {
            'subunits': bulk_schema([self.ribosome30S, self.ribosome50S]),
            'bulk_RNAs': bulk_schema([rna_id for rna_id in self.bulk_rna_ids]),
            'active_ribosome': dict_value_schema('active_ribosome'),
            'RNAs': dict_value_schema('RNAs'),
        }
        
    def next_update(self, timestep, states):
        update = {
            'bulk_RNAs': {},
            'RNAs': {
                '_delete': [],
                '_add': []},
            'active_ribosome': {
                '_delete': []},
            'subunits': {
                self.ribosome30S: 0,
                self.ribosome50S: 0
            }}
        
        TU_index, can_translate, is_full_transcript = (
            arrays_from(
                states['RNAs'].values(),
                ['TU_index', 'can_translate', 
                'is_full_transcript'])
            )
        rna_indexes = np.array(list(states['RNAs'].keys()))
        
        mRNA_index, = arrays_from(
            states['active_ribosome'].values(),
            ['mRNA_index'])
        ribosome_indexes = np.array(list(states['active_ribosome'].keys()))
        
        for srna_id, mrna_index, binding_prob, duplex_id in zip(
            self.srna_ids, self.target_tu_ids, self.binding_probs,
            self.duplex_ids
        ):
            # Get mask for complete sRNAs
            srna_count = states['bulk_RNAs'][srna_id]
            if srna_count == 0:
                continue
            
            # Get mask for translatable, complete target mRNAs
            # TODO: Is it worth it to account for duplexing of incomplete mRNAs?
            mrna_mask = np.logical_and(TU_index == mrna_index, can_translate)
            mrna_mask = np.logical_and(mrna_mask, is_full_transcript)
            n_mrna = mrna_mask.sum()
            if n_mrna == 0:
                continue
            
            # Each sRNA has probability binding_prob of binding a target mRNA
            n_duplexed = np.min([self.random_state.binomial(srna_count, binding_prob),
                                 mrna_mask.sum()])
            
            # Choose n_duplexed mRNAs and sRNAs randomly to delete
            mrna_to_delete = self.random_state.choice(
                size=n_duplexed, a=np.where(mrna_mask)[0], replace=False).tolist()
            update['RNAs']['_delete'] += list(rna_indexes[mrna_to_delete])
            if srna_id not in update['bulk_RNAs']:
                update['bulk_RNAs'][srna_id] = 0
            update['bulk_RNAs'][srna_id] -= n_duplexed
            
            # Dissociate ribosomes attached to new duplexes
            ribosomes_to_delete = list(ribosome_indexes[
                np.isin(mRNA_index.astype(str), rna_indexes[mrna_to_delete])
            ])
            update['active_ribosome']['_delete'] += ribosomes_to_delete
            update['subunits'][self.ribosome30S] += len(ribosomes_to_delete)
            update['subunits'][self.ribosome50S] += len(ribosomes_to_delete)
            
            # Ensure that additional sRNAs cannot bind to mRNAs that have
            # already been duplexed
            remainder_mask = np.ones(TU_index.size).astype('int')
            remainder_mask[mrna_to_delete] = False
            TU_index = TU_index[remainder_mask]
            can_translate = can_translate[remainder_mask]
            is_full_transcript = is_full_transcript[remainder_mask]
            rna_indexes = rna_indexes[remainder_mask]
            
            # Add new RNA duplexes
            if duplex_id not in update['bulk_RNAs']:
                update['bulk_RNAs'][duplex_id] = 0
            update['bulk_RNAs'][duplex_id] += n_duplexed
        
        return update

def test_rna_interference():
    test_config = {
        'time_step': 2,
        'ribosome30S': 'CPLX0-3953[c]',
        'ribosome50S': 'CPLX0-3962[c]',
        'srna_ids': ['MICF-RNA[c]'],
        'target_tu_ids': [661],
        'target_ids': ['EG10671_RNA[c]'],
        'duplex_ids': ['micF-ompF[c]'],
        'binding_probs': [0.5]
        }

    rna_inter = RnaInterference(test_config)

    initial_state = {
        'subunits': {
            'CPLX0-3953[c]': 100,
            'CPLX0-3962[c]': 100
        },
        'bulk_RNAs': {
            'MICF-RNA[c]': 4,
            'micF-ompF[c]': 0,    
        },
        'active_ribosome': {
            '1': {'mRNA_index': 1},
            '2': {'mRNA_index': 2},
            '3': {'mRNA_index': 1}
        },
        'RNAs': {
            '1': {'TU_index': 661, 'can_translate': True, 
                  'is_full_transcript': True, 'transcript_length': 1089},
            '2': {'TU_index': 661, 'can_translate': True, 
                  'is_full_transcript': True, 'transcript_length': 1089},
            '3': {'TU_index': 661, 'can_translate': True,
                  'is_full_transcript': True, 'transcript_length': 1089},
            '4': {'TU_index': 661, 'can_translate': True,
                  'is_full_transcript': True, 'transcript_length': 1089},
        }
    }

    settings = {
        'total_time': 4,
        'initial_state': initial_state,
        'return_raw_data': True}
    data = simulate_process(rna_inter, settings)

    return data, test_config

def validate(data, config):
    srna_counts = np.zeros((len(config['srna_ids']), len(data)))
    target_counts = np.zeros((len(config['target_tu_ids']), len(data)))
    duplex_counts = np.zeros((len(config['duplex_ids']), len(data)))
    ribosome_counts = np.zeros(len(data))
    for ticker, (timestep, state,) in enumerate(data.items()):
        TU_index, = arrays_from(
            state['RNAs'].values(), ['TU_index']) 
        for i, srna_id in enumerate(config['srna_ids']):
            srna_counts[i, ticker] = state['bulk_RNAs'][srna_id]
        for i, target_tu_id in enumerate(config['target_tu_ids']):
            target_counts[i, ticker] = np.count_nonzero(TU_index == target_tu_id)
        for i, duplex_id in enumerate(config['duplex_ids']):
            duplex_counts[i, ticker] = state['bulk_RNAs'][duplex_id]
        ribosome_counts[ticker] = len(state['active_ribosome'])
    print(dict(zip(config['srna_ids'], srna_counts)))
    print(dict(zip(config['target_ids'], target_counts)))
    print(dict(zip(config['duplex_ids'], duplex_counts)))
    print(f'Active ribosomes: {ribosome_counts}')

def main():
    data, config = test_rna_interference()
    validate(data, config)

if __name__ == '__main__':
    main()
