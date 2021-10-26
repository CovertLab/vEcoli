import numpy as np
from vivarium.core.registry import (
    divider_registry,
    updater_registry,
)

from ecoli.processes.cell_division import (
    divide_by_domain,
    divide_RNAs_by_domain,
    divide_unique,
)

from ecoli.library.registry import (
    divide_binomial,
    dict_value_updater,
    make_dict_value_updater,
)


UNIQUE_DEFAULTS = {
    'active_ribosome': {
        'protein_index': 0,
        'peptide_length': 0,
        'mRNA_index': 0,
        'unique_index': 0,
        'pos_on_mRNA': 0,
        'submass': np.zeros(9)
    },
    'full_chromosomes': {
        'domain_index': 0,
        'unique_index': 0,
        'division_time': 0,
        'has_triggered_division': 0,
        'submass': np.zeros(9)
    },
    'chromosome_domains': {
        'domain_index': 0,
        'child_domains': 0,
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'active_replisomes': {
        'domain_index': 0,
        'coordinates': 0,
        'unique_index': 0,
        'right_replichore': 0,
        'submass': np.zeros(9)
    },
    'oriCs': {
        'domain_index': 0,
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'promoters': {
        'TU_index': 0,
        'coordinates': 0,
        'domain_index': 0,
        'bound_TF': 0,
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'chromosomal_segments': {
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'DnaA_boxes': {
        'domain_index': 0,
        'coordinates': 0,
        'DnaA_bound': 0,
        'unique_index': 0,
        'submass': np.zeros(9)
    },
    'active_RNAPs': {
        'unique_index': 0,
        'domain_index': 0,
        'coordinates': 0,
        'direction': 0,
        'submass': np.zeros(9)
    },
    'RNAs': {
        'unique_index': 0,
        'TU_index': 0,
        'transcript_length': 0,
        'RNAP_index': 0,
        'is_mRNA': 0,
        'is_full_transcript': 0,
        'can_translate': 0,
        'submass': np.zeros(9)
    },
}

UNIQUE_DIVIDERS = {
    'active_ribosome': 'divide_unique',
    'full_chromosomes': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'chromosome_domains': '_',
    'active_replisomes': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'oriCs': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'promoters': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'chromosomal_segments': '_',
    'DnaA_boxes': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'active_RNAPs': {
        'divider': 'by_domain',
        'topology': {'chromosome_domain': ('..', 'chromosome_domain')}
    },
    'RNAs': {
        'divider': 'rna_by_domain',
        'topology': {'active_RNAP': ('..', 'active_RNAP',),
                     'chromosome_domain': ('..', 'chromosome_domain')}
    },
}

# register :term:`updaters`
updater_registry.register('dict_value', dict_value_updater)
for unique_mol, defaults in UNIQUE_DEFAULTS.items():
    updater_registry.register(f'{unique_mol}_updater',
                              make_dict_value_updater(defaults))

# register :term:`dividers`
divider_registry.register('binomial_ecoli', divide_binomial)
divider_registry.register('by_domain', divide_by_domain)
divider_registry.register('rna_by_domain', divide_RNAs_by_domain)
divider_registry.register('divide_unique', divide_unique)
