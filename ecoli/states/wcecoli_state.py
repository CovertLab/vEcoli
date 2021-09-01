import json

from wholecell.utils import units


def infinitize(value):
    if value == '__INFINITY__':
        return float('inf')
    else:
        return value


def load_states(path):
    with open(path, 'r') as states_file:
        states = json.load(states_file)

    states['environment'] = {
        key: infinitize(value)
        for key, value in states['environment'].items()}

    return states


def get_state_from_file(path='data/wcecoli_t0.json', aa=False):

    states = load_states(path)

    initial_state = {
        'environment': {
            'media_id': 'minimal',
            'amino_acids': {}, 
            'exchange_data': {
                'unconstrained': {
                    'CL-[p]',
                    'FE+2[p]',
                    'CO+2[p]',
                    'MG+2[p]',
                    'NA+[p]',
                    'CARBON-DIOXIDE[p]',
                    'OXYGEN-MOLECULE[p]',
                    'MN+2[p]',
                    'L-SELENOCYSTEINE[c]',
                    'K+[p]',
                    'SULFATE[p]',
                    'ZN+2[p]',
                    'CA+2[p]',
                    'PI[p]',
                    'NI+2[p]',
                    'WATER[p]',
                    'AMMONIUM[c]'},
                'constrained': {
                    'GLC[p]': 20.0 * units.mmol / (units.g * units.h)}},
            'external_concentrations': states['environment']},
        'listeners': states['listeners'],
        'bulk': states['bulk'],
        'unique': states['unique'],
        'process_state': {
            'polypeptide_elongation': {}}}

    if aa:
        initial_state['environment']['media_id'] = 'minimal_plus_amino_acids'
        initial_state['environment']['amino_acids'] = {
                'L-SELENOCYSTEINE[c]': 0, 
                'L-ALPHA-ALANINE[p]': 1, 
                'ARG[p]': 1, 
                'ASN[p]': 1, 
                'L-ASPARTATE[p]': 1, 
                'CYS[p]': 1, 
                'GLT[p]': 1, 
                'GLN[p]': 1, 
                'GLY[p]': 1, 
                'HIS[p]': 1, 
                'ILE[p]': 1, 
                'LEU[p]': 1, 
                'LYS[p]': 1, 
                'MET[p]': 1, 
                'PHE[p]': 1, 
                'PRO[p]': 1, 
                'SER[p]': 1, 
                'THR[p]': 1, 
                'TRP[p]': 1, 
                'TYR[p]': 1, 
                'VAL[p]': 1}
        initial_state['environment']['external_concentrations'].update({
                "L-ALPHA-ALANINE": 4.0,
                "ARG": 26.0,
                "ASN": 2.0,
                "L-ASPARTATE": 2.0,
                "CYS": 0.5,
                "GLT": 3.0,
                "GLN": 3.1,
                "GLY": 4.0,
                "HIS": 1.0,
                "ILE": 2.0,
                "LEU": 4.0,
                "LYS": 2.0,
                "MET": 1.0,
                "PHE": 2.0,
                "PRO": 2.0,
                "SER": 50.0,
                "THR": 2.0,
                "TRP": 0.5,
                "TYR": 1.0,
                "VAL": 3.0})

    return initial_state