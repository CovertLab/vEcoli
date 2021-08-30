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
                'L-SELENOCYSTEINE[c]': 100.0, 
                'L-ALPHA-ALANINE[p]': 0.8, 
                'ARG[p]': 5.2, 
                'ASN[p]': 0.4, 
                'L-ASPARTATE[p]': 0.4, 
                'CYS[p]': 0.1, 
                'GLT[p]': 0.6000000000000001, 
                'GLN[p]': 0.6200000000000001, 
                'GLY[p]': 0.8, 
                'HIS[p]': 0.2, 
                'ILE[p]': 0.4, 
                'LEU[p]': 0.8, 
                'LYS[p]': 0.4, 
                'MET[p]': 0.2, 
                'PHE[p]': 0.4, 
                'PRO[p]': 0.4, 
                'SER[p]': 10.0, 
                'THR[p]': 0.4, 
                'TRP[p]': 0.1, 
                'TYR[p]': 0.2, 
                'VAL[p]': 0.6000000000000001}
        initial_state['environment'].update({
                'L-SELENOCYSTEINE[c]': 100.0, 
                'L-ALPHA-ALANINE[p]': 0.8, 
                'ARG[p]': 5.2, 
                'ASN[p]': 0.4, 
                'L-ASPARTATE[p]': 0.4, 
                'CYS[p]': 0.1, 
                'GLT[p]': 0.6000000000000001, 
                'GLN[p]': 0.6200000000000001, 
                'GLY[p]': 0.8, 
                'HIS[p]': 0.2, 
                'ILE[p]': 0.4, 
                'LEU[p]': 0.8, 
                'LYS[p]': 0.4, 
                'MET[p]': 0.2, 
                'PHE[p]': 0.4, 
                'PRO[p]': 0.4, 
                'SER[p]': 10.0, 
                'THR[p]': 0.4, 
                'TRP[p]': 0.1, 
                'TYR[p]': 0.2, 
                'VAL[p]': 0.6000000000000001})

    return initial_state