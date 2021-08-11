import json
import numpy as np

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


def get_state_from_file(path='data/wcecoli_t0.json'):

    states = load_states(path)

    initial_state = {
        'environment': {
            'media_id': 'minimal',
            # TODO(Ryan): pull in environmental amino acid levels
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
                    'Pi[p]',
                    'NI+2[p]',
                    'WATER[p]',
                    'AMMONIUM[c]'},
                'constrained': {
                    'GLC[p]': 20.0 * units.mmol / (units.g * units.h)}},
            'external_concentrations': states['environment']},
        # TODO(Eran): deal with mass
        # add mw property to bulk and unique molecules
        # and include any "submass" attributes from unique molecules
        'listeners': states['listeners'],
        'bulk': states['bulk'],
        'unique': {},
        'process_state': {
            'polypeptide_elongation': {}}}
    
    massDiffs = {'massDiff_rRNA': 0, 'massDiff_tRNA': 1, 'massDiff_mRNA': 2, 
                 'massDiff_miscRNA': 3, 'massDiff_nonspecific_RNA': 4, 
                 'massDiff_protein': 5, 'massDiff_metabolite': 6, 
                 'massDiff_water': 7, 'massDiff_DNA': 8}
    for mol_type, molecules in states['unique'].items():
        initial_state['unique'].update({mol_type: {}})
        for molecule, values in molecules.items():
            initial_state['unique'][mol_type][molecule] = {'submass' : np.zeros(len(massDiffs))}
            for key, value in values.items():
                if key in massDiffs:
                    initial_state['unique'][mol_type][molecule]['submass'][massDiffs[key]] = value
                else:
                    initial_state['unique'][mol_type][molecule][key] = value

    return initial_state