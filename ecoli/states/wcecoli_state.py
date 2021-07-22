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


def get_state_from_file(path='data/wcecoli_t0.json'):

    states = load_states(path)

    initial_state = {
        'environment': {
            'media_id': 'minimal',
            # TODO(Ryan): pull in environmental amino acid levels
            'amino_acids': {},
            'exchange_data': {
                # Does this have to be a set?
                # Lists can be json serialized, sets cannot
                'unconstrained': [
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
                    'AMMONIUM[c]'],
                'constrained': {
                    'GLC[p]': 20.0 * units.mmol / (units.g * units.h)}},
            'external_concentrations': states['environment']},
        # TODO(Eran): deal with mass
        # add mw property to bulk and unique molecules
        # and include any "submass" attributes from unique molecules
        'listeners': states['listeners'],
        'bulk': states['bulk'],
        'unique': states['unique'],
        'process_state': {
            'polypeptide_elongation': {}}}

    return initial_state