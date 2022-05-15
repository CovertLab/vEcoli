from copy import deepcopy
import json
import numpy as np

from wholecell.utils import units

MASSDIFFS = {'massDiff_rRNA': 0, 'massDiff_tRNA': 1, 'massDiff_mRNA': 2, 
            'massDiff_miscRNA': 3, 'massDiff_nonspecific_RNA': 4, 
            'massDiff_protein': 5, 'massDiff_metabolite': 6, 
            'massDiff_water': 7, 'massDiff_DNA': 8}


def infinitize(value):
    if value == '__INFINITY__':
        return float('inf')
    else:
        return value


def load_states(path):
    with open(path, 'r') as states_file:
        states = json.load(states_file)
    if 'agents' in states.keys():
        for agent_id in states['agents'].keys():
            states['agents'][agent_id]['environment'] = {
                key: infinitize(value)
                for key, value in states['agents'][agent_id]['environment'].items()
            }
    else:
        states['environment'] = {
            key: infinitize(value)
            for key, value in states['environment'].items()
        }
    return states


def update_unique(modify_dict, source_dict, convert_unique_id_to_string):
    for mol_type, molecules in source_dict.items():
        modify_dict.update({mol_type: {}})
        for molecule_id, values in molecules.items():
            if convert_unique_id_to_string:
                molecule_id = str(molecule_id)
            modify_dict[mol_type][molecule_id] = {
                'submass': np.zeros(len(MASSDIFFS))}
            for key, value in values.items():
                if key in MASSDIFFS:
                    modify_dict[mol_type][molecule_id]['submass'][
                        MASSDIFFS[key]] = value
                elif key in ['unique_index', 'RNAP_index', 'mRNA_index'] and convert_unique_id_to_string:
                    # convert these values to strings
                    modify_dict[mol_type][molecule_id][key] = str(value)
                else:
                    modify_dict[mol_type][molecule_id][key] = value


def colony_initial_state(states, convert_unique_id_to_string):
    states_to_return = deepcopy(states)
    for agent_id in states['agents'].keys():
        states['agents'][agent_id]['environment']['exchange_data'] = {
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
                    'GLC[p]': 20.0 * units.mmol / (units.g * units.h)}}
        states_to_return['agents'][agent_id]['unique'] = {}
        update_unique(states_to_return['agents'][agent_id]['unique'], states['agents'][agent_id]['unique'],
                      convert_unique_id_to_string)
    return states_to_return

def get_state_from_file(
        path='data/wcecoli_t0.json',
        convert_unique_id_to_string=True,
):

    states = load_states(path)

    if 'agents' in states:
        return colony_initial_state(states, convert_unique_id_to_string)

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

    update_unique(initial_state['unique'], states['unique'], convert_unique_id_to_string)

    return initial_state
