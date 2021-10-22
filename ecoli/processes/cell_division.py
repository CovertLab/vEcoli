"""
=============
Cell Division
=============
"""
import random
from typing import Any, Dict
import numpy as np

from vivarium.core.process import Deriver

NAME = 'ecoli-cell-division'


def create_index_to_daughter(chromosome_domain):
    """
    Creates a dictionary linking domain indexes to their respective cells.
    If the index does not belong to a daughter cell, it is assigned a value of -1.
    """

    index_to_children = {}
    for domain_key in chromosome_domain:
        domain = chromosome_domain[domain_key]
        index_to_children[domain['domain_index']] = domain['child_domains']

    root_index = -1
    for root_candidate in index_to_children:
        root = True
        for domain_index_to_check in index_to_children:
            if root_candidate in index_to_children[domain_index_to_check]:
                root = False
        if root:
            root_index = root_candidate

    def get_cell_for_index(index_to_children, domain_index_to_add, root_index):
        if domain_index_to_add == root_index:  # If the root index:
            return -1
        if domain_index_to_add in index_to_children[root_index]:  # If a daughter cell index:
            return domain_index_to_add
        for domain_index in index_to_children:
            children = index_to_children[domain_index]
            if domain_index_to_add in children:
                cell = get_cell_for_index(index_to_children, domain_index, root_index)
        return cell

    index_to_daughter = {}
    for domain_index_to_add in index_to_children:
        index_to_daughter[domain_index_to_add] = get_cell_for_index(index_to_children, domain_index_to_add,
                                                                    root_index)

    return index_to_daughter


def divide_by_domain(values, state):
    """
    divide a dictionary into two daughters based on their domain_index
    """
    daughter1 = {}
    daughter2 = {}

    index_to_daughter = create_index_to_daughter(state['chromosome_domain'])
    cells = []
    for cell in index_to_daughter.values():
        if cell != -1 and cell not in cells:
            cells.append(cell)
    daughter1_index = min(cells)
    daughter2_index = max(cells)

    for state_id, value in values.items():
        domain_index = value['domain_index']
        if index_to_daughter[domain_index] == daughter1_index:
            daughter1[state_id] = value
        elif index_to_daughter[domain_index] == daughter2_index:
            daughter2[state_id] = value
    return [daughter1, daughter2]


def divide_unique(values, **args):
    daughter1 = {}
    daughter2 = {}
    for state_id, value in values.items():
        if random.choice([True, False]):
            daughter1[state_id] = value
        else:
            daughter2[state_id] = value
    return [daughter1, daughter2]


def divide_RNAs_by_domain(values, state):
    """
    divide a dictionary of unique RNAs into two daughters,
    with partial RNAs divided along with their domain index
    """
    daughter1 = {}
    daughter2 = {}
    full_transcripts = []

    index_to_daughter = create_index_to_daughter(state['chromosome_domain'])
    cells = []
    for cell in index_to_daughter.values():
        if cell != -1 and cell not in cells:
            cells.append(cell)
    daughter1_index = min(cells)
    daughter2_index = max(cells)

    # divide partial transcripts by domain_index
    for unique_id, specs in values.items():
        associated_rnap_key = str(values[unique_id]['RNAP_index'])
        if not specs['is_full_transcript']:
            domain_index = state['active_RNAP'][associated_rnap_key]['domain_index']
            if index_to_daughter[domain_index] == daughter1_index:
                daughter1[unique_id] = specs
            elif index_to_daughter[domain_index] == daughter2_index:
                daughter2[unique_id] = specs
        else:
            # save full transcripts
            full_transcripts.append(unique_id)

    # divide full transcripts binomially
    n_full_transcripts = len(full_transcripts)
    daughter1_counts = np.random.binomial(n_full_transcripts, 0.5)
    daughter1_ids = random.sample(full_transcripts, daughter1_counts)
    for unique_id in full_transcripts:
        specs = values[unique_id]
        if unique_id in daughter1_ids:
            daughter1[unique_id] = specs
        else:
            daughter2[unique_id] = specs

    return [daughter1, daughter2]


def daughter_phylogeny_id(mother_id):
    return [
        str(mother_id) + '0',
        str(mother_id) + '1']


class Division(Deriver):
    """ Division Deriver """

    name = NAME
    defaults: Dict[str, Any] = {
        'daughter_ids_function': daughter_phylogeny_id,
        'threshold': None,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # must provide a composer to generate new daughters
        self.agent_id = self.parameters['agent_id']
        self.composer = self.parameters['composer']

    def ports_schema(self):
        return {
            'variable': {},
            'agents': {
                '*': {}}}

    def next_update(self, timestep, states):
        variable = states['variable']

        print(f'division variable = {variable}')

        if variable >= self.parameters['threshold']:
            daughter_ids = self.parameters['daughter_ids_function'](self.agent_id)
            daughter_updates = []
            for daughter_id in daughter_ids:
                composer = self.composer.generate({'agent_id': daughter_id})
                daughter_updates.append({
                    'key': daughter_id,
                    'processes': composer['processes'],
                    'topology': composer['topology'],
                    'initial_state': {}})

            print(f'DIVIDE! MOTHER {self.agent_id} -> DAUGHTERS {daughter_ids}')

            return {
                'agents': {
                    '_divide': {
                        'mother': self.agent_id,
                        'daughters': daughter_updates}}}
        return {}
