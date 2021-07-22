"""
=============
Cell Division
=============

"""
from typing import Any, Dict

from vivarium.core.process import Deriver


def divide_by_domain(state, **args):
    """
    divide a dictionary into two daughters based on their domain_index
    """
    daughter1 = {}
    daughter2 = {}
    for state_id, value in state.items():
        domain_index = value['domain_index']
        if domain_index == 1:
            daughter1[state_id] = value
        elif domain_index == 2:
            daughter2[state_id] = value
            daughter2[state_id]['domain_index'] = 1
    return [daughter1, daughter2]


def daughter_phylogeny_id(mother_id):
    return [
        str(mother_id) + '0',
        str(mother_id) + '1']


class Division(Deriver):
    """ Division Process """
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

            return {
                'agents': {
                    '_divide': {
                        'mother': self.agent_id,
                        'daughters': daughter_updates}}}
        return {}
