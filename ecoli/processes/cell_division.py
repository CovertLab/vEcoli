"""
=============
Cell Division
=============
"""
from typing import Any, Dict

import numpy as np
from vivarium.core.process import Deriver

NAME = 'ecoli-cell-division'


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
        'seed': 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # must provide a composer to generate new daughters
        self.agent_id = self.parameters['agent_id']
        self.composer = self.parameters['composer']
        self.random_state = np.random.RandomState(
            seed=self.parameters['seed'])

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
                composer = self.composer.generate({
                    'agent_id': daughter_id,
                    'seed': self.random_state.randint(0, 2**31)
                })
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
