"""
=============
Cell Division
=============
"""
from typing import Any, Dict

import numpy as np
from vivarium.core.process import Step

from ecoli.library.sim_data import RAND_MAX

NAME = 'ecoli-cell-division'


def daughter_phylogeny_id(mother_id):
    return [
        str(mother_id) + '0',
        str(mother_id) + '1']


class Division(Step):
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

        if variable >= self.parameters['threshold']:
            daughter_ids = self.parameters['daughter_ids_function'](self.agent_id)
            daughter_updates = []
            for daughter_id in daughter_ids:
                composer = self.composer.generate({
                    'agent_id': daughter_id,
                    'seed': self.random_state.randint(0, RAND_MAX)
                })
                # Get shared process instances for partitioned processes
                process_states = {
                    process.parameters['process'].name: (process.parameters['process'],)
                    for process in composer.processes.values()
                    if 'process' in process.parameters
                }
                initial_state = {'process': process_states}
                daughter_updates.append({
                    'key': daughter_id,
                    'processes': composer['processes'],
                    'steps': composer['steps'],
                    'flow': composer['flow'],
                    'topology': composer['topology'],
                    'initial_state': initial_state})

            print(f'DIVIDE! MOTHER {self.agent_id} -> DAUGHTERS {daughter_ids}')

            return {
                'agents': {
                    '_divide': {
                        'mother': self.agent_id,
                        'daughters': daughter_updates}}}
        return {}
