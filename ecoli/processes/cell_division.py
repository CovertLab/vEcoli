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
                daughter_updates.append({
                    'key': daughter_id,
                    'processes': composer['processes'],
                    'steps': composer['steps'],
                    'flow': composer['flow'],
                    'topology': composer['topology'],
                    'initial_state': {}})

            print(f'DIVIDE! MOTHER {self.agent_id} -> DAUGHTERS {daughter_ids}')

            return {
                'agents': {
                    '_divide': {
                        'mother': self.agent_id,
                        'daughters': daughter_updates}}}
        return {}


def main():
    from vivarium.core.emitter import (
        data_from_database, get_local_client, timeseries_from_data)
    data, conf = data_from_database('789cf4a8-7805-11ec-9575-1e00312eb299',
                                    get_local_client("localhost", "27017", "simulations"))
    data = timeseries_from_data(data)

    sa_sum = 0
    ompc_sum = 0
    ompf_sum = 0
    sa_len = len(data['agents']['0']['boundary']['surface_area'])
    ompc_len = len(data['agents']['0']['bulk']['EG10670-MONOMER[o]'])
    ompf_len = len(data['agents']['0']['bulk']['EG10671-MONOMER[o]'])
    for i in range(sa_len):
        sa_sum += data['agents']['0']['boundary']['surface_area'][i]
    for i in range(ompc_len):
        ompc_sum += data['agents']['0']['bulk']['EG10670-MONOMER[o]'][i]
    for i in range(ompf_len):
        ompf_sum += data['agents']['0']['bulk']['EG10671-MONOMER[o]'][i]
    sa_average = sa_sum / sa_len
    ompc_average = ompc_sum / ompc_len
    ompf_average = ompf_sum / ompf_len

    import ipdb;
    ipdb.set_trace()


if __name__ == '__main__':
    main()
