"""
=============
Cell Division
=============
"""
from typing import Any, Dict

import binascii
import numpy as np
from vivarium.core.process import Step

from ecoli.library.sim_data import RAND_MAX
from wholecell.utils import units

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
        self.composer_config = self.parameters['composer_config']
        self.random_state = np.random.RandomState(
            seed=self.parameters['seed'])

        self.division_mass_multiplier = 1
        if self.parameters['division_threshold'] == 'massDistribution':
            division_random_seed = binascii.crc32(b'CellDivision',
                self.parameters['seed']) & 0xffffffff
            division_random_state = np.random.RandomState(
                seed=division_random_seed)
            self.division_mass_multiplier = division_random_state.normal(
                loc=1.0, scale=0.1)
        self.dry_mass_inc_dict = self.parameters['dry_mass_inc_dict']
            
    def ports_schema(self):
        return {
            'division_variable': {},
            'full_chromosome': {},
            'agents': {
                '*': {}},
            'media_id': {},
            'division_threshold': {
                '_default': self.parameters['division_threshold'],
                '_updater': 'set',
                '_divider': {'divider': 'set_value',
                    'config': {'value': self.parameters['division_threshold']}}
            }
        }

    def next_update(self, timestep, states):
        # Figure out division threshold at first timestep if
        # using massDistribution setting
        if states['division_threshold'] == 'massDistribution':
            current_media_id = states['media_id']
            return {'division_threshold': (states['division_variable'] + 
                self.dry_mass_inc_dict[current_media_id].asNumber(
                    units.fg) * self.division_mass_multiplier)}

        division_variable = states['division_variable']

        if (division_variable >= self.parameters['threshold']) and (
            states['full_chromosome']['_entryState'].sum() >= 2
        ):
            daughter_ids = self.parameters['daughter_ids_function'](self.agent_id)
            daughter_updates = []
            for daughter_id in daughter_ids:
                config = dict(self.composer_config)
                config['agent_id'] = daughter_id
                config['seed'] = self.random_state.randint(0, RAND_MAX)
                # Regenerate composite to avoid unforeseen shared states
                composite = self.composer(config).generate()
                # Get shared process instances for partitioned processes
                process_states = {
                    process.parameters['process'].name: (
                        process.parameters['process'],)
                    for process in composite.processes.values()
                    if 'process' in process.parameters
                }
                initial_state = {'process': process_states}
                daughter_updates.append({
                    'key': daughter_id,
                    'processes': composite['processes'],
                    'steps': composite['steps'],
                    'flow': composite['flow'],
                    'topology': composite['topology'],
                    'initial_state': initial_state})

            print(f'DIVIDE! MOTHER {self.agent_id} -> DAUGHTERS {daughter_ids}')

            return {
                'agents': {
                    '_divide': {
                        'mother': self.agent_id,
                        'daughters': daughter_updates}}}
        return {}
