from ecoli.processes.registries import topology_registry
from vivarium.core.process import Step
from vivarium.library.units import units

NAME = 'media_update'
TOPOLOGY = {
    'boundary': ('boundary',),
    'environment': ('environment',),
    'first_update': ('first_update', 'media_update')
}
topology_registry.register(NAME, TOPOLOGY)

class MediaUpdate(Step):
    """
    Update environment concentrations according to current media ID.
    """
    name = NAME
    topology = TOPOLOGY
    defaults = {
        'saved_media': {},
        'time_step': 1,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.saved_media = self.parameters['saved_media']
        
    def ports_schema(self):
        return {
            'boundary': {
                'external': {
                    '*': {'_default': 0 * units.mM}
                }
            },
            'environment': {
                'media_id': {'_default': ''}
            },
            'first_update': {
                '_default': True,
                '_updater': 'set',
                '_divider': {'divider': 'set_value',
                    'config': {'value': True}}},
        }
    
    def next_update(self, timestep, states):
        if states['first_update']:
            return {'first_update': False}

        env_concs = self.saved_media[states['environment']['media_id']]
        conc_update = {}
        # Calculate concentration delta to get from environment specified
        # by old media ID to the one specified by the current media ID
        for mol, conc in env_concs.items():
            conc_update[mol] = conc * units.mM - states['boundary']['external'][mol]

        return {
            'boundary': {
                'external': conc_update
            }
        }
