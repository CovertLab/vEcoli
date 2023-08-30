from ecoli.processes.registries import topology_registry
from vivarium.core.process import Step
from vivarium.library.units import units

NAME = 'exchange_data'
TOPOLOGY = {
    'boundary': ('boundary',),
    'environment': ('environment',),
    'first_update': ('first_update', 'exchange_data')
}
topology_registry.register(NAME, TOPOLOGY)

class ExchangeData(Step):
    """
    Update environment concentrations and metabolism exchange constraints.
    """
    name = NAME
    topology = TOPOLOGY
    defaults = {
        'exchange_data_from_concentrations': lambda _: (set(), set()),
        'environment_molecules': [],
        'saved_media': {},
        'time_step': 1,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.exchange_data_from_concentrations = self.parameters[
            'exchange_data_from_concentrations']
        self.environment_molecules = self.parameters['environment_molecules']
        self.saved_media = self.parameters['saved_media']
        
    def ports_schema(self):
        return {
            'boundary': {
                'external': {
                    '*': {'_default': 0 * units.mM}
                }
            },
            'environment': {
                'media_id': {'_default': ''},
                'exchange_data': {
                    'constrained': {'_default': {}, '_updater': 'set'},
                    'unconstrained': {'_default': set(), '_updater': 'set'}
                }
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

        # Set exchange constraints for metabolism
        exchange_data = self.exchange_data_from_concentrations(env_concs)
        unconstrained = exchange_data['importUnconstrainedExchangeMolecules']
        constrained = exchange_data['importConstrainedExchangeMolecules']
        return {
            'boundary': {
                'external': conc_update
            },
            'environment': {
                'exchange_data': {
                    'constrained': constrained,
                    'unconstrained': list(unconstrained)
                }
            }
        }
