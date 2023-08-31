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
    Update metabolism exchange constraints according to environment concs.
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
        
    def ports_schema(self):
        return {
            'boundary': {
                'external': {
                    '*': {'_default': 0 * units.mM}
                }
            },
            'environment': {
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
        
        # Set exchange constraints for metabolism
        env_concs = {mol: states['boundary']['external'][mol].to('mM').magnitude
            for mol in self.environment_molecules}
        exchange_data = self.exchange_data_from_concentrations(env_concs)
        unconstrained = exchange_data['importUnconstrainedExchangeMolecules']
        constrained = exchange_data['importConstrainedExchangeMolecules']
        return {
            'environment': {
                'exchange_data': {
                    'constrained': constrained,
                    'unconstrained': list(unconstrained)
                }
            }
        }
