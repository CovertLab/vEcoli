from scipy.constants import N_A

from vivarium.core.process import Step
from vivarium.library.units import units, Quantity


AVOGADRO = N_A / units.mol


class ConcentrationsDeriver(Step):

    defaults = {
        'variables': []
    }
    name = 'concentrations_deriver'

    def ports_schema(self):
        schema = {
            'counts': {
                variable: {
                    '_default': 0,  # In counts
                }
                for variable in self.parameters['variables']
            },
            'concentrations': {
                variable: {
                    '_default': 0 * units.mM,
                    '_updater': 'set',
                }
                for variable in self.parameters['variables']
            },
            'volume': {
                '_default': 0 * units.fL,
            },
        }
        return schema

    def next_update(self, timestep, states):
        volume = states['volume']
        assert isinstance(volume, Quantity)
        concentrations = {
            var: (
                count * units.count / AVOGADRO / volume
            ).to(units.millimolar)
            for var, count in states['counts'].items()
        }
        return {
            'concentrations': concentrations
        }
