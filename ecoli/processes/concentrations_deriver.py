from scipy.constants import N_A

from vivarium.core.process import Step
from vivarium.library.units import units, Quantity


AVOGADRO = N_A / units.mol


def get_delta(before, after):
    # assuming before and after have the same keys
    return {
        key: after[key] - before_value
        for key, before_value in before.items()}


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
                    '_updater': 'accumulate',
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
        new_concentrations = {
            var: (
                count * units.count / AVOGADRO / volume
            ).to(units.millimolar)
            for var, count in states['counts'].items()
        }
        update = {
            'concentrations': get_delta(
                states['concentrations'],
                new_concentrations)
        }
        return update