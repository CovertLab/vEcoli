from vivarium.library.units import units
from vivarium.core.process import Step
from vivarium.core.engine import Engine
from vivarium.plots.simulation_output import plot_variables
from vivarium.processes.timeline import TimelineProcess
from vivarium.core.emitter import timeseries_from_data

from ecoli.library.schema import bulk_schema

CONV_UNITS = 1 / units.mM


class ConcToCounts(Step):
    '''Convert concentrations to counts (requires Pint units)'''

    name = 'conc_to_counts'
    defaults = {
        'molecules_to_convert': ['antibiotic'],
        'initial_state': {
            'internal': {
                'antibiotic': 0 * units.mM,
            },
            'bulk': {
                'antibiotic': 0
            },
            'volume_global': {
                'volume': 0 * units.fL
            }
        },
        'n_avogadro': 6.0221409e23 / units.mol,  # 1/mol
    }
    
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.nAvogadro = self.parameters['n_avogadro']

    def ports_schema(self):
        schema = {
            'internal': {
                molecule: {
                    '_default': 0.0 * units.mM,
                    '_divider': 'set',
                    '_emit': True,
                }
                for molecule in self.parameters['molecules_to_convert']
            },
            'bulk': bulk_schema(
                self.parameters['molecules_to_convert'],
                updater='set'
            ),
            'volume_global': {
                'volume': {
                    '_default': 0 * units.fL,
                    '_divider': 'split',
                },
            }
        }
        return schema

    def initial_state(self, config=None):
        config = config or {}
        parameters = self.parameters
        parameters.update(config)
        
        initial_state = {
            'internal': {
                molecule: 0 * units.mM
                for molecule in self.parameters['molecules_to_convert']
            },
            'bulk': {
                molecule: 0
                for molecule in self.parameters['molecules_to_convert']
            },
            'volume_global': {
                'volume': 0 * units.fL,
            },
        }
        # No deep_merge: keys in `parameters` that are not in 
        # `initial_state` are ignored
        for port, port_state in initial_state.items():
            for variable, value, in port_state.items():
                port_state[variable] = parameters[
                    'initial_state'].get(port, {}).get(
                    variable, value)
        return initial_state

    def next_update(self, timestep, states):
        cellVolume = states['volume_global']['volume']
        
        # Calculate state values
        molar_to_counts = self.nAvogadro * cellVolume * 1e-18
        update = {'bulk': {}}
        for molecule, conc in states['internal'].items():
            update['bulk'][molecule] = int((conc * molar_to_counts).magnitude)

        return update

def main():
    sim_time = 10

    initial_state = {}
    initial_state['boundary'] = {'internal': {'antibiotic': 0 * units.mM}}
    initial_state['listeners'] = {'mass': {'volume': 1.352695 * units.fL}}

    conv_process = ConcToCounts()

    timeline = []
    for i in range(5):
        timeline.append(
            (i * 2, {
                ('internal', 'antibiotic'): initial_state['boundary'][
                    'internal']['antibiotic'] + (i + 1) * units.uM,
            })
        )
    timeline_params = {
        'time_step': 2.0,
        'timeline': timeline,
    }
    timeline_process = TimelineProcess(timeline_params)

    sim = Engine(processes={'conc_to_counts': conv_process,
                            'timeline': timeline_process},
                 topology={
                     'conc_to_counts': {
                        'internal': ('boundary', 'internal',),
                        'bulk': ('bulk',),
                        'volume_global': ('listeners', 'mass',),
                     },
                     'timeline': {
                        'global': ('global',),  # The global time is read here
                        'internal': ('boundary', 'internal',),  # This port is based on the declared timeline
                     }
                 },
                 initial_state=initial_state)
    sim.update(sim_time)
    timeseries_data = timeseries_from_data(sim.emitter.get_data())
    antibiotic_str_to_float = []
    for string in timeseries_data['boundary']['internal']['antibiotic']:
        antibiotic_str_to_float.append(units(string).magnitude)
    timeseries_data['boundary']['internal']['antibiotic'] = antibiotic_str_to_float
    print(timeseries_data)
    plot_variables(timeseries_data, [('bulk', 'antibiotic'), 
                                     ('boundary', 'internal', 'antibiotic')],
                   out_dir='out', filename='conc_to_counts')

if __name__ == '__main__':
    main()
