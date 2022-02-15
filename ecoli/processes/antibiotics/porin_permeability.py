from ecoli.library.schema import bulk_schema
from ecoli.states.wcecoli_state import get_state_from_file
from vivarium.core.emitter import timeseries_from_data
from vivarium.core.engine import Engine
from vivarium.core.process import Step
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables
from vivarium.processes.timeline import TimelineProcess

# To calculate SA_AVERAGE, we calculated the average surface area of the model up until division.
SA_AVERAGE = 6.22200939450696
# To calculate CEPH_OMPC_CON_PERM and CEPH_OMPF_CON_PERM, we calculated the average counts of ompC and ompF
# in the model up until division and divided each by the average surface area to get the average concentrations
# of ompC and ompF. We then divided the corresponding cephaloridine permeability coefficients from Nikaido, 1983
# by these average concentrations to get our permeability per concentration constants.
CEPH_OMPC_CON_PERM = 0.003521401200296894 * 1e-5 * units.cm * units.micron * units.micron / units.sec
CEPH_OMPF_CON_PERM = 0.01195286573132685 * 1e-5 * units.cm * units.micron * units.micron / units.sec


class PorinPermeability(Step):
    defaults = {
        'porin_ids': [],
        'diffusing_molecules': [],
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.porin_ids = self.parameters['porin_ids']
        self.diffusing_molecules = self.parameters['diffusing_molecules']

    def ports_schema(self):
        return {
            'porins': bulk_schema(self.porin_ids),
            'permeabilities': {mol_id: {
                '_default': 1e-5 * units.cm / units.sec,
                '_emit': True,
                '_updater': 'set'
            } for mol_id in self.diffusing_molecules},  # Different permeability for every antibiotic
            'surface_area': {'_default': 0.0}
        }

    def next_update(self, timestep, states):
        porins = states['porins']
        surface_area = states['surface_area'] * units.micron * units.micron
        permeabilities = {}
        for molecule in self.diffusing_molecules:
            cell_permeability = 0
            for porin_id, permeability in self.diffusing_molecules[molecule].items():
                cell_permeability += (porins[porin_id] / surface_area) * permeability
            permeabilities[molecule] = cell_permeability
        return {'permeabilities': permeabilities}


def main():
    sim_time = 10

    initial_state = get_state_from_file(path='data/vivecoli_t1000.json')
    initial_state['boundary'] = {}
    initial_state['boundary']['surface_area'] = SA_AVERAGE

    porin_parameters = {
        'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
        'diffusing_molecules': {
            'cephaloridine': {
                'CPLX0-7533[o]': CEPH_OMPC_CON_PERM,
                'CPLX0-7534[o]': CEPH_OMPF_CON_PERM
            }
        },
    }
    porin_process = PorinPermeability(porin_parameters)


    timeline = []
    for i in range(5):
        timeline.append(
            (i * 2, {
                ('porins', 'CPLX0-7533[o]'): initial_state['bulk']['CPLX0-7533[o]'] + ((i + 1) * 500),
                ('porins', 'CPLX0-7534[o]'): initial_state['bulk']['CPLX0-7534[o]'] + ((i + 1) * 500),
            })
        )
    timeline_params = {
        'time_step': 2.0,
        'timeline': timeline,
    }
    timeline_process = TimelineProcess(timeline_params)

    sim = Engine(processes={'porin_permeability': porin_process,
                            'timeline': timeline_process},
                 topology={
                     'porin_permeability': {
                         'porins': ('bulk',),
                         'permeabilities': ('boundary', 'permeabilities',),
                         'surface_area': ('boundary', 'surface_area',)
                     },
                     'timeline': {
                         'global': ('global',),  # The global time is read here
                         'porins': ('bulk',),  # This port is based on the declared timeline
                     }
                 },
                 initial_state=initial_state)
    sim.update(sim_time)
    timeseries_data = timeseries_from_data(sim.emitter.get_data())
    str_to_float = []
    for string in timeseries_data['boundary']['permeabilities']['cephaloridine']:
        str_to_float.append(float(string.split()[0]))
    timeseries_data['boundary']['permeabilities']['cephaloridine'] = str_to_float
    plot_variables(timeseries_data, [('bulk', 'CPLX0-7533[o]'), ('bulk', 'CPLX0-7534[o]'),
                                     ('boundary', 'permeabilities', 'cephaloridine')],
                   out_dir='out', filename='porin_permeability_counts')


if __name__ == '__main__':
    main()
