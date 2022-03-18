"""
==========================
Minimal E. coli Chemotaxis
==========================
"""

import os

from vivarium.core.composer import Composer
from vivarium.core.composition import COMPOSITE_OUT_DIR, simulate_composite
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.core.control import run_library_cli
from vivarium.core.engine import Engine, pf

# processes
from ecoli.processes.chemotaxis.chemoreceptor_cluster import (
    ReceptorCluster,
    get_exponential_random_timeline,
)
from ecoli.processes.chemotaxis.coarse_motor import MotorActivity
from ecoli.processes.environment.static_field import StaticField, get_exponential_config


NAME = 'chemotaxis_minimal'


class ChemotaxisMinimal(Composer):
    """ Chemotaxis Minimal Composite

     A chemotactic cell with only receptor and coarse motor processes.
     """

    name = NAME
    defaults = {
        'ligand_id': 'MeAsp',
        'initial_ligand': 0.1,
        'boundary_path': ('boundary',),
        'receptor': {},
        'motor': {},
    }

    def __init__(self, config):
        super().__init__(config)

    def generate_processes(self, config):

        receptor_config = config['receptor']
        motor_config = config['motor']

        ligand_id = config['ligand_id']
        initial_ligand = config['initial_ligand']
        receptor_config.update({
            'ligand_id': ligand_id,
            'initial_ligand': initial_ligand})

        # declare the processes
        receptor = ReceptorCluster(receptor_config)
        motor = MotorActivity(motor_config)

        return {
            'receptor': receptor,
            'motor': motor}

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        external_path = boundary_path + ('external',)
        return {
            'receptor': {
                'external': external_path,
                'internal': ('cell',)},
            'motor': {
                'external': boundary_path,
                'internal': ('cell',)}}



def test_chemotaxis_minimal(total_time=10):
    environment_port = ('external',)
    ligand_id = 'MeAsp'
    initial_conc = 0
    time_step = 0.1

    # make the compartment
    compartment_config = {
        'external_path': (environment_port,),
        'ligand_id': ligand_id,
        'initial_ligand': initial_conc}
    composite = ChemotaxisMinimal(compartment_config).generate()

    # configure timeline
    exponential_random_config = {
        'ligand': ligand_id,
        'environment_port': environment_port,
        'time': total_time,
        'timestep': time_step,
        'initial_conc': initial_conc,
        'base': 1 + 4e-4,
        'speed': 14,
    }
    timeline = get_exponential_random_timeline(exponential_random_config)

    # run experiment
    experiment_settings = {
        'timeline': {
            'timeline': timeline,
            'paths': {'external': ('boundary', 'external')}},
        'timestep': time_step,
        'total_time': total_time}
    timeseries = simulate_composite(composite, experiment_settings)

    return timeseries

def run_in_static_field():
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    environment_port = ('external',)
    ligand_id = 'MeAsp'
    initial_conc = 0
    time_step = 0.1

    # make the compartment
    chemotaxis_config = {
        'external_path': (environment_port,),
        'ligand_id': ligand_id,
        'initial_ligand': initial_conc}
    chemotaxis_composite = ChemotaxisMinimal(chemotaxis_config).generate(path=('agents', '1'))

    # add a static field process
    bounds = [1000, 1000]
    static_field_params = get_exponential_config(molecule='MeAsp', bounds=bounds, scale=1000)  # TODO -- put in parameters!
    static_field = StaticField(static_field_params)

    # TODO create multibody
    from ecoli.processes.environment.multibody_physics import Multibody
    parameters = {
        'bounds': static_field_params['bounds'],
        'time_step': 0.1
    }
    multibody = Multibody(parameters)

    # merge
    chemotaxis_composite.merge(
        processes={
            'static_field': static_field,
            'multibody': multibody,
        },
        topology={
            'static_field': {
                'agents': ('agents',)},
            'multibody': {
                'agents': ('agents',)
            }
        }
    )

    # put the composite in an engine and run it
    sim = Engine(processes=chemotaxis_composite.processes,
                 topology=chemotaxis_composite.topology)
    sim.update(300)

    # get the data
    data = sim.emitter.get_data()
    field = make_field(config=static_field_params)
    field = field.T

    print(pf(data))


    # times = data.keys()
    # list_thrust = []

    # modify to get x and y list
    location = np.zeros([2, len(data.keys())])
    for i, timepoint in enumerate(data.keys()):
        # list_thrust.append(data[timepoint]['agents']['1']['boundary']['thrust'])
        location[:, i] = data[timepoint]['agents']['1']['boundary']['location']

    print(location)

    shape = field.shape
    im = plt.imshow(field, origin='lower', cmap='Greys', extent=[0,shape[1],0,shape[0]])
    cbar = plt.colorbar(im)
    cbar.set_label('concentration')
    plt.plot(location[0], location[1])
    plt.savefig('out/location.png')




    # np.save('out/location.npy', location)

    #seaborn lineplot

    data = sim.emitter.get_timeseries()

    # plot
    plot_settings = {
        'max_rows': 20,
        'remove_zeros': True,
        }
    plot_simulation_output(
        data,
        plot_settings,
        out_dir,
        'chemotaxis_timeseries')

    # print(pf(data))



def main():
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # run the composite
    timeseries = test_chemotaxis_minimal(total_time=60)

    # plot
    plot_settings = {
        'max_rows': 20,
        'remove_zeros': True,
        'overlay': {
            'reactions': 'flux'},
        'skip_ports': ['prior_state', 'null', 'global']}
    plot_simulation_output(
        timeseries,
        plot_settings,
        out_dir,
        'exponential_timeline')


library = {
    '0': main,
    '1': run_in_static_field
}

# python ecoli/composites/chemotaxis_minimal.py -n [exp #]
if __name__ == '__main__':
    run_library_cli(library)
