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
        'time_step': 1,
    }

    def __init__(self, config):
        super().__init__(config)

    def generate_processes(self, config):

        receptor_config = config['receptor']
        motor_config = config['motor']
        time_step = config['time_step']

        ligand_id = config['ligand_id']
        initial_ligand = config['initial_ligand']
        receptor_config.update({
            'ligand_id': ligand_id,
            'initial_ligand': initial_ligand,
            'time_step': time_step})

        motor_config.update({
            'time_step': time_step})

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

def get_chemotaxis_static_envrionment_composite(initial_conc=1, static_field_params=None):
    environment_port = ('external',)
    ligand_id = 'MeAsp'
    time_step = 0.01

    # make the compartment
    chemotaxis_config = {
        'external_path': (environment_port,),
        'ligand_id': ligand_id,
        'initial_ligand': initial_conc,
        'time_step': time_step}
    chemotaxis_composite = ChemotaxisMinimal(chemotaxis_config).generate(path=('agents', '1'))


    static_field = StaticField(static_field_params)

    parameters = {
        'bounds': static_field_params['bounds'],
        'time_step': time_step
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

    return chemotaxis_composite

def run_in_static_field(initial_conc=1):
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)

    # add a static field process
    bounds = [1000, 1000]
    # static_field_params = get_exponential_config(molecule='MeAsp', bounds=bounds, scale=1000)  # TODO -- put in parameters!
    static_field_params = {
        'bounds': bounds,
        'molecules': ['MeAsp'],
        # 'gradient': {
        #     'type': 'linear',
        #     'molecules': {
        #         'MeAsp': {
        #             'center': [0,0],
        #             'slope': 0,
        #             'base': initial_conc
        #         }
        #     }
        # }
        'gradient': {
            'type': 'exponential',
            'molecules': {
                'MeAsp': {
                    'center': [0, 0],
                    'scale': 1,
                    'base': 0.05
                }
            }
        }
    }
    chemotaxis_composite = get_chemotaxis_static_envrionment_composite(initial_conc=initial_conc,
                                                                       static_field_params=static_field_params)

    # Get initial state
    initial_state = chemotaxis_composite.initial_state()

    # put the composite in an engine and run it
    sim = Engine(processes=chemotaxis_composite.processes,
                 topology=chemotaxis_composite.topology,
                 initial_state=initial_state)
    sim.update(50, global_time_precision=5)

    # get the data
    data = sim.emitter.get_data()
    field = make_field(config=static_field_params)
    field = field.T

    print(pf(data))

    # Begin analysis of data
    run_distances = []
    run_durations = []
    tumble_duration = []
    velocities = []
    CheY_run = []
    CheY_tumble = []

    thrusts = []
    torques = []

    isRunning = False

    startTime = None
    startLoc = [0,0]
    # 1 = tumble, -1 = run
    for t,datum in data.items():
        thrusts.append(datum['agents']['1']['boundary']['thrust'])
        torques.append(datum['agents']['1']['boundary']['torque'])
        # Is it in run or tumble?
        motile_state = datum['agents']['1']['cell']['motile_state']
        location = datum['agents']['1']['boundary']['location']
        if motile_state == -1 and not isRunning:
            # Just started running
            if startTime:
                tumble_duration.append(t - startTime)
                CheY_tumble.append(datum['agents']['1']['cell']['CheY_P'] / 0.3)
            startTime = t
            startLoc = location
            isRunning = True
        elif motile_state == -1 and isRunning:
            pass
        elif motile_state == 1 and isRunning:
            if startTime:
                run_durations.append(t - startTime)
                distance = ((location[0] - startLoc[0])**2 + (location[1] - startLoc[1])**2) **0.5
                velocities.append(distance / (t - startTime))
                run_distances.append(distance)
                CheY_run.append(datum['agents']['1']['cell']['CheY_P']/0.3)
            isRunning = False
            startTime = t
            startLoc = location
        elif motile_state == 1 and not isRunning:
            pass

    fig, ax = plt.subplots(4, 2)

    # Experimental time = 0.86s
    ax[0,0].hist(run_durations)
    ax[0,0].set_title('Run Durations')
    ax[0,0].axvline(x=0.86, linestyle='--', color='red')
    ax[0,0].axvline(np.mean(run_durations), color='k', linestyle='--')
    #plt.savefig('out/run_durations.png')

    # Experimental time = 0.14s
    ax[0,1].hist(tumble_duration)
    ax[0,1].set_title('Tumble Durations')
    ax[0,1].axvline(x=0.14, linestyle='--', color='red')
    ax[0,1].axvline(np.mean(tumble_duration), color='k', linestyle='--')
    #plt.savefig('out/tumble_durations.png')

    ax[1,0].hist(run_distances)
    ax[1,0].set_title('Run Distances')
    #plt.savefig('out/run_distances.png')

    # Experimental velocity = 14.2 micrometers/sec
    ax[1,1].hist(velocities)
    ax[1,1].set_title('Velocities')
    ax[1,1].axvline(x=21.2, linestyle='--', color='red')
    ax[1,1].axvline(np.mean(velocities), color='k', linestyle='--')

    # CheY = P-CheY / 0.3 (Response regulator output in bacterial chemotaxis)
    ax[2,0].scatter(x=CheY_run, y=run_durations)
    ax[2,0].set_title('Run Durations vs. CheY')

    ax[2,1].scatter(x=CheY_tumble, y=tumble_duration)
    ax[2,1].set_title('Tumble Durations vs. CheY')

    ax[3,0].plot(thrusts)
    ax[3,0].set_title('Thrust')
    ax[3,0].axhline(0.57, linestyle='--', color='red')
    ax[3,0].axhline(np.mean(thrusts), color='k', linestyle='--')

    ax[3,1].plot(torques)
    ax[3,1].set_title('Torque')
    ax[3,1].axhline((5*(10**-19)), linestyle='--', color='red')
    ax[3,1].axhline(np.mean(torques), color='k', linestyle='--')

    plt.tight_layout()
    fig.savefig('out/chemotaxis.png')

    # times = data.keys()
    # list_thrust = []

    # modify to get x and y list
    location = np.zeros([2, len(data.keys())])
    for i, timepoint in enumerate(data.keys()):
        # list_thrust.append(data[timepoint]['agents']['1']['boundary']['thrust'])
        location[:, i] = data[timepoint]['agents']['1']['boundary']['location']

    print(location)

    shape = field.shape
    fig2, ax = plt.subplots(1,1)
    im = ax.imshow(field, origin='lower', cmap='Greys', extent=[0,shape[1],0,shape[0]])
    cbar = plt.colorbar(im)
    cbar.set_label('concentration')
    ax.plot(location[0], location[1])
    fig2.savefig('out/location.png')


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

    #print(pf(data))



def scan_chemotaxis(initial_conc=1):
    # add a static field process
    bounds = [1000, 1000]
    # static_field_params = get_exponential_config(molecule='MeAsp', bounds=bounds, scale=1000)  # TODO -- put in parameters!
    static_field_params = {
        'bounds': bounds,
        'molecules': ['MeAsp'],
        # 'gradient': {
        #     'type': 'linear',
        #     'molecules': {
        #         'MeAsp': {
        #             'center': [0,0],
        #             'slope': 0,
        #             'base': initial_conc
        #         }
        #     }
        # }
        'gradient': {
            'type': 'exponential',
            'molecules': {
                'MeAsp': {
                    'center': [0, 0],
                    'scale': 1,
                    'base': 0.05
                }
            }
        }
    }
    chemotaxis_composite = get_chemotaxis_static_envrionment_composite(initial_conc=initial_conc,
                                                                       static_field_params=static_field_params)
    parameter_sets = {
        '0': {
            'static_field': {
                'gradient': {
                    'type': 'exponential',
                    'molecules': {
                        'MeAsp': {
                            'center': [0, 0],
                            'scale': 1,
                            'base': 0.1
                        }
                    }
                }
            }
        },
        '1': {
            'static_field': {
                'gradient': {
                    'type': 'exponential',
                    'molecules': {
                        'MeAsp': {
                            'center': [0, 0],
                            'scale': 1,
                            'base': 1
                        }
                    }
                }
            }
        }
    }
    metrics = [distance_from_center] #TODO: Add chemotaxis metric
    scanner = Scan(simulator_class=chemotaxis_composite, parameter_sets=parameter_sets,
                   total_time=30, metrics=metrics)
    scanner.run_scan()

def distance_from_center(paramter_sets):
    center = paramter_sets['static_field']['gradient']['molecules']['MeAsp']['center']
    return 0


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
    '1': run_in_static_field,
    '2': scan_chemotaxis
}

# python ecoli/composites/chemotaxis_minimal.py -n [exp #]
if __name__ == '__main__':
    run_library_cli(library)
