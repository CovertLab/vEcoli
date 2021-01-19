
from vivarium.core.composition import simulate_compartment_in_experiment
from ecoli.composites.ecoli_master import Ecoli, get_state_from_file

def report_process_update():
    pass




def run_migration_check():
    ecoli = Ecoli({'agent_id': '1'})
    initial_state = get_state_from_file()


    settings = {
        'timestep': 1,
        'total_time': 2,
        'initial_state': initial_state}

    output = simulate_compartment_in_experiment(ecoli, settings)


    # separate data by port
    bulk = output['bulk']
    unique = output['unique']
    listeners = output['listeners']
    process_state = output['process_state']
    environment = output['environment']

    import ipdb;
    ipdb.set_trace()

    return output



if __name__ == '__main__':
    run_migration_check()
