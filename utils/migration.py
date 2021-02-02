import json
import numpy as np
from scipy.stats import chisquare
from vivarium.core.composition import simulate_compartment_in_experiment
from ecoli.composites.ecoli_master import Ecoli, get_state_from_file


def report_process_update():
    pass

def test_protein_degradation():
    with open("data/prot_deg_update_t2.json") as f:
        wc_data = json.load(f)

        #get comparable update from protein_degradation
        d_proteins = np.random.normal(size=len(wc_data["proteins_to_degrade"]))
        d_metabolites = np.random.normal(size=len(wc_data["metabolite_update"]))

        print(chisquare(wc_data["proteins_to_degrade"], d_proteins))
        print(chisquare(wc_data["metabolite_update"], d_metabolites))


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

    test_protein_degradation()

    import ipdb;
    ipdb.set_trace()

    return output


if __name__ == '__main__':
    run_migration_check()
