import json
import numpy as np
from scipy.stats import chisquare
from vivarium.core.composition import process_in_experiment

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH, get_state_from_file

from ecoli.processes.protein_degradation import ProteinDegradation

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)


def test_protein_degradation():
    time_step = 2
    total_time = 2

    config = load_sim_data.get_protein_degradation_config(time_step=time_step)
    prot_deg_process = ProteinDegradation(config)
    initial_state = get_state_from_file(path='data/wcecoli_t10.json')
    experiment = process_in_experiment(prot_deg_process, initial_state=initial_state)

    process_paths = experiment.process_paths
    path = list(process_paths.keys())[0]
    process = list(process_paths.values())[0]
    update_tuple = experiment.process_update(path, process, total_time)
    update, process_topology, state = update_tuple

    # This actual update comes from the process, and can be compared to wcEcoli process json
    actual_update = update.get()
    import ipdb;
    ipdb.set_trace()


    # experiment.update(total_time)
    # data = experiment.emitter.get_data()
    #
    # initial = data[0.0]
    # final = data[total_time]
    # import ipdb; ipdb.set_trace()



    with open("data/prot_deg_update_t2.json") as f:
        wc_data = json.load(f)

        #get comparable update from protein_degradation
        d_proteins = np.random.normal(size=len(wc_data["proteins_to_degrade"]))
        d_metabolites = np.random.normal(size=len(wc_data["metabolite_update"]))

        print(chisquare(wc_data["proteins_to_degrade"], d_proteins))
        print(chisquare(wc_data["metabolite_update"], d_metabolites))


if __name__ == "__main__":
    test_protein_degradation()
