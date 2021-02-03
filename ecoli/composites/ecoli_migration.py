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
    process = process_paths[path]
    update_tuple = experiment.process_update(path, process, total_time)
    update, process_topology, state = update_tuple

    # This actual update comes from the process, and can be compared to wcEcoli process json
    actual_update = update.get()

    d_proteins = actual_update['proteins']
    d_metabolites = actual_update['metabolites']

    ''' METHOD 2
    experiment.update(total_time)
    data = experiment.emitter.get_data()

    initial = data[0.0]
    final = data[total_time]

    d_proteins = {id : final['proteins'][id] - initial['proteins'][id] for id in initial['proteins'].keys()}
    d_metabolites = {id : final['metabolites'][id] - initial['metabolites'][id] for id in initial['metabolites'].keys()}
    '''

    with open("data/prot_deg_update_t2.json") as f:
        wc_data = json.load(f)

        assert len(d_proteins) == len(wc_data['proteins_to_degrade']) == len(wc_data['protein_ids']), \
            (f"Mismatch in lengths: vivarium-ecoli protein update has length {len(d_proteins)}\n"
             f"while wcecoli has {len(wc_data['protein_ids'])} proteins with {len(wc_data['proteins_to_degrade'])} values.")
        assert len(d_metabolites) == len(wc_data['metabolite_update']) == len(wc_data['metabolite_ids']), \
            (f"Mismatch in lengths: vivarium-ecoli metabolite update has length {len(d_metabolites)}\n"
             f"while wcecoli has {len(wc_data['metabolite_ids'])} metabolites with {len(wc_data['metabolite_update'])} values.")
        assert set(d_proteins.keys()) == set(wc_data['protein_ids']), \
            "Mismatch between protein ids in vivarium-ecoli and wcEcoli."
        assert set(d_metabolites.keys()) == set(wc_data['metabolite_ids']), \
            "Mismatch between metabolite ids in vivarium-ecoli and wcEcoli."

        # TODO: chisquare is not really the best here, need to look into this
        threshold = 0.05
        assert chisquare(wc_data["proteins_to_degrade"],
                         [d_proteins[id] for id in wc_data["protein_ids"]]).pvalue > threshold, \
            f"Numbers of proteins degraded are significantly different between vivarium-ecoli and wcEcoli."

        assert chisquare(wc_data["metabolite_update"],
                         [d_metabolites[id] for id in wc_data["metabolite_ids"]]).pvalue > threshold, \
            f"Metabolite updates are significantly different between vivarium-ecoli and wcEcoli."


if __name__ == "__main__":
    test_protein_degradation()
