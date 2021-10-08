from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from vivarium.core.engine import pf
import json
import os
import matplotlib.pyplot as plt
from migration.plots import qqplot
from migration.migration_utils import *

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

PI_TOPOLOGY = PolypeptideInitiation.topology


def test_polypeptide_initiation_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_polypeptide_initiation_config()
    polypeptide_initiation_process = PolypeptideInitiation(config)
    total_time = 2
    initial_times = [0, 10, 100]
    for initial_time in initial_times:
        with open(f"data/polypeptide_initiation_update_t{initial_time + total_time}.json") as f:
            wc_update = json.load(f)
        # run the process and get an update
        initial_state = get_state_from_file(
            path=f'data/wcecoli_t{initial_time}.json')
        states, _ = get_process_state(polypeptide_initiation_process, PI_TOPOLOGY, initial_state)
        # This value is normally modified by polypeptide elongation
        states['listeners']['ribosome_data']['effective_elongation_rate'] = wc_update[
            'listeners']['ribosome_data'].pop('ribosomeElongationRate')
        actual_update = polypeptide_initiation_process.next_update(total_time, states)
        plots(actual_update, wc_update, total_time + initial_time)
        assertions(actual_update, wc_update)

def test_polypeptide_initiation():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_polypeptide_initiation_config()
    polypeptide_initiation_process = PolypeptideInitiation(config)

    polypeptide_initiation_composite = polypeptide_initiation_process.generate()

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t0.json')

    experiment = Engine(**{
        'processes': polypeptide_initiation_composite['processes'],
        'topology': {polypeptide_initiation_process.name: PI_TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)
    data = experiment.emitter.get_data()
    print(pf(data))

    return data


def plots(actual_update, expected_update, time):
    os.makedirs("out/migration/polypeptide_initiation/", exist_ok=True)

    def unpack(update):
        ar_peptides, ar_pos_on_mrna = [], []
        for i in range(len(update['active_ribosome']['_add'])):
            ar_peptides.append(update['active_ribosome']['_add'][i]['state']['peptide_length'])
            ar_pos_on_mrna.append(update['active_ribosome']['_add'][i]['state']['pos_on_mRNA'])
        prob_translation_per_transcript = update['listeners']['ribosome_data']['prob_translation_per_transcript']
        return ar_peptides, ar_pos_on_mrna, prob_translation_per_transcript

    # Unpack updates
    (ar_peptides, ar_pos_on_mrna, prob_translation_per_transcript) = unpack(actual_update)
    (wc_ar_peptides, wc_ar_pos_on_mrna, wc_prob_translation_per_transcript) = unpack(expected_update)

    # Plots ============================================================================

    plt.subplot(2, 2, 1)
    qqplot(ar_peptides, wc_ar_peptides)
    plt.ylabel('wcEcoli')
    plt.xlabel('Vivarium')
    plt.title('Q-Q Plot of ar_peptides')

    plt.subplot(2, 2, 2)
    qqplot(ar_pos_on_mrna, wc_ar_pos_on_mrna)
    plt.ylabel('wcEcoli')
    plt.xlabel('Vivarium')
    plt.title('Q-Q Plot of ar_pos_on_mrna')

    plt.subplot(2, 2, 3)
    qqplot(prob_translation_per_transcript, wc_prob_translation_per_transcript)
    plt.ylabel('wcEcoli')
    plt.xlabel('Vivarium')
    plt.title('Q-Q Plot of prob_translation_per_transcript')

    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(f"out/migration/polypeptide_initiation/polypeptide_initiation_t{time}.png")

def assertions(actual_update, expected_update):

    def unpack(update):
        ar_peptides, ar_pos_on_mrna, subunits = [], [], []
        for i in range(len(update['active_ribosome']['_add'])):
            ar_peptides.append(update['active_ribosome']['_add'][i]['state']['peptide_length'])
            ar_pos_on_mrna.append(update['active_ribosome']['_add'][i]['state']['pos_on_mRNA'])
        for key in update['subunits'].keys():
            subunits.append(update['subunits'][key])
        ribosomes_initialized = update['listeners']['ribosome_data']['ribosomes_initialized']
        prob_translation_per_transcript = update['listeners']['ribosome_data']['prob_translation_per_transcript']

        return ar_peptides, ar_pos_on_mrna, subunits, ribosomes_initialized, prob_translation_per_transcript

    # Unpack updates
    (ar_peptides, ar_pos_on_mrna, subunits, ribosomes_initialized,
     prob_translation_per_transcript) = unpack(actual_update)

    (wc_ar_peptides, wc_ar_pos_on_mrna, wc_subunits, wc_ribosomes_initialized,
     wc_prob_translation_per_transcript) = unpack(expected_update)

    # Assertions fail by initial_time = 1000 due to stochasticity
    assert array_equal(np.array(ar_peptides), wc_ar_peptides)
    assert array_equal(np.array(ar_pos_on_mrna), wc_ar_pos_on_mrna)
    assert array_equal(np.array(subunits), wc_subunits)
    assert scalar_equal(np.array(ribosomes_initialized), wc_ribosomes_initialized)
    assert array_equal(np.array(prob_translation_per_transcript), wc_prob_translation_per_transcript)


def run_polypeptide_initiation():
    test_polypeptide_initiation_migration()


if __name__ == "__main__":
    run_polypeptide_initiation()
