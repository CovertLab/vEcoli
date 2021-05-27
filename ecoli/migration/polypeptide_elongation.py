from vivarium.core.experiment import Experiment, pf
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.migration.migration_utils import run_ecoli_process
from ecoli.composites.ecoli_master import get_state_from_file
from ecoli.processes.polypeptide_elongation import PolypeptideElongation

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

# topology from ecoli_master
PE_TOPOLOGY = {
    'environment': ('environment',),
    'listeners': ('listeners',),
    'active_ribosome': ('unique', 'active_ribosome'),
    'molecules': ('bulk',),
    'monomers': ('bulk',),
    'amino_acids': ('bulk',),
    'ppgpp_reaction_metabolites': ('bulk',),
    'uncharged_trna': ('bulk',),
    'charged_trna': ('bulk',),
    'charging_molecules': ('bulk',),
    'synthetases': ('bulk',),
    'subunits': ('bulk',),
    'polypeptide_elongation': ('process_state', 'polypeptide_elongation')}


def test_polypeptide_elongation_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_polypeptide_elongation_config()
    polypeptide_elongation_process = PolypeptideElongation(config)

    # run the process and get an update
    actual_update = run_ecoli_process(polypeptide_elongation_process, PE_TOPOLOGY, total_time=2)

    print(f"molecules: {actual_update['molecules']}")
    print(f"amino_acids: {pf(actual_update['amino_acids'])}")
    import ipdb; ipdb.set_trace()


def run_polypeptide_elongation():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_polypeptide_elongation_config()
    polypeptide_elongation_process = PolypeptideElongation(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t0.json')

    polypeptide_elongation_composite = polypeptide_elongation_process.generate()
    experiment = Experiment({
        'processes': polypeptide_elongation_composite['processes'],
        'topology': {polypeptide_elongation_process.name: PE_TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_polypeptide_elongation_migration()
    # run_polypeptide_elongation()
