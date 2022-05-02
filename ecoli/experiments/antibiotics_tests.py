from vivarium.core.control import run_library_cli
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


def test_antibiotics_nitrocefin():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'antibiotics_nitrocefin.json')
    sim.emitter = 'timeseries'
    sim.total_time = 2
    sim.run()
    data = sim.query()


def test_antibiotics_tetracycline_cephaloridine():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'antibiotics_tetracycline_cephaloridine.json')
    sim.emitter = 'timeseries'
    sim.total_time = 2
    sim.run()
    data = sim.query()


def test_lysis_rxn_dff_environment():
    # sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'spatial.json')
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'lysis_environment.json')
    sim.emitter = 'timeseries'
    sim.total_time = 2
    sim.run()
    data = sim.query()

    import ipdb; ipdb.set_trace()


library = {
    '0': test_antibiotics_nitrocefin,
    '1': test_antibiotics_tetracycline_cephaloridine,
    '2': test_lysis_rxn_dff_environment,
}

# python ecoli/experiments/antibiotics_tests.py -n library_id
if __name__ == "__main__":
    run_library_cli(library)
