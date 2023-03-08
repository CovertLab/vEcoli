import numpy as np

from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


def testDefault():
    sim = EcoliSim.from_file()
    sim.total_time = 2
    sim.run()


def testAddProcess():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "test_configs/test_add_process.json")
    sim.total_time = 2
    sim.run()
    data = sim.query()

    assert "clock" in sim.ecoli.processes.keys()
    assert "global_time" in data.keys()
    assert np.array_equal(data["global_time"], [0, 2])
    assert sim.ecoli.processes["clock"].parameters["test"] == "Hello vivarium"


def testExcludeProcess():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "test_configs/test_exclude_process.json")
    sim.total_time = 2
    sim.run()
    assert "ecoli-polypeptide-elongation" not in sim.ecoli.processes.keys()
    assert "ecoli-polypeptide-elongation" not in sim.ecoli.topology.keys()
    assert "ecoli-two-component-system" not in sim.ecoli.processes.keys()
    assert "ecoli-two-component-system" not in sim.ecoli.topology.keys()


def test_merge():
    sim1 = EcoliSim.from_file()
    sim2 = EcoliSim.from_file()
    sim1.total_time = 10
    sim2.total_time = 20
    sim1.merge(sim2)

    assert sim1.total_time == 20
    assert sim2.total_time == 20


def test_export():
    sim1 = EcoliSim.from_file()
    sim1.export_json(CONFIG_DIR_PATH + "test_configs/test_export.json")


def test_load_state():
    sim1 = EcoliSim.from_file(CONFIG_DIR_PATH + "test_configs/test_save_state.json")
    sim1.run()
    sim2 = EcoliSim.from_file(CONFIG_DIR_PATH + "test_configs/test_load_state.json")
    sim2.run()


def test_initial_state_overrides():
    sim_default = EcoliSim.from_file()
    sim = EcoliSim.from_file(
        CONFIG_DIR_PATH + "test_configs/test_initial_state_overrides.json"
    )
    sim_default.build_ecoli()
    sim.build_ecoli()

    assert sim.initial_state["bulk"]["CPD-12261[p]"] == 558735
    assert "murein_state" in sim.initial_state and sim.initial_state[
        "murein_state"
    ] == {
        "shadow_murein": 0,
        "unincorporated_murein": 2234940,
        "incorporated_murein": 0,
    }
    assert sim_default.initial_state["environment"] == sim.initial_state["environment"]


def main():
    testDefault()
    testAddProcess()
    testExcludeProcess()
    test_merge()
    test_export()
    test_load_state()
    test_initial_state_overrides()


if __name__ == "__main__":
    main()
