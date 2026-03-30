import numpy as np
import os
import tempfile

from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


def testDefault():
    sim = EcoliSim.from_file()
    sim.max_duration = 2
    sim.build_ecoli()
    sim.run()


def testAddProcess():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "test_configs/test_add_process.json")
    sim.max_duration = 2
    sim.build_ecoli()
    assert "clock" in sim.ecoli.processes["agents"]["0"].keys()
    sim.ecoli.processes["agents"]["0"].pop("global_clock")
    assert (
        sim.ecoli.processes["agents"]["0"]["clock"].parameters["test"]
        == "Hello vivarium"
    )

    sim.run()
    data = sim.query()
    assert "global_time" in data["agents"]["0"].keys()
    assert np.array_equal(data["agents"]["0"]["global_time"], [0, 2])


def testExcludeProcess():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "test_configs/test_exclude_process.json")
    sim.max_duration = 2
    sim.build_ecoli()
    assert "ecoli-polypeptide-initiation" not in sim.ecoli.processes.keys()
    assert "ecoli-polypeptide-initiation" not in sim.ecoli.topology.keys()
    assert "ecoli-two-component-system" not in sim.ecoli.processes.keys()
    assert "ecoli-two-component-system" not in sim.ecoli.topology.keys()
    sim.run()


def test_merge():
    sim1 = EcoliSim.from_file()
    sim2 = EcoliSim.from_file()
    sim1.max_duration = 10
    sim2.max_duration = 20
    sim1.merge(sim2)

    assert sim1.max_duration == 20
    assert sim2.max_duration == 20


def test_export():
    sim1 = EcoliSim.from_file()
    sim1.export_json(CONFIG_DIR_PATH + "test_configs/test_export.json")


def test_load_state():
    sim1 = EcoliSim.from_file(CONFIG_DIR_PATH + "test_configs/test_save_state.json")
    sim1.build_ecoli()
    sim1.run()
    sim2 = EcoliSim.from_file(CONFIG_DIR_PATH + "test_configs/test_load_state.json")
    sim2.build_ecoli()
    sim2.run()


def test_initial_state_overrides():
    sim_default = EcoliSim.from_file()
    sim = EcoliSim.from_file(
        CONFIG_DIR_PATH + "test_configs/test_initial_state_overrides.json"
    )
    sim_default.build_ecoli()
    sim.build_ecoli()

    murein_row_idx = np.where(
        sim.generated_initial_state["agents"]["0"]["bulk"]["id"] == "CPD-12261[p]"
    )[0]
    assert (
        sim.generated_initial_state["agents"]["0"]["bulk"]["count"][murein_row_idx]
        == 558735
    )
    assert sim.generated_initial_state["agents"]["0"]["murein_state"] == {
        "shadow_murein": 0,
        "unincorporated_murein": 2234940,
        "incorporated_murein": 0,
    }
    assert (
        sim_default.generated_initial_state["agents"]["0"]["environment"]
        == sim.generated_initial_state["agents"]["0"]["environment"]
    )


def test_division_time_recording():
    """
    Test that division time is accurately recorded to division_time.sh file.

    This test verifies that the division time recorded in division_time.sh
    is higher than the last emit time and all emits only have one agent.
    """
    # Create a temporary directory for daughter state outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        division_time_file = "division_time.sh"

        try:
            # Set up simulation with division enabled
            sim = EcoliSim.from_file()
            sim.config["initial_state_file"] = "vivecoli_t2526"  # Near division state
            sim.config["divide"] = True
            sim.config["generations"] = 1  # Stop after first division
            sim.config["agent_id"] = "0"
            sim.config["max_duration"] = 10  # Enough time to divide
            sim.config["daughter_outdir"] = temp_dir
            sim.config["initial_global_time"] = 2527.0
            sim.build_ecoli()

            # Run simulation until division
            try:
                sim.run()
            except SystemExit:
                # Expected - division triggers sys.exit()
                pass

            # Verify division_time.sh was created in the root directory
            assert os.path.exists(division_time_file), (
                "division_time.sh was not created"
            )

            # Read and parse the division time from the file
            with open(division_time_file, "r") as f:
                content = f.read().strip()

            # Extract the time value
            assert content.startswith("export division_time="), (
                f"Unexpected format in division_time.sh: {content}"
            )
            recorded_time = float(content.split("=")[1])

            # Verify daughter states were created and contain consistent times
            daughter_files = [
                os.path.join(temp_dir, f"daughter_state_{i}.json") for i in range(2)
            ]

            for daughter_file in daughter_files:
                assert os.path.exists(daughter_file), (
                    f"Daughter state file {daughter_file} was not created"
                )

            sim_state = sim.ecoli_experiment.emitter.get_data()
            assert len(sim_state[recorded_time]["agents"]) == 2, (
                f"Emit at recorded division time ({recorded_time}) does not have two agents"
            )
            assert (
                len(sim_state[recorded_time - sim.config["time_step"]]["agents"]) == 1
            ), (
                f"Emit just before recorded division time ({recorded_time}) does not have one agent"
            )

        finally:
            # Clean up division_time.sh file in root directory
            if os.path.exists(division_time_file):
                os.remove(division_time_file)


def main():
    testDefault()
    testAddProcess()
    testExcludeProcess()
    test_merge()
    test_export()
    test_load_state()
    test_initial_state_overrides()
    test_division_time_recording()


if __name__ == "__main__":
    main()
