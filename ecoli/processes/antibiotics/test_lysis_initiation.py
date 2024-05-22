import os
import numpy as np
from ecoli.experiments.ecoli_master_sim import EcoliSim


def test_lysis_initiation():
    from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH

    ecoli = EcoliSim.from_file(
        os.path.join(CONFIG_DIR_PATH, "test_configs/test_lysis_initiation.json")
    )
    ecoli.total_time = 20
    ecoli.build_ecoli()
    agent_data = ecoli.generated_initial_state["agents"]["0"]
    zero_mass = (0,) * 9
    agent_data["bulk"] = np.append(
        agent_data["bulk"],
        np.array(
            [
                ("beta-lactam[p]", 0) + zero_mass,
                ("hydrolyzed-beta-lactam[p]", 0) + zero_mass,
            ],
            dtype=agent_data["bulk"].dtype,
        ),
    )
    ecoli.run()

    data = ecoli.query()

    # Assert lysis has occurred by the end of the simulation
    # (agents store exists at the beginning, deleted by the end)
    time = sorted(data.keys())
    assert "agents" in data[time[0]].keys()
    assert "agents" not in data[time[-1]].keys()


def main():
    test_lysis_initiation()


if __name__ == "__main__":
    main()
