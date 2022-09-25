import os
from vivarium.core.serialize import deserialize_value
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


def run_experiment():
    ecoli = EcoliSim.from_file(os.path.join(CONFIG_DIR_PATH, "cell_wall_division.json"))
    ecoli.total_time = 10
    ecoli.run()
    return ecoli.query()


def validate_division(data):
    # Expectations:
    # - there are two cells at the end
    # - immediately after dividing, wall state and pbp state are reset to their
    #   defaults. Murein state...
    # - on the next timestep, these are updated to be reasonable values.add()

    time = sorted(data.keys())

    assert "agents" in data[time[-1]] and len(data[time[-1]]["agents"].keys()) == 2
    divide_index = [len(data[t]["agents"]) == 2 for t in time].index(True)
    divide_time = time[divide_index]

    for _, cell_data in data[divide_time]["agents"].items():
        assert cell_data["wall_state"]["lattice_rows"] == 0
        assert cell_data["wall_state"]["lattice_cols"] == 0
        assert cell_data["wall_state"]["extension_factor"] == 1
        assert cell_data["wall_state"]["cracked"] == False
        assert cell_data["murein_state"]["incorporated_murein"] == 0
        assert cell_data["murein_state"]["unincorporated_murein"] == 0
        assert cell_data["murein_state"]["shadow_murein"] == 0
        assert deserialize_value(cell_data["pbp_state"]["active_fraction_PBP1A"]) == 1
        assert deserialize_value(cell_data["pbp_state"]["active_fraction_PBP1B"]) == 1

    next_timestep = time[divide_index + 1]
    for _, cell_data in data[next_timestep]["agents"].items():
        assert cell_data["wall_state"]["lattice_rows"] > 0
        assert cell_data["wall_state"]["lattice_cols"] > 0
        assert cell_data["wall_state"]["cracked"] == False
        assert sum(cell_data["murein_state"].values()) == 4 * cell_data["bulk"]["CPD-12261[p]"]


def test_cell_wall_division():
    validate_division(run_experiment())


def main():
    test_cell_wall_division()


if __name__ == "__main__":
    main()
