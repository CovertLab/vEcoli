import os

from vivarium.core.serialize import deserialize_value
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, EcoliSim


DATA = "data/cell_wall/cell_wall_test_rig_17_09_2022_00_41_51.csv"


def run_experiment():
    # TODO: Run all colony sims with EngineProcess because otherwise Processes
    # will run before Steps after division and timesteps are counter-intuitive
    # (e.g. if a cell divides at t=4 and has a CellWall process w/ a time step
    # of 10, the CellWall process will run using the state after t=4 and apply
    # its update at t=10 instead of at t=14)
    
    total_time = 10

    ecoli = EcoliSim.from_file(
        os.path.join(CONFIG_DIR_PATH, "test_configs/cell_wall_division.json")
    )
    ecoli.total_time = total_time

    ecoli.run()
    return ecoli.query()


def validate_division(data):
    # Expectations:
    # - there are two cells at the end
    # - immediately after dividing, wall state, pbp state, incorporated murein
    #   and unincorporated murein are reset to their defaults. Shadow murein
    #   is divided binomially.
    # - on the next timestep, these are updated to be reasonable values.
    #   Murein state should sum to bulk murein * 4.

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
        assert deserialize_value(cell_data["pbp_state"]["active_fraction_PBP1A"]) == 1
        assert deserialize_value(cell_data["pbp_state"]["active_fraction_PBP1B"]) == 1

    # Shadow murein should divide binomially
    assert (
        sum(
            cell_data["murein_state"]["shadow_murein"]
            for _, cell_data in data[divide_time]["agents"].items()
        )
        == data[time[divide_index - 1]]["agents"]["0"]["murein_state"]["shadow_murein"]
    )

    next_timestep = time[divide_index + 1]
    for _, cell_data in data[next_timestep]["agents"].items():
        assert cell_data["wall_state"]["lattice_rows"] > 0
        assert cell_data["wall_state"]["lattice_cols"] > 0
        assert cell_data["wall_state"]["cracked"] == False
        assert (
            sum(cell_data["murein_state"].values())
            == 4 * cell_data["bulk"]["CPD-12261[p]"]
        )


def test_cell_wall_division():
    validate_division(run_experiment())


def main():
    test_cell_wall_division()


if __name__ == "__main__":
    main()
