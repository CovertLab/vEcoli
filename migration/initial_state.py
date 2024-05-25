import numpy as np

from ecoli.library.sim_data import LoadSimData, SIM_DATA_PATH_NO_OPERONS
from ecoli.states.wcecoli_state import get_state_from_file


def compare_states(viv_state, wc_state):
    assert np.all(viv_state["bulk"] == wc_state["bulk"])
    for unique_id, unique_data in viv_state["unique"].items():
        for colname in unique_data.dtype.names:
            # These indices are generated completely differently in wcEcoli
            if colname not in ["mRNA_index", "unique_index", "RNAP_index"]:
                assert np.all(
                    unique_data[colname] == wc_state["unique"][unique_id][colname]
                )
    assert viv_state["environment"] == wc_state["environment"]
    assert viv_state["boundary"] == wc_state["boundary"]


def test_initial_state():
    sim_data = LoadSimData()
    initial_state = sim_data.generate_initial_state()
    wc_initial_state = get_state_from_file("data/migration/wcecoli_t0.json")
    compare_states(initial_state, wc_initial_state)

    sim_data = LoadSimData(sim_data_path=SIM_DATA_PATH_NO_OPERONS)
    initial_state = sim_data.generate_initial_state()
    wc_initial_state = get_state_from_file("data/migration_no_operons/wcecoli_t0.json")
    compare_states(initial_state, wc_initial_state)


if __name__ == "__main__":
    test_initial_state()
