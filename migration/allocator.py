"""
tests that vivarium-ecoli allocator process update is the same as saved wcEcoli updates
"""

import json
import pytest
import numpy as np

from ecoli.processes.allocator import Allocator
from migration import LOAD_SIM_DATA
from migration.migration_utils import run_non_partitioned_process, recursive_compare
from ecoli.library.wcecoli_state import get_state_from_file


with open("data/proc_to_name.json", "r") as f:
    viv_to_wc_proc = json.load(f)
wc_to_viv_proc = {v: k for k, v in viv_to_wc_proc.items()}

layers = {
    1: ["ecoli-equilibrium", "ecoli-two-component-system", "ecoli-rna-maturation"],
    3: [
        "ecoli-transcript-initiation",
        "ecoli-polypeptide-initiation",
        "ecoli-chromosome-replication",
        "ecoli-protein-degradation",
        "ecoli-rna-degradation",
        "ecoli-complexation",
    ],
    4: [
        "ecoli-transcript-elongation",
        "ecoli-polypeptide-elongation",
    ],
}


def run_and_compare(init_time):
    # Set time parameters
    init_time = init_time

    # Create process, experiment, loading in initial state from file.
    process_names = layers[1] + layers[3] + layers[4]
    config = LOAD_SIM_DATA.get_allocator_config(process_names=process_names)
    allocator_process = Allocator(config)
    allocator_process.is_step = lambda: False

    # Load requested and allocated counts from wcEcoli
    with open(f"data/migration/bulk_requested_t{init_time}.json", "r") as f:
        initial_request = json.load(f)
    with open(f"data/migration/bulk_partitioned_t{init_time}.json", "r") as f:
        wc_update = json.load(f)

    for layer, processes in layers.items():
        initial_state = get_state_from_file(
            path=f"data/migration/wcecoli_t{init_time}_before_layer_{layer}.json"
        )
        bulk_idx = np.arange(len(initial_state["bulk"]))

        # Load requests from wcEcoli into initial state
        initial_state["request"] = {}
        for process in processes:
            proc_name = viv_to_wc_proc.get(process, None)
            initial_state["request"][process] = {
                "bulk": [(bulk_idx, initial_request[proc_name])]
            }

        # Run the process and get an update
        actual_update = run_non_partitioned_process(
            allocator_process, Allocator.topology, initial_state=initial_state
        )
        actual_allocated = {}
        wc_allocated = {}
        for process in processes:
            proc_name = viv_to_wc_proc.get(process, None)
            actual_allocated[proc_name] = actual_update["allocate"][process]["bulk"]
            wc_allocated[proc_name] = wc_update[proc_name]

        # Compare to wcEcoli partitioned counts
        assert recursive_compare(
            actual_allocated, wc_allocated, check_keys_strict=False
        )


@pytest.mark.master
def test_allocator_migration():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time)


if __name__ == "__main__":
    test_allocator_migration()
