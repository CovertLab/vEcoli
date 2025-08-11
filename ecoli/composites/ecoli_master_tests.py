"""
======================
Tests for Ecoli Master
======================
"""

import os
import numpy as np
import pytest
import warnings

from vivarium.core.engine import Engine
from vivarium.core.control import run_library_cli

from ecoli.library.schema import attrs, bulk_name_to_idx
from ecoli.analysis.colony.snapshots import (
    plot_snapshots,
    format_snapshot_data,
    make_video,
)
from configs import (
    ECOLI_DEFAULT_PROCESSES,
    ECOLI_DEFAULT_TOPOLOGY,
)
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


@pytest.mark.slow
def test_division(agent_id="0", max_duration=4):
    """tests that a cell can be divided and keep running"""

    # get initial mass from Ecoli composer
    sim = EcoliSim.from_file()
    # Initial state saved right before division set to be
    # triggered according to D period
    sim.config["initial_state_file"] = "vivecoli_t2527"
    sim.config["divide"] = True
    sim.config["agent_id"] = agent_id
    sim.config["max_duration"] = max_duration
    # Ensure unique molecules are emitted
    sim.config["emit_unique"] = True
    sim.build_ecoli()

    sim.run()

    # retrieve output
    output = sim.ecoli_experiment.emitter.get_data()
    # metabolism runs and greatly changes molecule counts after
    # previoius emit but before division, so ignore metabolites
    metabolite_idx = sim.ecoli_experiment.state["agents"]["00"][
        "ecoli-metabolism"
    ].value.metabolite_idx
    # ignore trna charging molecules, which also change before division
    polypep_elong = sim.ecoli_experiment.state["agents"]["00"]["process"][
        "ecoli-polypeptide-elongation"
    ].value[0]
    trna_charging_idx = polypep_elong.charging_molecule_idx
    ppgpp_idx = np.array([polypep_elong.ppgpp_idx])
    # ignore inactive ribosomes, which also change before division
    polypep_init = sim.ecoli_experiment.state["agents"]["00"]["process"][
        "ecoli-polypeptide-initiation"
    ].value[0]
    ribosome_idx = np.array(
        [polypep_init.ribosome30S_idx, polypep_init.ribosome50S_idx]
    )
    # ignore fragment bases, which also change before division
    fragment_base_idx = (
        sim.ecoli_experiment.state["agents"]["00"]["process"]["ecoli-rna-maturation"]
        .value[0]
        .fragment_base_idx
    )
    # ignore high-count membrane-related proteins, which also change before division
    membrane_idx = bulk_name_to_idx(
        ["EG10544-MONOMER[m]", "EG10669-MONOMER[o]", "EG50003-MONOMER[c]"],
        sim.ecoli_experiment.state["agents"]["00"]["bulk"].value["id"],
    )
    ignore_idx = np.concatenate(
        [
            metabolite_idx,
            trna_charging_idx,
            ppgpp_idx,
            ribosome_idx,
            fragment_base_idx,
            membrane_idx,
        ]
    )
    mother_state = next(iter(output[1]["agents"].values()))
    mother_bulk = np.delete(mother_state["bulk"], ignore_idx)
    daughter_states = list(output[2]["agents"].values())
    daughter_bulk = [np.delete(ds["bulk"], ignore_idx) for ds in daughter_states]

    # compare the counts of bulk molecules between the mother and daughters
    # this is not exact because the mother grew slightly in the timestep
    # after its last emit but before being split into two daughter cells
    assert np.allclose(
        mother_bulk, np.array(daughter_bulk[0]) + np.array(daughter_bulk[1]), atol=60
    )

    # compare the counts of unique molecules between the mother and daughters
    for name, mols in mother_state["unique"].items():
        d1_state = daughter_states[0]["unique"][name]
        d2_state = daughter_states[1]["unique"][name]
        mol_keys = sim.ecoli_experiment.state["agents"]["00"]["unique"][
            name
        ].value.dtype.names
        entryState_col = np.where(np.array(mol_keys) == "_entryState")[0][0]
        n_mother = sum(mols[entryState_col])
        n_daughter = sum(d1_state[entryState_col]) + sum(d2_state[entryState_col])
        if name == "chromosome_domain":
            # Chromosome domain 0 is lost after division because
            # it has been fully split into child domains 1 and 2
            n_daughter += 1
        assert np.isclose(n_mother, n_daughter, rtol=0.1), (
            f"{name}: mother has {n_mother}, daughters have {n_daughter}"
        )
        # Assert that no unique mol is in both daughters
        unique_idx_col = np.where(np.array(mol_keys) == "unique_index")[0][0]
        assert not (set(d1_state[unique_idx_col]) & set(d2_state[unique_idx_col]))

    # asserts
    final_agents = output[max_duration]["agents"].keys()
    print(f"initial agent id: {agent_id}")
    print(f"final agent ids: {final_agents}")
    assert len(final_agents) == 2
    # Check that MarkDPeriod updated the has_triggered_division attribute
    for agent_id in final_agents:
        full_chrom = sim.ecoli_experiment.state["agents"][agent_id]["unique"][
            "full_chromosome"
        ].value
        (has_triggered_division,) = attrs(full_chrom, ["has_triggered_division"])
        assert np.all(has_triggered_division)


def test_division_topology():
    """test that the topology is correctly dividing"""
    timestep = 2
    agent_id = "0"
    # get initial mass from Ecoli composer
    sim = EcoliSim.from_file()
    sim.config["seed"] = 1
    sim.config["initial_state_file"] = "vivecoli_t2527"
    sim.config["divide"] = True
    sim.config["agent_id"] = agent_id
    sim.build_ecoli()

    # make the experiment
    experiment_config = {
        "processes": sim.ecoli.processes,
        "steps": sim.ecoli.steps,
        "flow": sim.ecoli.flow,
        "topology": sim.ecoli.topology,
        "initial_state": sim.generated_initial_state,
    }

    # Since unique numpy updater is an class method, internal
    # deepcopying in vivarium-core causes this warning to appear
    warnings.filterwarnings(
        "ignore",
        message="Incompatible schema "
        "assignment at .+ Trying to assign the value <bound method "
        r"UniqueNumpyUpdater\.updater .+ to key updater, which already "
        r"has the value <bound method UniqueNumpyUpdater\.updater",
    )
    sim.ecoli_experiment = Engine(**experiment_config)

    # Only emit designated stores
    sim.ecoli_experiment.state.set_emit_values([tuple()], False)
    sim.ecoli_experiment.state.set_emit_values(
        sim.config["emit_paths"],
        True,
    )
    # Divide immediately
    sim.ecoli_experiment.state["agents"]["0"]["divide"] = True

    # Clean up unnecessary references
    sim.generated_initial_state = None
    sim.ecoli_experiment.initial_state = None

    full_topology = sim.ecoli_experiment.state.get_topology()
    mother_topology = full_topology["agents"][agent_id].copy()

    # update one time step at a time until division
    while len(full_topology["agents"]) <= 1:
        sim.ecoli_experiment.update(timestep)
        full_topology = sim.ecoli_experiment.state.get_topology()
    sim.ecoli_experiment.end()

    # assert that the daughter topologies are the same as the mother topology
    daughter_ids = list(full_topology["agents"].keys())
    for daughter_id in daughter_ids:
        daughter_topology = full_topology["agents"][daughter_id]
        assert daughter_topology == mother_topology


def test_ecoli_generate():
    sim = EcoliSim.from_file()
    sim.divide = False
    sim.build_ecoli()

    # asserts to ecoli_composite['processes'] and ecoli_composite['topology']
    assert all(
        "_requester" in k
        or "_evolver" in k
        or "allocator" in k
        or "unique_update" in k
        or k == "division"
        or k == "mark_d_period"
        or isinstance(v, ECOLI_DEFAULT_PROCESSES[k])
        for k, v in sim.ecoli["steps"].items()
    )
    ignore_keys = ["request", "process", "allocate", "global_time", "next_update_time"]
    for k, v in sim.ecoli["topology"].items():
        proc_name = k.split("_requester")[0].split("_evolver")[0]
        # Clock topology is not registered in registry
        if proc_name == "clock":
            pass
        elif proc_name in ECOLI_DEFAULT_TOPOLOGY:
            if "_requester" in k or "_evolver" in k:
                # Ignore partitioning and timestepping keys
                for ignore_key in ignore_keys:
                    v.pop(ignore_key, None)
            assert ECOLI_DEFAULT_TOPOLOGY[proc_name] == v


def test_lattice_lysis(plot=False):
    """
    Run plots:
    '''
    > uvenv ecoli/composites/ecoli_master_tests.py -n 4 -o plot=True
    '''

    ANTIBIOTIC_KEY = 'nitrocefin'
    PUMP_KEY = 'TRANS-CPLX-201[s]'
    PORIN_KEY = 'porin'
    BETA_LACTAMASE_KEY = 'EG10040-MONOMER[p]'

    TODO: connect glucose! through local_field
    """
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "lysis.json")
    sim.max_duration = 10
    sim.process_configs.update({"global_clock": {"time_step": 2}})
    sim.build_ecoli()
    # Add beta-lactam to bulk store
    initial_state = sim.generated_initial_state
    zero_mass = (0,) * 9
    initial_state["agents"]["0"]["bulk"] = np.append(
        initial_state["agents"]["0"]["bulk"],
        np.array(
            [
                ("beta-lactam[p]", 0) + zero_mass,
                ("hydrolyzed-beta-lactam[p]", 0) + zero_mass,
            ],
            dtype=initial_state["agents"]["0"]["bulk"].dtype,
        ),
    )
    sim.run()
    data = sim.query()

    if plot:
        plot_spatial_snapshots(data, sim, experiment_dir="ecoli_lysis")


def plot_spatial_snapshots(data, sim, experiment_dir="ecoli_test"):
    out_dir = os.path.join("out", "experiments", experiment_dir)
    os.makedirs(out_dir, exist_ok=True)

    bounds = sim.config["spatial_environment_config"]["multibody"]["bounds"]

    # format the data for plot_snapshots
    agents, fields = format_snapshot_data(data)

    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=5,
        out_dir=out_dir,
        filename="snapshots",
    )

    # make snapshot video
    make_video(
        data,
        bounds,
        plot_type="fields",
        out_dir=out_dir,
        filename="video",
    )


def test_emit_unique():
    """
    Test that the ``emit_unique`` configuration option works. This can be broken
    if a new process is added whose ports schema connects to a unique molecule
    without setting the ``_emit`` property to ``config['emit_unique']``.
    """
    sim = EcoliSim.from_file()
    sim.config["emit_unique"] = True
    sim.config["max_duration"] = 1
    sim.build_ecoli()
    sim.run()
    unique_molecules = sim.ecoli_experiment.state["agents"]["0"]["unique"].inner.keys()
    data = sim.query(
        [
            (
                "agents",
                "0",
                "unique",
            )
        ]
    )
    for val in data.values():
        for unique_mol in unique_molecules:
            assert unique_mol in val["agents"]["0"]["unique"]
            assert isinstance(val["agents"]["0"]["unique"][unique_mol], list)


test_library = {
    "1": test_division,
    "2": test_division_topology,
    "3": test_ecoli_generate,
    "4": test_lattice_lysis,
    "5": test_emit_unique,
}

# run experiments in test_library from the command line with:
# uvenv ecoli/composites/ecoli_master_tests.py -n [experiment id]
if __name__ == "__main__":
    run_library_cli(test_library)
