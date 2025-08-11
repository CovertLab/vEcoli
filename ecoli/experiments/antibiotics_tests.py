import os
import numpy as np

from vivarium.core.control import run_library_cli
from vivarium.core.composition import EXPERIMENT_OUT_DIR
from vivarium.library.topology import get_in
from vivarium.core.serialize import deserialize_value
from vivarium.library.units import remove_units

from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.analysis.colony.snapshots import plot_snapshots


def test_antibiotics_tetracycline():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "antibiotics_tetracycline.json")
    sim.emitter = "timeseries"
    sim.max_duration = 2
    sim.build_ecoli()
    sim.run()
    data = sim.query()
    assert data is not None


def remove_empty_values(d):
    """remove {key: value} pairs with values that are None"""
    to_delete = []
    for k, v in d.items():
        if not v:
            to_delete.append(k)
        elif isinstance(v, dict):
            v2 = remove_empty_values(v)
            if not v2:
                to_delete.append(k)
            else:
                d[k] = v2
    for k in to_delete:
        del d[k]
    return d


def test_lysis_rxn_dff_environment(max_duration=10):
    beta_lactamase = "EG10040-MONOMER[p]"
    beta_lactam = "beta-lactam[p]"
    hydrolyzed_beta_lactam = "hydrolyzed-beta-lactam[p]"
    lysis_time = 4

    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "lysis_trigger.json")
    sim.emitter = "timeseries"
    sim.max_duration = max_duration

    # add to the timeline, triggering burst
    sim.process_configs["bulk-timeline"] = {
        "timeline": {
            0: {
                ("bulk", beta_lactamase): 100,
                ("bulk", beta_lactam): 100,
            },
            lysis_time: {("burst",): True},
        }
    }
    sim.build_ecoli()
    bulk_array = sim.generated_initial_state["agents"]["0"]["bulk"]
    # For simplicity, estimate hydrolyzed beta-lactam as same mass
    sim.generated_initial_state["agents"]["0"]["bulk"] = np.append(
        bulk_array,
        np.array(
            [
                (beta_lactam, 0, 0, 0, 0, 0, 0, 0, 5.8e-7, 0, 0),
                (hydrolyzed_beta_lactam, 0, 0, 0, 0, 0, 0, 0, 5.8e-7, 0, 0),
            ],
            dtype=bulk_array.dtype,
        ),
    )
    mass_listener = sim.ecoli.steps["agents"]["0"]["ecoli-mass-listener"]
    mass_listener._bulk_molecule_by_compartment = np.stack(
        [
            np.core.defchararray.chararray.endswith(
                mass_listener.bulk_ids, abbrev + "]"
            )
            for abbrev in mass_listener.compartment_abbrev_to_index
        ]
    )
    sim.run()

    # retrieve data and pre-process for plotting
    query = [
        ("dimensions", "bounds"),
        ("fields", beta_lactamase[:-3]),
        ("fields", beta_lactam[:-3]),
        ("fields", hydrolyzed_beta_lactam[:-3]),
        ("agents", "0", "boundary"),
        ("agents", "0", "burst"),
        ("agents", "0", "bulk", beta_lactamase),
        ("agents", "0", "bulk", beta_lactam),
    ]
    data = sim.query(query=query)
    data = deserialize_value(data)
    data = remove_units(data)
    for t, v in data.items():
        if "agents" not in v:
            data[t]["agents"] = {}  # add empty agents back in

    assert (
        "0" in data[0.0]["agents"] and len(data[0.0]["agents"]) == 1
    )  # agent 0 is present at time=0
    after_lysis = remove_empty_values(data[lysis_time + 2.0]["agents"])
    assert len(after_lysis) == 0  # no agents after lysis_time

    # plot
    out_dir = os.path.join(EXPERIMENT_OUT_DIR, "lysis_environment")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plot_snapshots(
        n_snapshots=6,
        bounds=get_in(data, (max(data), "dimensions", "bounds")),
        agents={time: d["agents"] for time, d in data.items()},
        fields={time: d["fields"] for time, d in data.items()},
        out_dir=out_dir,
        filename="snapshots",
        colorbar_decimals=8,
    )


library = {
    "1": test_antibiotics_tetracycline,
    "2": test_lysis_rxn_dff_environment,
}

# uvenv ecoli/experiments/antibiotics_tests.py -n library_id
if __name__ == "__main__":
    run_library_cli(library)
