"""
=====
Lysis
=====
"""

import os
import random
import numpy as np
from scipy import constants

from vivarium.core.process import Step, Process
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.library.units import units
from ecoli.processes.environment.multibody_physics import PI
from ecoli.processes.environment.local_field import LocalField
from ecoli.library.lattice_utils import (
    get_bin_site,
    get_bin_volume,
    count_to_concentration,
)
from ecoli.library.schema import bulk_name_to_idx, counts, numpy_schema


AVOGADRO = constants.N_A / units.mol


class Lysis(Step):
    name = "lysis"
    defaults = {
        "secreted_molecules": [],
        "bin_volume": 1e-6 * units.L,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.agent_id = self.parameters["agent_id"]
        self.bin_volume = self.parameters["bin_volume"]

        # Helper indices for Numpy indexing
        self.secreted_mol_idx = None

    def ports_schema(self):
        fields_schema = {
            mol_id: {
                "_default": np.ones(1),
            }
            for mol_id in self.parameters["secreted_molecules"]
        }

        return {
            "trigger": {"_default": False},
            "agents": {},
            "internal": numpy_schema("bulk"),
            "fields": {
                **fields_schema,
                "_output": True,
            },
            "location": {
                "_default": [0.5, 0.5] * units.um,
            },
            "dimensions": {
                "bounds": {
                    "_default": [1, 1] * units.um,
                },
                "n_bins": {
                    "_default": [1, 1],
                },
                "depth": {
                    "_default": 1 * units.um,
                },
            },
        }

    def next_update(self, timestep, states):
        if self.secreted_mol_idx is None:
            self.secreted_mol_idx = {
                mol_name: bulk_name_to_idx(mol_name, states["internal"]["id"])
                for mol_name in self.parameters["secreted_molecules"]
            }

        if states["trigger"]:
            location = states["location"]
            n_bins = states["dimensions"]["n_bins"]
            bounds = states["dimensions"]["bounds"]
            depth = states["dimensions"]["depth"]

            # get bin volume
            bin_site = get_bin_site(location, n_bins, bounds)
            bin_volume = get_bin_volume(n_bins, bounds, depth)

            # apply internal states to fields
            internal = states["internal"]
            delta_fields = {}
            for mol_id, mol_idx in self.secreted_mol_idx.items():
                value = counts(internal, mol_idx)

                # delta concentration
                exchange = value
                concentration = count_to_concentration(exchange, bin_volume).to(
                    units.mM
                )

                delta_field = np.zeros((n_bins[0], n_bins[1]), dtype=np.float64)
                delta_field[bin_site[0], bin_site[1]] += concentration.to(
                    units.mM
                ).magnitude
                delta_fields[mol_id] = {"_value": delta_field, "_updater": "accumulate"}

            # remove agent and apply delta to field
            return {"agents": {"_delete": [self.agent_id]}, "fields": delta_fields}
        return {}


def mass_from_count(count, mw):
    mol = count / AVOGADRO
    return mw * mol


class ToyTransportBurst(Process):
    """
    Toy process for testing Lysis.
    Uptakes a molecule from a field, and triggers lysis.
    """

    defaults = {
        "uptake_rate": {"GLC": 1},
        "molecular_weights": {"GLC": 1 * units.fg / units.mol},
        "burst_mass": 2000 * units.fg,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecules = list(self.parameters["uptake_rate"].keys())

        # Helper indices for Numpy arrays
        self.molecule_idx = None

    def ports_schema(self):
        return {
            "external": {
                key: {
                    "_default": 0.0 * units.mM,
                    "_emit": True,
                }
                for key in self.molecules
            },
            "exchanges": {
                key: {
                    "_default": 0.0,
                    "_emit": True,
                }
                for key in self.molecules
            },
            "internal": numpy_schema("bulk"),
            "mass": {
                "_default": 0.0 * units.fg,
            },
            "length": {
                "_default": 0.0 * units.um,
            },
            "burst_trigger": {
                "_default": False,
                "_updater": "set",
                "_emit": True,
            },
        }

    def next_update(self, timestep, states):
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.molecules, states["internal"]["id"]
            )
        added = np.zeros(len(self.molecules))
        exchanged = {}
        added_mass = 0.0 * units.fg
        for i, (mol_id, e_state) in enumerate(states["external"].items()):
            exchange_concs = e_state * self.parameters["uptake_rate"][mol_id]
            # NOTE: This is not correct. We are just hacking this
            # together for testing.
            exchange_counts = exchange_concs.magnitude
            added[i] = exchange_counts
            exchanged[mol_id] = -1 * exchange_counts
            added_mass += mass_from_count(
                exchange_counts, self.parameters["molecular_weights"][mol_id]
            )

        if states["mass"] + added_mass >= self.parameters["burst_mass"]:
            return {"burst_trigger": True}

        # extend length relative to mass
        added_length = added_mass * states["length"] / states["mass"]

        return {
            "internal": [(self.molecule_idx, added.astype(int))],
            "exchanges": exchanged,
            "mass": added_mass,
            "length": added_length,
        }


class LysisAgent(Composer):
    """
    Agent that uptakes a molecule from a lattice environment,
    bursts upon reaching a set mass, and spills the molecules
    back into the environment
    """

    defaults = {
        "lysis": {"secreted_molecules": ["GLC"]},
        "transport_burst": {
            "uptake_rate": {
                "GLC": 2,
            },
            "molecular_weights": {
                "GLC": 1e22 * units.fg / units.mol,
            },
            "burst_mass": 2000 * units.fg,
        },
        "local_field": {},
        "boundary_path": ("boundary",),
        "fields_path": ("..", "..", "fields"),
        "dimensions_path": (
            "..",
            "..",
            "dimensions",
        ),
        "agents_path": (
            "..",
            "..",
            "agents",
        ),
    }

    def generate_processes(self, config):
        return {"transport_burst": ToyTransportBurst(config["transport_burst"])}

    def generate_steps(self, config):
        assert config["agent_id"]
        lysis_config = {"agent_id": config["agent_id"], **config["lysis"]}
        return {
            "local_field": LocalField(config["local_field"]),
            "lysis": Lysis(lysis_config),
        }

    def generate_flow(self, config):
        return {
            "local_field": [],
            "lysis": [],
        }

    def generate_topology(self, config):
        boundary_path = config["boundary_path"]
        fields_path = config["fields_path"]
        dimensions_path = config["dimensions_path"]
        agents_path = config["agents_path"]

        return {
            "transport_burst": {
                "internal": ("internal",),
                "exchanges": boundary_path + ("exchanges",),
                "external": boundary_path + ("external",),
                "mass": boundary_path + ("mass",),
                "length": boundary_path + ("length",),
                "burst_trigger": boundary_path + ("burst",),
            },
            "local_field": {
                "exchanges": boundary_path + ("exchanges",),
                "location": boundary_path + ("location",),
                "fields": fields_path,
                "dimensions": dimensions_path,
            },
            "lysis": {
                "trigger": boundary_path + ("burst",),
                "internal": ("internal",),
                "agents": agents_path,
                "fields": fields_path,
                "location": boundary_path + ("location",),
                "dimensions": dimensions_path,
            },
        }


def test_lysis(
    n_cells=1,
    molecule_name="beta-lactam",
    total_time=60,
    emit_step=1,
    bounds=[25, 25] * units.um,
    n_bins=[5, 5],
    uptake_rate_max=25,
    return_data=False,
):
    from ecoli.composites.environment.lattice import Lattice

    lattice_composer = Lattice(
        {
            "reaction_diffusion": {
                "molecules": [molecule_name],
                "bounds": bounds,
                "n_bins": n_bins,
                "gradient": {
                    "type": "uniform",
                    "molecules": {
                        molecule_name: 10.0 * units.mM,
                    },
                },
            },
            "multibody": {
                "bounds": bounds,
            },
        }
    )

    # initialize the composite with a lattice
    full_composite = lattice_composer.generate()

    # configure the agent composer
    agent_composer = LysisAgent(
        {
            "lysis": {"secreted_molecules": [molecule_name]},
            "transport_burst": {
                "molecular_weights": {
                    molecule_name: 1e22 * units.fg / units.mol,
                },
                "burst_mass": 2000 * units.fg,
            },
        }
    )

    # make individual agents, with unique uptake rates
    agent_ids = [str(idx) for idx in range(n_cells)]
    for agent_id in agent_ids:
        uptake_rate = random.randrange(uptake_rate_max)
        agent_composite = agent_composer.generate(
            {
                "agent_id": agent_id,
                "transport_burst": {
                    "uptake_rate": {
                        molecule_name: uptake_rate,
                    }
                },
            }
        )
        agent_path = ("agents", agent_id)
        full_composite.merge(composite=agent_composite, path=agent_path)

    # get initial state
    initial_state = full_composite.initial_state()
    initial_state["agents"] = {}
    for agent_id in agent_ids:
        agent_angle = random.uniform(0, 2 * PI)
        initial_state["agents"][agent_id] = {
            "boundary": {"angle": agent_angle},
            "internal": np.array(
                [
                    ("beta-lactam", 0),
                    ("hydrolyzed-beta-lactam", 0),
                    ("EG10040-MONOMER[p]", 0),
                ],
                dtype=[("id", "U40"), ("count", int)],
            ),
        }

    # run the simulation and return the data
    sim = Engine(
        processes=full_composite.processes,
        steps=full_composite.steps,
        topology=full_composite.topology,
        flow=full_composite.flow,
        initial_state=initial_state,
        emit_step=emit_step,
    )
    sim.update(total_time)
    data = sim.emitter.get_data_unitless()

    if return_data:
        return data


def main():
    from ecoli.analysis.colony.snapshots import (
        plot_snapshots,
        format_snapshot_data,
        make_video,
    )

    bounds = [15, 15] * units.um
    molecule_name = "beta-lactam"

    data = test_lysis(
        n_cells=8,
        molecule_name=molecule_name,
        total_time=1000,
        emit_step=10,
        bounds=bounds,
        n_bins=[11, 11],
        return_data=True,
    )

    # format the data for plot_snapshots
    agents, fields = format_snapshot_data(data)

    out_dir = os.path.join("out", "experiments", "lysis")
    os.makedirs(out_dir, exist_ok=True)
    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=5,
        out_dir=out_dir,
        filename="lysis_snapshots",
    )

    # make snapshot video
    make_video(
        data,
        bounds,
        plot_type="fields",
        out_dir=out_dir,
        filename="lysis_video",
    )


# uv run ecoli/processes/environment/lysis.py
if __name__ == "__main__":
    main()
