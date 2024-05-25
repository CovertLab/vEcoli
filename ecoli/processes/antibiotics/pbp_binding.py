import os

import numpy as np
from vivarium.core.composer import Composite
from vivarium.core.engine import Engine
from vivarium.core.process import Step
from vivarium.library.units import units, remove_units
from vivarium.plots.simulation_output import plot_variables

from ecoli.library.parameters import param_store
from ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts
from ecoli.processes.registries import topology_registry
from ecoli.processes.shape import length_from_volume
from ecoli.processes.bulk_timeline import BulkTimelineProcess
from ecoli.library.cell_wall.column_sampler import (
    geom_sampler,
    sample_lattice,
)
from ecoli.library.cell_wall.lattice import (
    calculate_lattice_size,
)


# Register default topology for this process, associating it with process name
NAME = "ecoli-pbp-binding"
TOPOLOGY = {
    "bulk": ("bulk",),
    "murein_state": ("murein_state",),
    "concentrations": ("concentrations",),
    "pbp_state": ("pbp_state",),
    "wall_state": ("wall_state",),
    "volume": ("boundary", "volume"),
    "first_update": (
        "first_update",
        "pbp_binding",
    ),
}
topology_registry.register(NAME, TOPOLOGY)


class PBPBinding(Step):
    name = NAME
    topology = TOPOLOGY

    defaults = {
        "murein_name": "CPD-12261[p]",
        "beta_lactam": "ampicillin",  # Supports cephaloridine, ampicillin
        "PBP": {  # penicillin-binding proteins
            "PBP1A": "CPLX0-7717[m]",  # transglycosylase-transpeptidase ~100
            # PBP1B has three isoforms: α (currently not produced by model),
            # β (degradation product of α, not in vivo), and γ (made by model)
            "PBP1B_alpha": "CPLX0-3951[i]",
            "PBP1B_gamma": "CPLX0-8300[c]",
        },
        "kinetic_params": {
            "cephaloridine": {
                "K_A": {
                    "PBP1A": param_store.get(
                        ("cephaloridine", "pbp_binding", "K_A (micromolar)", "PBP1A")
                    ),
                    "PBP1B": param_store.get(
                        ("cephaloridine", "pbp_binding", "K_A (micromolar)", "PBP1B")
                    ),
                },
                "Hill_n": {
                    "PBP1A": 1,
                    "PBP1B": 1,
                },
            },
            "ampicillin": {
                "K_A": {
                    "PBP1A": param_store.get(
                        ("ampicillin", "pbp_binding", "K_A (micromolar)", "PBP1A")
                    ),
                    "PBP1B": param_store.get(
                        ("ampicillin", "pbp_binding", "K_A (micromolar)", "PBP1B")
                    ),
                },
                "Hill_n": {
                    "PBP1A": 1,
                    "PBP1B": 1,
                },
            },
        },
        # Parameters to initialize cell wall after division (see cell_wall.py)
        "strand_term_p": param_store.get(("cell_wall", "strand_term_p")),
        "cell_radius": param_store.get(("cell_wall", "cell_radius")),
        "disaccharide_height": param_store.get(("cell_wall", "disaccharide_height")),
        "disaccharide_width": param_store.get(("cell_wall", "disaccharide_width")),
        "inter_strand_distance": param_store.get(
            ("cell_wall", "inter_strand_distance")
        ),
        # Simulation parameters
        "seed": 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.murein = self.parameters["murein_name"]
        self.beta_lactam = self.parameters["beta_lactam"]
        self.PBP1A = self.parameters["PBP"]["PBP1A"]
        self.PBP1B_alpha = self.parameters["PBP"]["PBP1B_alpha"]
        self.PBP1B_gamma = self.parameters["PBP"]["PBP1B_gamma"]
        self.K_A_1a = self.parameters["kinetic_params"][self.beta_lactam]["K_A"][
            "PBP1A"
        ]
        self.n_1a = self.parameters["kinetic_params"][self.beta_lactam]["Hill_n"][
            "PBP1A"
        ]
        self.K_A_1b = self.parameters["kinetic_params"][self.beta_lactam]["K_A"][
            "PBP1B"
        ]
        self.n_1b = self.parameters["kinetic_params"][self.beta_lactam]["Hill_n"][
            "PBP1B"
        ]

        # Parameters to initialize cell wall after division
        self.strand_term_p = self.parameters["strand_term_p"]
        self.cell_radius = self.parameters["cell_radius"]
        self.circumference = 2 * np.pi * self.cell_radius
        self.disaccharide_height = self.parameters["disaccharide_height"]
        self.disaccharide_width = self.parameters["disaccharide_width"]
        self.inter_strand_distance = self.parameters["inter_strand_distance"]
        self.rng = np.random.default_rng(self.parameters["seed"])

        # Helper indices for Numpy arrays
        self.murein_idx = None

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "murein_state": {
                "incorporated_murein": {
                    "_default": 0,
                    "_updater": "set",
                    "_emit": True,
                },
                "unincorporated_murein": {
                    "_default": 0,
                    "_emit": True,
                },
                "shadow_murein": {"_default": 0, "_emit": True},
            },
            "concentrations": {
                self.beta_lactam: {"_default": 0.0 * units.micromolar, "_emit": True},
            },
            "pbp_state": {
                "active_fraction_PBP1A": {
                    "_default": 1.0,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {"divider": "set_value", "config": {"value": 1.0}},
                },
                "active_fraction_PBP1B": {
                    "_default": 1.0,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {"divider": "set_value", "config": {"value": 1.0}},
                },
            },
            "wall_state": {
                "lattice": {
                    "_default": None,
                    "_updater": "set",
                    "_emit": False,
                },
                "lattice_rows": {
                    "_default": 0,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "zero",
                },
                "lattice_cols": {
                    "_default": 0,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": "zero",
                },
                "extension_factor": {
                    "_default": 1,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {"divider": "set_value", "config": {"value": 1}},
                },
            },
            "volume": {"_default": 1 * units.fL, "_emit": True},
            "first_update": {
                "_default": True,
                "_updater": "set",
                "_divider": {"divider": "set_value", "config": {"value": True}},
            },
        }

    def next_update(self, timestep, states):
        if self.murein_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.murein_idx = bulk_name_to_idx(self.murein, bulk_ids)
            self.pbp1a_idx = bulk_name_to_idx(self.PBP1A, bulk_ids)
            self.pbp1ba_idx = bulk_name_to_idx(self.PBP1B_alpha, bulk_ids)
            self.pbp1bg_idx = bulk_name_to_idx(self.PBP1B_gamma, bulk_ids)
        update = {"murein_state": {}}

        # Calculate fraction of active PBP1a, PBP1b using Hill Equation
        # (calculating prop NOT bound, i.e. 1 - Hill eq value)
        beta_lactam = states["concentrations"][self.beta_lactam]
        active_fraction_1a = 1 / (1 + (beta_lactam / self.K_A_1a) ** self.n_1a)
        active_fraction_1b = 1 / (1 + (beta_lactam / self.K_A_1b) ** self.n_1b)

        update["pbp_state"] = {
            "active_fraction_PBP1A": active_fraction_1a,
            "active_fraction_PBP1B": active_fraction_1b,
        }

        if states["first_update"]:
            update["first_update"] = False
            # Initialize cell wall if necessary (first cell in sim)
            if states["wall_state"]["lattice"] is None:
                # Make sure that all usable murein is initiailly unincorporated
                unincorporated_monomers = (
                    4 * counts(states["bulk"], self.murein_idx)
                    - states["murein_state"]["shadow_murein"]
                )
                incorporated_monomers = 0

                # Get cell size information
                length = length_from_volume(states["volume"], self.cell_radius * 2).to(
                    "micrometer"
                )

                # Get dimensions of the lattice
                rows, cols = calculate_lattice_size(
                    length,
                    self.inter_strand_distance,
                    self.disaccharide_height,
                    self.disaccharide_width,
                    self.circumference,
                    1,
                )

                # Populate the lattice
                lattice = sample_lattice(
                    unincorporated_monomers,
                    rows,
                    cols,
                    geom_sampler(self.rng, self.strand_term_p),
                    self.rng,
                )

                incorporated_monomers = lattice.sum()
                unincorporated_monomers -= incorporated_monomers
                update.update(
                    {
                        "wall_state": {
                            "lattice": lattice,
                            "extension_factor": 1,
                            "lattice_rows": lattice.shape[0],
                            "lattice_cols": lattice.shape[1],
                        },
                        "murein_state": {
                            "incorporated_murein": incorporated_monomers,
                            "unincorporated_murein": -incorporated_monomers,
                        },
                    }
                )
                return update

            # Set lattice rows, cols, and extension factor after division when
            # running in EngineProcess
            elif states["wall_state"]["lattice_rows"] == 0:
                # Get cell size information
                length = length_from_volume(states["volume"], self.cell_radius * 2).to(
                    "micrometer"
                )

                # Set extension factor such that lattice covers the cell
                lattice = states["wall_state"]["lattice"]
                extension = remove_units(
                    (
                        length
                        / (
                            lattice.shape[1]
                            * (self.inter_strand_distance + self.disaccharide_width)
                        )
                    ).to("dimensionless")
                )

                update.update(
                    {
                        "wall_state": {
                            "lattice_rows": lattice.shape[0],
                            "lattice_cols": lattice.shape[1],
                            "extension_factor": extension,
                        }
                    }
                )
                return update

        # New murein to allocate
        new_murein = 4 * counts(states["bulk"], self.murein_idx) - sum(
            states["murein_state"].values()
        )

        # Allocate real vs. shadow murein based on
        # what fraction of PBPs are active
        PBP1A = counts(states["bulk"], self.pbp1a_idx)
        PBP1B_alpha = counts(states["bulk"], self.pbp1ba_idx)
        PBP1B_gamma = counts(states["bulk"], self.pbp1bg_idx)
        total_PBP = PBP1A + PBP1B_alpha + PBP1B_gamma

        if total_PBP > 0:
            real_new_murein = int(
                round(
                    (
                        active_fraction_1a * (PBP1A / total_PBP)
                        + active_fraction_1b * (PBP1B_alpha / total_PBP)
                        + active_fraction_1b * (PBP1B_gamma / total_PBP)
                    )
                    * new_murein
                )
            )
        else:
            real_new_murein = new_murein

        update["murein_state"].update(
            {
                "unincorporated_murein": real_new_murein,
                "shadow_murein": new_murein - real_new_murein,
            }
        )

        return update


def test_pbp_binding():
    # Create process
    params = {"beta_lactam": "ampicillin"}
    process = PBPBinding(params)
    initial_murein = 450000
    timeline_params = {
        "timeline": {
            time: {
                ("bulk", "CPD-12261[p]"): int(initial_murein + 1000 * time),
                ("concentrations", "ampicillin"): (
                    (time - 50) / 10 * units.micromolar
                    if time > 50
                    else 0 * units.micromolar
                ),
            }
            for time in range(0, 100)
        }
    }
    timeline_process = BulkTimelineProcess(timeline_params)

    # Create composite with timeline
    processes = {"pbp_binding": process, "bulk-timeline": timeline_process}
    topology = {
        "pbp_binding": TOPOLOGY,
        "bulk-timeline": {"bulk": ("bulk",), "concentrations": ("concentrations",)},
    }

    # Run experiment
    settings = {
        "total_time": 100,
        "initial_state": {
            "murein_state": {
                "incorporated_murein": 0,
                "unincorporated_murein": initial_murein * 4,
                "shadow_murein": 0,
            },
            "concentrations": {
                "ampicillin": 0 * units.micromolar,
            },
            "bulk": np.array(
                [
                    ("CPD-12261[p]", initial_murein),
                    ("CPLX0-7717[m]", 100),
                    ("CPLX0-3951[i]", 100),
                    ("CPLX0-8300[c]", 0),
                ],
                dtype=[("id", "U40"), ("count", int)],
            ),
        },
    }
    composite = Composite(
        {
            "processes": processes,
            "topology": topology,
            "state": settings["initial_state"],
        }
    )

    sim = Engine(composite=composite)
    sim.update(settings["total_time"])
    data = sim.emitter.get_timeseries()

    bulk_array = np.array(data["bulk"])
    data["bulk"] = {"CPD-12261[p]": bulk_array[:, 0]}

    # Plot output
    fig = plot_variables(
        data,
        variables=[
            ("concentrations", ("ampicillin", "micromolar")),
            ("bulk", "CPD-12261[p]"),
            ("murein_state", "incorporated_murein"),
            ("murein_state", "unincorporated_murein"),
            ("murein_state", "shadow_murein"),
            ("pbp_state", ("active_fraction_PBP1A", "dimensionless")),
            ("pbp_state", ("active_fraction_PBP1B", "dimensionless")),
        ],
    )
    fig.get_axes()[-1].set_ylim(0, 1)
    fig.get_axes()[-2].set_ylim(0, 1)
    os.makedirs("out/processes/pbp_binding", exist_ok=True)
    fig.savefig("out/processes/pbp_binding/test.png")

    # Validate output data
    total_murein = np.array(data["bulk"]["CPD-12261[p]"])
    incorporated_murein = np.array(data["murein_state"]["incorporated_murein"])
    unincorporated_murein = np.array(data["murein_state"]["unincorporated_murein"])
    shadow_murein = np.array(data["murein_state"]["shadow_murein"])
    assert all(
        4 * total_murein == incorporated_murein + unincorporated_murein + shadow_murein
    )


def main():
    test_pbp_binding()


if __name__ == "__main__":
    main()
