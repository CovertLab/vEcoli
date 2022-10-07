import os

import numpy as np
from vivarium.core.composer import Composite
from vivarium.core.composition import add_timeline
from vivarium.core.engine import Engine
from vivarium.core.process import Step
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables

from ecoli.library.parameters import param_store
from ecoli.library.schema import bulk_schema
from ecoli.processes.registries import topology_registry
from ecoli.processes.shape import length_from_volume
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
    "total_murein": ("bulk",),
    "murein_state": ("murein_state",),
    "concentrations": ("concentrations",),
    "bulk": ("bulk",),
    "pbp_state": ("pbp_state",),
    "wall_state": ("wall_state",),
    "volume": ("boundary", "volume")
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
            "PBP1B": "CPLX0-3951[i]",  # transglycosylase-transpeptidase ~100
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
        self.PBP1B = self.parameters["PBP"]["PBP1B"]
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

    def ports_schema(self):
        return {
            "total_murein": bulk_schema([self.parameters["murein_name"]]),
            "murein_state": {
                "incorporated_murein": {
                    "_default": 0,
                    "_updater": "set",
                    "_emit": True,
                },
                "unincorporated_murein": {
                    "_default": 0,
                    "_updater": "set",
                    "_emit": True,
                },
                "shadow_murein": {"_default": 0, "_updater": "set", "_emit": True},
            },
            "concentrations": {
                self.beta_lactam: {"_default": 0.0 * units.micromolar, "_emit": True},
            },
            "bulk": bulk_schema(list(self.parameters["PBP"].values())),
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
                    "_divider": "set_none",
                },
                "extension_factor": {
                    "_default": 1,
                    "_updater": "set",
                    "_emit": True,
                    "_divider": {"divider": "set_value", "config": {"value": 1}},
                },
            },
            "volume": {"_default": 1 * units.fL, "_emit": True}
        }

    def next_update(self, timestep, states):
        update = {"murein_state": {}}
        
        # CellWall normally calculates the un/incorporated murein counts after
        # division but before this runs. When running inside an EngineProcess,
        # a new instance of Engine is created after division, causing all steps
        # (including this one) to run before any processes (like CellWall). If
        # so, create lattice and calculate un/incorporated murein counts here.
        if states["wall_state"]["lattice"] is None:
            # Make sure that all usable murein is initiailly unincorporated
            unincorporated_monomers = (
                4 * states["total_murein"][self.murein]
                - states["murein_state"]["shadow_murein"]
            )
            incorporated_monomers = 0

            # Get dimensions of the lattice
            length = length_from_volume(states["volume"],
                self.cell_radius * 2).to("micrometer")
            rows, cols = calculate_lattice_size(
                length,
                self.inter_strand_distance,
                self.disaccharide_height,
                self.disaccharide_width,
                self.circumference,
                states["wall_state"]["extension_factor"],
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
            states["murein_state"][
                "incorporated_murein"] = incorporated_monomers
            states["murein_state"][
                "unincorporated_murein"] = unincorporated_monomers
            update.update({
                "wall_state": {"lattice": lattice},
                "murein_state": {"incorporated_murein": incorporated_monomers}
            })

        # New murein to allocate
        new_murein = 4 * states["total_murein"][self.murein] - sum(
            states["murein_state"].values()
        )

        # Calculate fraction of active PBP1a, PBP1b using Hill Equation
        # (calculating prop NOT bound, i.e. 1 - Hill eq value)
        beta_lactam = states["concentrations"][self.beta_lactam]
        active_fraction_1a = 1 / (1 + (beta_lactam / self.K_A_1a) ** self.n_1a)
        active_fraction_1b = 1 / (1 + (beta_lactam / self.K_A_1b) ** self.n_1b)

        # Allocate real vs. shadow murein based on
        # what fraction of PBPs are active
        PBP1A = states["bulk"][self.PBP1A]
        PBP1B = states["bulk"][self.PBP1B]
        total_PBP = PBP1A + PBP1B

        if total_PBP > 0:
            real_new_murein = int(
                round(
                    (
                        active_fraction_1a * (PBP1A / total_PBP)
                        + active_fraction_1b * (PBP1B / total_PBP)
                    )
                    * new_murein
                )
            )
        else:
            real_new_murein = new_murein

        update["murein_state"].update({
            "unincorporated_murein": (
                real_new_murein + states["murein_state"]["unincorporated_murein"]
            ),
            "shadow_murein": (
                new_murein - real_new_murein + states["murein_state"]["shadow_murein"]
            ),
        })

        update["pbp_state"] = {
            "active_fraction_PBP1A": active_fraction_1a,
            "active_fraction_PBP1B": active_fraction_1b,
        }

        return update


def test_pbp_binding():
    # Create process
    params = {"beta_lactam": "ampicillin"}
    process = PBPBinding(params)

    # Create composite with timeline
    initial_murein = 450000
    processes = {"pbp_binding": process}
    topology = {
        "pbp_binding": {
            "total_murein": ("bulk",),
            "murein_state": ("murein_state",),
            "concentrations": ("concentrations",),
            "bulk": ("bulk",),
            "pbp_state": ("pbp_state",),
        }
    }
    add_timeline(
        processes,
        topology,
        {
            "timeline": [
                (
                    time,
                    {
                        ("bulk", "CPD-12261[p]"): int(initial_murein + 1000 * time),
                        ("concentrations", "ampicillin"): (
                            (time - 50) / 10 * units.micromolar
                            if time > 50
                            else 0 * units.micromolar
                        ),
                    },
                )
                for time in range(0, 100)
            ]
        },
    )

    # Run experiment
    settings = {
        "total_time": 100,
        "initial_state": {
            "murein_state": {
                "incorporated_murein": initial_murein * 4,
                "unincorporated_murein": 0,
                "shadow_murein": 0,
            },
            "concentrations": {
                "ampicillin": 0 * units.micromolar,
            },
            "bulk": {
                "CPD-12261[p]": initial_murein,
                "CPLX0-7717[m]": 100,
                "CPLX0-3951[i]": 100,
            },
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
