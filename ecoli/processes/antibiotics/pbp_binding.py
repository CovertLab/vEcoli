import numpy as np
from ecoli.library.schema import bulk_schema
from ecoli.processes.registries import topology_registry
from vivarium.core.composition import add_timeline, simulate_composite
from vivarium.core.process import Step
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables


# Register default topology for this process, associating it with process name
NAME = "ecoli-pbp-binding"
TOPOLOGY = {
    "total_murein": ("bulk",),
    "murein_state": ("murein_state",),
    "concentrations": ("concentrations",),
    "bulk": ("bulk",),
    "listeners": ("listeners",),
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
                    # From Curtis et al. 1979 (same values in Fontana et al. 2000)
                    #
                    # PBP1A: 0.25 ug / mL
                    # PBP1B: 2.5 ug / mL
                    #
                    # converted to molar units using molar mass of cephaloridine = 415.488 g/mol
                    "PBP1A": 0.6017020948860136 * units.micromolar,
                    "PBP1B": 6.017020948860137 * units.micromolar,
                },
                "Hill_n": {
                    "PBP1A": 1,
                    "PBP1B": 1,
                },
            },
            "ampicillin": {
                "K_A": {
                    # From Curtis et al. 1979
                    #
                    # PBP1A: 1.4 ug / mL
                    # PBP1B: 3.9 ug / mL
                    #
                    # converted to molar units using molar mass of ampicillin = 349.406 g/mol
                    "PBP1A": 4.00680011 * units.micromolar,
                    "PBP1B": 11.1618003 * units.micromolar,
                },
                "Hill_n": {
                    "PBP1A": 1,
                    "PBP1B": 1,
                },
            },
        },
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

    def ports_schema(self):
        return {
            "total_murein": bulk_schema([self.parameters["murein_name"]]),
            "murein_state": bulk_schema(
                ["incorporated_murein", "unincorporated_murein", "shadow_murein"],
                updater="set",
            ),
            "concentrations": {
                "beta_lactam": {"_default": 0.0 * units.micromolar, "_emit": True},
            },
            "bulk": bulk_schema(list(self.parameters["PBP"].values())),
            "listeners": {
                "active_fraction_PBP1A": {
                    "_default": 0.0,
                    "_updater": "set",
                    "_emit": True,
                },
                "active_fraction_PBP1B": {
                    "_default": 0.0,
                    "_updater": "set",
                    "_emit": True,
                },
            },
        }

    def next_update(self, timestep, states):
        update = {}

        # New murein to allocate
        new_murein = 4 * states["total_murein"][self.murein] - sum(
            states["murein_state"].values()
        )

        # Calculate fraction of active PBP1a, PBP1b using Hill Equation
        # (calculating prop NOT bound, i.e. 1 - Hill eq value)
        beta_lactam = states["concentrations"]["beta_lactam"]
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

        update["murein_state"] = {
            "unincorporated_murein": (
                real_new_murein + states["murein_state"]["unincorporated_murein"]
            ),
            "shadow_murein": (
                new_murein - real_new_murein + states["murein_state"]["shadow_murein"]
            ),
        }

        update["listeners"] = {
            "active_fraction_PBP1A": active_fraction_1a,
            "active_fraction_PBP1B": active_fraction_1b,
        }

        return update


def test_pbp_binding():
    # Create process
    params = {"beta_lactam": "ampicillin"}
    process = PBPBinding(params)

    # Create composite with timeline
    processes = {"pbp_binding": process}
    topology = {
        "pbp_binding": {
            "total_murein": ("bulk",),
            "murein_state": ("murein_state",),
            "concentrations": ("concentrations",),
            "bulk": ("bulk",),
            "listeners": ("listeners",),
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
                        ("bulk", "CPD-12261[p]"): int(3e6 + 1000 * time),
                        ("concentrations", "beta_lactam"): (
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
                "incorporated_murein": int(3e6),
                "unincorporated_murein": 0,
                "shadow_murein": 0,
            },
            "concentrations": {
                "beta_lactam": 0 * units.micromolar,
            },
            "bulk": {
                "CPD-12261[p]": int(3e6),
                "CPLX0-7717[m]": 100,
                "CPLX0-3951[i]": 100,
            },
        },
    }
    data = simulate_composite({"processes": processes, "topology": topology}, settings)

    # Plot output
    fig = plot_variables(
        data,
        variables=[
            ("concentrations", ("beta_lactam", "micromolar")),
            ("bulk", "CPD-12261[p]"),
            ("murein_state", "incorporated_murein"),
            ("murein_state", "unincorporated_murein"),
            ("murein_state", "shadow_murein"),
            ("listeners", ("active_fraction_PBP1A", "dimensionless")),
            ("listeners", ("active_fraction_PBP1B", "dimensionless")),
        ],
    )
    fig.get_axes()[-1].set_ylim(0, 1)
    fig.get_axes()[-2].set_ylim(0, 1)
    fig.savefig("out/processes/pbp_binding/test.png")

    # Validate output data
    total_murein = np.array(data["bulk"]["CPD-12261[p]"])
    incorporated_murein = np.array(data["murein_state"]["incorporated_murein"])
    unincorporated_murein = np.array(data["murein_state"]["unincorporated_murein"])
    shadow_murein = np.array(data["murein_state"]["shadow_murein"])
    assert all(total_murein == incorporated_murein + unincorporated_murein + shadow_murein)


def main():
    test_pbp_binding()


if __name__ == "__main__":
    main()
