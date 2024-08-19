import os

from vivarium.library.units import units
from vivarium.core.composer import Composer
from vivarium.core.composition import (
    composer_in_experiment,
    COMPOSITE_OUT_DIR,
)
from vivarium.plots.agents_multigen import plot_agents_multigen

# processes
from vivarium.processes.growth_rate import GrowthRate
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.meta_division import MetaDivision
from ecoli.processes.environment.derive_globals import DeriveGlobals
from ecoli.processes.environment.exchange import Exchange


NAME = "grow_divide"


GROW_DIVIDE_DEFAULTS = {
    "growth": {
        "variables": ["mass"],
    },
    "divide_condition": {"threshold": 2000.0 * units.fg},
    "boundary_path": ("boundary",),
    "agents_path": (
        "..",
        "..",
        "agents",
    ),
    "_schema": {
        "growth": {
            "variables": {
                "mass": {
                    "_default": 1000.0 * units.fg,
                    "_divider": "split",
                }
            }
        }
    },
}


class GrowDivide(Composer):
    defaults = GROW_DIVIDE_DEFAULTS

    def generate_processes(self, config):
        # division config
        agent_id = config["agent_id"]
        division_config = dict(
            config.get("division", {}),
            agent_id=agent_id,
            composer=self,
        )

        return {
            "growth": GrowthRate(config["growth"]),
            "globals_deriver": DeriveGlobals(),
            "divide_condition": DivideCondition(config["divide_condition"]),
            "division": MetaDivision(division_config),
        }

    def generate_topology(self, config):
        boundary_path = config["boundary_path"]
        agents_path = config["agents_path"]
        return {
            "growth": {
                "variables": boundary_path,
                "rates": ("rates",),
            },
            "globals_deriver": {"global": boundary_path},
            "divide_condition": {
                "variable": boundary_path + ("mass",),
                "divide": boundary_path + ("divide",),
            },
            "division": {"global": boundary_path, "agents": agents_path},
        }


class GrowDivideExchange(GrowDivide):
    name = "grow_divide_exchange"
    defaults = GROW_DIVIDE_DEFAULTS
    defaults.update(
        {
            "exchange": {
                "molecules": ["A"],
            },
            "fields_path": ("..", "..", "fields"),
            "dimensions_path": (
                "..",
                "..",
                "dimensions",
            ),
        }
    )

    def generate_processes(self, config):
        processes = super().generate_processes(config)

        added_processes = {
            "exchange": Exchange(config["exchange"]),
        }
        processes.update(added_processes)
        return processes

    def generate_topology(self, config):
        topology = super().generate_topology(config)

        boundary_path = config["boundary_path"]

        added_topology = {
            "exchange": {
                "exchanges": boundary_path + ("exchanges",),
                "external": boundary_path + ("external",),
                "internal": ("internal",),
            },
        }
        topology.update(added_topology)
        return topology


def test_grow_divide(total_time=2000, return_data=False):
    agent_id = "0"
    composite = GrowDivide(
        {
            "agent_id": agent_id,
            "growth": {
                "growth_rate": 0.006,  # very fast growth
                "default_growth_noise": 1e-3,
            },
        }
    )

    initial_state = {"agents": {agent_id: {"global": {"mass": 1000 * units.fg}}}}

    settings = {"experiment_id": "grow_divide"}
    experiment = composer_in_experiment(
        composite,
        initial_state=initial_state,
        outer_path=("agents", agent_id),
        settings=settings,
    )

    experiment.update(total_time)
    output = experiment.emitter.get_data_unitless()

    # assert division occurred
    assert list(output[0.0]["agents"].keys()) == [agent_id]
    assert agent_id not in list(output[total_time]["agents"].keys())
    assert len(output[0.0]["agents"]) == 1
    assert len(output[total_time]["agents"]) > 1

    if return_data:
        return output


def test_grow_divide_exchange(total_time=2000, return_data=False):
    agent_id = "0"
    molecule_id = "A"
    composite = GrowDivideExchange(
        {
            "agent_id": agent_id,
            "growth": {
                "growth_rate": 0.006,  # very fast growth
                "default_growth_noise": 1e-3,
            },
            "exchange": {
                "molecules": [molecule_id],
            },
        }
    )

    initial_state = {
        "agents": {
            agent_id: {
                "global": {"mass": 1000 * units.fg},
                "external": {molecule_id: 10.0 * units.mmol / units.L},
                "internal": {molecule_id: 0.0 * units.mmol / units.L},
            }
        }
    }

    settings = {"experiment_id": "grow_divide_exchange"}
    experiment = composer_in_experiment(
        composite,
        initial_state=initial_state,
        outer_path=("agents", agent_id),
        settings=settings,
    )

    experiment.update(total_time)
    output = experiment.emitter.get_data_unitless()

    # assert
    # external starts at 1, goes down until death, and then back up
    # internal does the inverse
    assert list(output[0.0]["agents"].keys()) == [agent_id]
    assert agent_id not in list(output[total_time]["agents"].keys())
    assert len(output[0.0]["agents"]) == 1
    assert len(output[total_time]["agents"]) > 1

    if return_data:
        return output


def main():
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # if grow_divide:
    output = test_grow_divide(2000, return_data=True)
    plot_settings = {}
    plot_agents_multigen(output, plot_settings, out_dir, "grow_divide")

    # if grow_divide_exchange:
    output = test_grow_divide_exchange(2000, return_data=True)
    plot_settings = {}
    plot_agents_multigen(output, plot_settings, out_dir, "grow_divide_exchange")


if __name__ == "__main__":
    main()
