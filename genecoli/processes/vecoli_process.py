from pathlib import Path
import datetime
import warnings
from pathlib import Path
from urllib import parse

from bigraph_schema.core import Core
from process_bigraph import allocate_core
from process_bigraph import Composite
from process_bigraph import Process as PbgProcess
from vivarium.core.engine import Engine

from genecoli.processes.ecoli_types import ECOLI_TYPES as ECOLI_TYPES_REPRESENTATION
from ecoli.experiments.ecoli_master_sim import EcoliSim, SimConfig
from ecoli.library.schema import not_a_process


class VEcoliProcess(PbgProcess):
    config_schema = {"config_path": "string"}

    def initialize(self, config):
        self.simulation = initialize_simulation(config_path=config["config_path"])
        self.t = 0

    def initial_state(self):
        y_0: dict = query_simulation(sim=self.simulation)
        return {
            "exchange": y_0["environment"]["exchange"],
            "mass": y_0["listeners"]["mass"],
            "t": self.simulation.ecoli_experiment.global_time,
        }

    def inputs(self):
        return {"environment": "tree[integer]"}

    def outputs(self):
        return {"exchange": "tree[integer]", "mass": "tree[float]", "t": "float"}

    def update(self, state, interval):
        engine: Engine = self.simulation.ecoli_experiment
        if engine is None:
            raise RuntimeError(
                "Build the composite by calling build_ecoli() \
                before updating!"
            )

        env_input = state["environment"]
        state_update = {"agents": {"0": {"environment": env_input}}}
        engine.state.set_value(state_update)

        self.simulation.update_experiment(interval)
        self.t = engine.global_time

        y_i = engine.state.get_value(condition=not_a_process)["agents"]["0"]

        return {"exchange": y_i["environment"]["exchange"], "mass": y_i["listeners"]["mass"], "t": self.t}


def initialize_simulation(config_path: str | None = None, sim_config: SimConfig | None = None) -> EcoliSim:
    simulation: EcoliSim = _create_simulation(config_path=config_path, config=sim_config)
    # validate initialization
    if simulation.ecoli is None:
        raise RuntimeError(
            "Build the composite by calling build_ecoli() \
            before calling run()."
        )

    # initialize experiment config
    metadata = simulation.get_metadata()
    metadata["output_metadata"] = simulation.output_metadata()

    # make the experiment
    if isinstance(simulation.emitter, str):
        simulation.emitter_config = {"type": simulation.emitter}
        if simulation.emitter_arg is not None:
            for key, value in simulation.emitter_arg.items():
                simulation.emitter_config[key] = value
        if simulation.emitter == "parquet":
            raise RuntimeError("You cannot specify a parquet emitter for now...")

    experiment_config = {
        "description": simulation.description,
        "metadata": metadata,
        "processes": simulation.ecoli.processes,
        "steps": simulation.ecoli.steps,
        "flow": simulation.ecoli.flow,
        "topology": simulation.ecoli.topology,
        "initial_state": simulation.generated_initial_state,
        "progress_bar": simulation.progress_bar,
        "emit_topology": simulation.emit_topology,
        "emit_processes": simulation.emit_processes,
        "emit_config": simulation.emit_config,
        "emitter": simulation.emitter_config,
        "initial_global_time": simulation.initial_global_time,
    }

    if simulation.experiment_id:
        # Store backup of base experiment ID,
        # in case multiple experiments are run in a row
        # with suffix_time = True.
        if not simulation.experiment_id_base:
            simulation.experiment_id_base = simulation.experiment_id
        if simulation.suffix_time:
            simulation.experiment_id = datetime.now().strftime(f"{simulation.experiment_id_base}_%Y%m%d-%H%M%S")
        # Special characters can break Hive partitioning so do not allow them
        if simulation.experiment_id != parse.quote_plus(simulation.experiment_id):
            raise TypeError(
                "Experiment ID cannot contain special characters"
                f"that change the string when URL quoted: {simulation.experiment_id}"
                f" != {parse.quote_plus(simulation.experiment_id)}"
            )
        experiment_config["experiment_id"] = simulation.experiment_id

    experiment_config["profile"] = simulation.profile

    # configure Engine
    # Since unique numpy updater is an class method, internal
    # deepcopying in vivarium-core causes this warning to appear
    warnings.filterwarnings(
        "ignore",
        message="Incompatible schema "
        "assignment at .+ Trying to assign the value <bound method "
        r"UniqueNumpyUpdater\.updater .+ to key updater, which already "
        r"has the value <bound method UniqueNumpyUpdater\.updater",
    )
    simulation.ecoli_experiment = Engine(**experiment_config)
    # Only emit designated stores if specified
    if simulation.config["emit_paths"]:
        simulation.ecoli_experiment.state.set_emit_values([tuple()], False)
        simulation.ecoli_experiment.state.set_emit_values(
            simulation.config["emit_paths"],
            True,
        )
    # Clean up unnecessary references
    # self.generated_initial_state = None
    # self.ecoli_experiment.initial_state = None
    # del metadata, experiment_config
    # self.ecoli = None
    return simulation


def query_simulation(sim: EcoliSim):
    return sim.ecoli_experiment.state.get_value(condition=not_a_process)["agents"]["0"]


def _create_simulation(
    config_path: str | None = None,
    config: SimConfig | None = None,
    **config_overrides
) -> EcoliSim:
    def _new_ecoli(config, config_path):
        if config_path is not None:
            if not Path(config_path).exists():
                raise ValueError(f"You must pass a valid config path, not: {config_path}")
            return EcoliSim.from_file(filepath=config_path)
        if config is not None:
            return EcoliSim(config.to_dict())
        return None

    sim: EcoliSim | None = _new_ecoli(config=config, config_path=config_path)
    if sim is None:
        raise RuntimeError("You must pass either a valid config path or config instance")

    # parameterize sim config
    if len(config_overrides):
        sim.config.update(config_overrides)

    # build vivarium ecoli
    sim.build_ecoli()
    print("Ecoli has been built!")
    return sim


def test_vecoli_composition() -> None:
    config_path = Path(__file__).parent.parent / "ecoli_configs" / "single_cell.json"
    config = {"config_path": str(config_path)}
    state = {
        "ecoli_0": {
            "_type": "process",
            "address": "local:vecoli-process",
            "config": config,
            "inputs": {"environment": ["environment_store"]},
            "outputs": {
                "exchange": ["exchange_store_0"],
                "mass": ["mass_store_0"],
                "t": ["t_store_0"],  # sanity check
            },
        },
        "ecoli_1": {
            "_type": "process",
            "address": "local:vecoli-process",
            "config": config,
            "inputs": {"environment": ["environment_store"]},
            "outputs": {"exchange": ["exchange_store_1"], "mass": ["mass_store_1"], "t": ["t_store_1"]},
        },
        "ecoli_2": {
            "_type": "process",
            "address": "local:vecoli-process",
            "config": config,
            "inputs": {"environment": ["environment_store"]},
            "outputs": {"exchange": ["exchange_store_2"], "mass": ["mass_store_2"], "t": ["t_store_2"]},
        },
    }
    bridge = {
        "outputs": {
            "environment": ["environment_store"],
            "exchange_e0": ["exchange_store_0"],
            "mass_e0": ["mass_store_0"],
            "t_e0": ["t_store_0"],
            "exchange_e1": ["exchange_store_1"],
            "mass_e1": ["mass_store_1"],
            "t_e1": ["t_store_1"],
            "exchange_e2": ["exchange_store_2"],
            "mass_e2": ["mass_store_2"],
            "t_e2": ["t_store_2"],
        }
    }
    composite = Composite(config={"state": state, "bridge": bridge}, core=core)
    composite.save("/Users/alexanderpatrie/sms/genEcoli/colony_state.json", state=True)
    composite.save(
        "/Users/alexanderpatrie/sms/genEcoli/colony_state_with_schema.json", schema=True, state=True
    )
    composite.run(2)

    results = composite.read_bridge()
    assert list(results.keys()) == [
        "exchange_e0",
        "mass_e0",
        "t_e0",
        "exchange_e1",
        "mass_e1",
        "t_e1",
        "exchange_e2",
        "mass_e2",
        "t_e2",
    ]


def get_core() -> Core:
    return allocate_core()


def initialize_core() -> Core:
    c = get_core()
    c.register_types(ECOLI_TYPES_REPRESENTATION)
    c.register_link("vecoli-process", VEcoliProcess)
    return c


core = initialize_core()

