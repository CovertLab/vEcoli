from vivarium.core.engine import Engine
from ecoli.composites.ecoli_master import Ecoli
from ecoli.analysis.single.blame import blame_plot


def run_division(agent_id="1", total_time=60):
    """
    Work in progress to get division working
    * TODO -- unique molecules need to be divided between daughter cells!!! This can get sophisticated
    """

    # get initial mass from Ecoli composer
    initial_state = Ecoli({}).initial_state({"initial_state": "vivecoli_t2526"})
    initial_mass = initial_state["listeners"]["mass"]["cell_mass"]
    division_mass = initial_mass + 1
    print(f"DIVIDE AT {division_mass} fg")

    # make a new composer under an embedded path
    config = {
        "log_updates": True,
        "divide": True,
        "agent_id": agent_id,
        "division": {"threshold": division_mass},  # fg
    }
    agent_path = ("agents", agent_id)
    ecoli_composer = Ecoli(config)
    ecoli_composite = ecoli_composer.generate(path=agent_path)

    # make and run the experiment
    experiment = Engine(
        processes=ecoli_composite.processes,
        topology=ecoli_composite.topology,
        initial_state={"agents": {agent_id: initial_state}},
    )
    experiment.update(total_time)

    # retrieve output
    output = experiment.emitter.get_data()
    timeseries = experiment.emitter.get_timeseries()

    # asserts
    initial_agents = output[0.0]["agents"].keys()
    final_agents = output[total_time]["agents"].keys()
    print(f"initial agent ids: {initial_agents}")
    print(f"final agent ids: {final_agents}")
    assert len(final_agents) == 2 * len(initial_agents)

    timeseries["agents"]["1"]["time"] = []
    timeseries["agents"]["10"]["time"] = [12.0, 14.0, 16.0]
    timeseries["agents"]["11"]["time"] = [12.0, 14.0, 16.0]
    for i in range(6):
        timeseries["agents"]["1"]["time"].append(i * 2.0)
    blame_plot(
        timeseries["agents"]["1"],
        experiment.topology["agents"]["10"],
        "out/ecoli_sim/GLT_1.png",
        selected_molecules=["GLT[c]"],
    )
    blame_plot(
        timeseries["agents"]["10"],
        experiment.topology["agents"]["10"],
        "out/ecoli_sim/GLT_10.png",
        selected_molecules=["GLT[c]"],
    )
    blame_plot(
        timeseries["agents"]["11"],
        experiment.topology["agents"]["11"],
        "out/ecoli_sim/GLT_11.png",
        selected_molecules=["GLT[c]"],
    )


if __name__ == "__main__":
    run_division()
