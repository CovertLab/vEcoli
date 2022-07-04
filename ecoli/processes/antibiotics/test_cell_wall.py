import os

from ecoli.library.create_timeline import add_computed_value, create_timeline_from_csv
from ecoli.library.schema import bulk_schema
from ecoli.processes.antibiotics.cell_wall import CellWall
from vivarium.core.composition import add_timeline, simulate_composite
from vivarium.core.process import Process
from vivarium.plots.simulation_output import plot_variables

DATA = "data/cell_wall/test_murein_21_06_2022_17_42_11.csv"


def create_composite():
    timeline = create_timeline_from_csv(
        DATA,
        {
            "CPD-12261[p]": ("bulk", "CPD-12261[p]"),
            "CPLX0-7717[m]": ("bulk", "CPLX0-7717[m]"),
            "CPLX0-3951[i]": ("bulk", "CPLX0-3951[i]"),
        },
    )
    timeline = add_computed_value(timeline, lambda t, value: {})

    # stub process to ensure updaters exist
    class Stub(Process):
        def __init__(self, parameters=None):
            super().__init__(parameters)

        def ports_schema(self):
            return {
                "bulk_murein": bulk_schema(["CPD-12261[p]"]),
                "PBP": bulk_schema(["CPLX0-7717[m]", "CPLX0-3951[i]"]),
            }

        def next_update(self, timestep, states):
            return {}

    processes = {"stub": Stub({})}
    topology = {"stub": {"bulk_murein": ("bulk",), "PBP": ("bulk",)}}

    add_timeline(processes, topology, timeline)

    return {"processes": processes, "topology": topology}


def output_data(data, filepath="out/processes/cell_wall/test_cell_wall.png"):
    plot_variables(
        data,
        variables=[
            ("bulk", "CPD-12261[p]"),
            ("bulk", "CPLX0-7717[m]"),
            ("bulk", "CPLX0-3951[i]"),
        ],
        out_dir=os.path.dirname(filepath),
        filename=os.path.basename(filepath),
    )


def test_cell_wall():
    composite, initial_state = create_composite()

    settings = {"initial_state": initial_state, "return_raw_data": False, "total_time": 500}
    data = simulate_composite(composite, settings)

    output_data(data)


def main():
    test_cell_wall()


if __name__ == "__main__":
    main()
