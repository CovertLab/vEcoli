import os
import numpy as np
import pandas as pd
from ecoli.library.cell_wall.column_sampler import geom_sampler
from ecoli.library.cell_wall.lattice import de_novo_lattice

from ecoli.library.create_timeline import add_computed_value, create_timeline_from_csv
from ecoli.library.schema import bulk_schema
from ecoli.processes.antibiotics.cell_wall import CellWall
from vivarium.core.composition import add_timeline, simulate_composite
from vivarium.core.process import Process
from vivarium.library.units import units
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
    timeline = add_computed_value(
        timeline,
        lambda t, value: {
            ("murein_state", "incorporated_murein"): value[("bulk", "CPD-12261[p]")],
            ("cell_global", "volume"): (1 + t / 1000) * units.fL,
        },
    )

    processes = {"cell_wall": CellWall({})}
    topology = {
        "cell_wall": {
            "shape": ("cell_global",),
            "bulk_murein": ("bulk",),
            "murein_state": ("murein_state",),
            "PBP": ("bulk",),
            "wall_state": ("wall_state",),
        }
    }

    add_timeline(processes, topology, timeline)

    return {"processes": processes, "topology": topology}


def output_data(data, filepath="out/processes/cell_wall/test_cell_wall.png"):
    plot_variables(
        data,
        variables=[
            ("bulk", "CPD-12261[p]"),
            ("bulk", "CPLX0-7717[m]"),
            ("bulk", "CPLX0-3951[i]"),
            ("murein_state", "incorporated_murein"),
            ("murein_state", "unincorporated_murein"),
        ],
        out_dir=os.path.dirname(filepath),
        filename=os.path.basename(filepath),
    )


def test_cell_wall():
    composite = create_composite()

    # Create initial state
    df = pd.read_csv(DATA, skipinitialspace=True)
    initial_murein = int(df.loc[0]["CPD-12261[p]"])
    rng = np.random.default_rng(0)
    inital_state = {
        "bulk": {"CPD-12261[p]": initial_murein},
        "wall_state": {
            "lattice": de_novo_lattice(
                initial_murein * 4, 3050, 700, geom_sampler(rng, 0.058), rng
            )
        },
    }

    settings = {
        "return_raw_data": False,
        "total_time": 500,
        "inital_state": inital_state,
    }
    data = simulate_composite(composite, settings)

    output_data(data)


def main():
    test_cell_wall()


if __name__ == "__main__":
    main()
