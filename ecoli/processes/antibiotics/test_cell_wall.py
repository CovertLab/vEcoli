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
from vivarium.plots.topology import plot_topology

from ecoli.processes.antibiotics.pbp_binding import PBPBinding

DATA = "data/cell_wall/test_murein_21_06_2022_17_42_11.csv"


def create_composite():
    # Create timeline process from saved simulation
    timeline = create_timeline_from_csv(
        DATA,
        {
            "CPD-12261[p]": ("bulk", "CPD-12261[p]"),
            "CPLX0-7717[m]": ("bulk", "CPLX0-7717[m]"),
            "CPLX0-3951[i]": ("bulk", "CPLX0-3951[i]"),
        },
    )

    # Add cell volume, and account for too much murein
    timeline = add_computed_value(
        timeline,
        lambda t, value: {
            ("bulk", "CPD-12261[p]"): value[("bulk", "CPD-12261[p]")] // 3,
            ("cell_global", "volume"): (1.06 + t / 1000) * units.fL,
        },
    )

    # Add cell wall process, pbp binding process, and timeline process into composite
    processes = {"cell_wall": CellWall({}), "pbp_binding": PBPBinding({})}
    topology = {
        "cell_wall": {
            "shape": ("cell_global",),
            "bulk_murein": ("bulk",),
            "murein_state": ("murein_state",),
            "PBP": ("bulk",),
            "wall_state": ("wall_state",),
        },
        "pbp_binding": {
            "total_murein": ("bulk",),
            "murein_state": ("murein_state",),
            "concentrations": ("concentrations",),
            "bulk": ("bulk",),
            "listeners": ("listeners",),
        },
    }

    add_timeline(processes, topology, timeline)

    return {"processes": processes, "topology": topology}


def output_data(data, filepath="out/processes/cell_wall/test_cell_wall.png"):
    plot_variables(
        data,
        variables=[
            ("concentrations", "beta_lactam"),
            ("bulk", "CPD-12261[p]"),
            ("bulk", "CPLX0-7717[m]"),
            ("bulk", "CPLX0-3951[i]"),
            ("murein_state", "incorporated_murein"),
            ("murein_state", "unincorporated_murein"),
            ("murein_state", "shadow_murein"),
            ("listeners", "active_fraction_PBP1A"),
            ("listeners", "active_fraction_PBP1B")
        ],
        out_dir=os.path.dirname(filepath),
        filename=os.path.basename(filepath),
    )


def test_cell_wall():
    composite = create_composite()
    plot_topology(
        composite, out_dir="out/processes/cell_wall/", filename="test_rig_topology.png"
    )

    # Create initial state
    df = pd.read_csv(DATA, skipinitialspace=True)
    initial_murein = int(df.loc[0]["CPD-12261[p]"]) // 3  # account for too much murein
    rng = np.random.default_rng(0)
    initial_state = {
        "bulk": {"CPD-12261[p]": initial_murein},
        "murein_state": {
            "incorporated_murein": initial_murein,
            "unincorporated_murein": 0,
            "shadow_murein": 0,
        },
        "wall_state": {
            "lattice": de_novo_lattice(
                initial_murein * 4, 3050, 700, geom_sampler(rng, 0.058), rng
            )
        },
    }

    settings = {
        "return_raw_data": False,
        "total_time": 500,
        "initial_state": initial_state,
    }
    data = simulate_composite(composite, settings)

    output_data(data)


def main():
    test_cell_wall()


if __name__ == "__main__":
    main()
