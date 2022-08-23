import os
import re
import numpy as np
import pandas as pd
from ecoli.library.cell_wall.column_sampler import geom_sampler, sample_lattice

from ecoli.library.create_timeline import add_computed_value, create_timeline_from_df
from ecoli.processes.antibiotics.cell_wall import CellWall
from vivarium.core.composition import add_timeline, simulate_composite
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables
from vivarium.plots.topology import plot_topology

from ecoli.processes.antibiotics.pbp_binding import PBPBinding

DATA = "data/cell_wall/cell_wall_test_rig_31_07_2022_00_26_44.csv"


def parse_unit_string(unit_str):
    parse_result = re.search(r"!units\[(?P<value>\d+[.]\d+) (?P<units>\w+)\]", unit_str)
    return float(parse_result["value"]) * units.parse_expression(parse_result["units"])


def create_composite(timeline_data):
    # Create timeline process from saved simulation
    timeline = create_timeline_from_df(
        timeline_data,
        {
            "CPD-12261[p]": ("bulk", "CPD-12261[p]"),
            "CPLX0-7717[m]": ("bulk", "CPLX0-7717[m]"),
            "CPLX0-3951[i]": ("bulk", "CPLX0-3951[i]"),
            "Volume": ("cell_global", "volume"),
        },
    )

    # Add cell volume, and account for too much murein
    timeline = add_computed_value(
        timeline,
        lambda t, value: {
            ("cell_global", "volume"): parse_unit_string(
                value[("cell_global", "volume")]
            ),
            ("concentrations", "beta_lactam"): (
                0 * units.micromolar if t < 500 else 9.16 * units.micromolar
            ),
            ("bulk", "CPD-12261[p]"): int(value[("bulk", "CPD-12261[p]")]),
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
            "pbp_state": ("pbp_state",),
            "listeners": ("listeners",),
        },
        "pbp_binding": {
            "total_murein": ("bulk",),
            "murein_state": ("murein_state",),
            "concentrations": ("concentrations",),
            "bulk": ("bulk",),
            "pbp_state": ("pbp_state",),
        },
    }

    add_timeline(processes, topology, timeline)

    return {"processes": processes, "topology": topology}


def output_data(data, filepath="out/processes/cell_wall/test_cell_wall.png"):
    plot_variables(
        data,
        variables=[
            ("concentrations", ("beta_lactam", "micromolar")),
            ("wall_state", "cracked"),
            ("bulk", "CPD-12261[p]"),
            ("bulk", "CPLX0-7717[m]"),
            ("bulk", "CPLX0-3951[i]"),
            ("murein_state", "incorporated_murein"),
            ("murein_state", "unincorporated_murein"),
            ("murein_state", "shadow_murein"),
            ("wall_state", "stretch_factor"),
            ("pbp_state", ("active_fraction_PBP1A", "dimensionless")),
            ("pbp_state", ("active_fraction_PBP1B", "dimensionless")),
            ("listeners", "porosity"),
        ],
        out_dir=os.path.dirname(filepath),
        filename=os.path.basename(filepath),
    )


def test_cell_wall():
    timeline_data = pd.read_csv(DATA, skipinitialspace=True)

    composite = create_composite(timeline_data)
    plot_topology(
        composite, out_dir="out/processes/cell_wall/", filename="test_rig_topology.png"
    )

    # Create initial state
    initial_murein = int(timeline_data.iloc[0]["CPD-12261[p]"])
    initial_PBP1A = int(timeline_data.iloc[0]["CPLX0-7717[m]"])
    initial_PBP1B = int(timeline_data.iloc[0]["CPLX0-3951[i]"])
    initial_volume = parse_unit_string(timeline_data.iloc[0]["Volume"])
    rng = np.random.default_rng(0)

    initial_lattice = sample_lattice(
        initial_murein * 4, 3050, 599, geom_sampler(rng, 0.058), rng
    )

    initial_state = {
        "bulk": {
            "CPD-12261[p]": initial_murein,
            "CPLX0-7717[m]": initial_PBP1A,
            "CPLX0-3951[i]": initial_PBP1B,
        },
        "murein_state": {
            "incorporated_murein": initial_lattice.sum(),
            "unincorporated_murein": int(4 * initial_murein - initial_lattice.sum()),
            "shadow_murein": 0,
        },
        "wall_state": {"lattice": initial_lattice},
        "cell_global": {"volume": initial_volume},
    }

    settings = {
        "return_raw_data": False,
        "total_time": 2000,
        "initial_state": initial_state,
        "emitter": "database",
    }

    data = simulate_composite(composite, settings)
    output_data(data)


def main():
    test_cell_wall()


if __name__ == "__main__":
    main()
