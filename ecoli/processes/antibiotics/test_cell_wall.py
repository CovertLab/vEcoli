import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vivarium.core.composer import Composite
from vivarium.core.composition import add_timeline
from vivarium.core.engine import Engine
from vivarium.core.serialize import deserialize_value
from vivarium.library.dict_utils import get_value_from_path
from vivarium.library.units import remove_units, units
from vivarium.plots.topology import plot_topology

from ecoli.library.create_timeline import (add_computed_value,
                                           create_timeline_from_df)
from ecoli.processes.antibiotics.cell_wall import CellWall
from ecoli.processes.antibiotics.pbp_binding import PBPBinding
from ecoli.processes.antibiotics.murein_division import MureinDivision


DATA = "data/cell_wall/cell_wall_test_rig_17_09_2022_00_41_51.csv"


def parse_unit_string(unit_str):
    parse_result = re.search(r"!units\[(?P<value>\d+[.]\d+) (?P<units>\w+)\]", unit_str)
    return float(parse_result["value"]) * units.parse_expression(parse_result["units"])


def create_composite(timeline_data, antibiotics=True):
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
            ("concentrations", "ampicillin"): (
                10 * units.micromolar
                if antibiotics and t > 0
                else 0 * units.micromolar
            ),
            ("bulk", "CPD-12261[p]"): int(value[("bulk", "CPD-12261[p]")]),
        },
    )

    # Add cell wall process, pbp binding process, and timeline process into composite
    processes = {
        "cell_wall": CellWall({}),
        "pbp_binding": PBPBinding({}),
        "murein-division": MureinDivision({})
    }
    topology = {
        "cell_wall": {
            "shape": ("cell_global",),
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
            "wall_state": ("wall_state",),
        },
        "murein-division": {
            "total_murein": ("bulk",),
            "murein_state": ("murein_state",),
            "wall_state": ("wall_state",),
        }
    }

    add_timeline(processes, topology, timeline)

    # Create initial state
    initial_murein = int(timeline_data.iloc[0]["CPD-12261[p]"])
    initial_PBP1A = int(timeline_data.iloc[0]["CPLX0-7717[m]"])
    initial_PBP1B = int(timeline_data.iloc[0]["CPLX0-3951[i]"])
    initial_volume = parse_unit_string(timeline_data.iloc[0]["Volume"])

    initial_state = {
        "bulk": {
            "CPD-12261[p]": initial_murein,
            "CPLX0-7717[m]": initial_PBP1A,
            "CPLX0-3951[i]": initial_PBP1B,
        },
        "murein_state": {
            "incorporated_murein": 0,
            "unincorporated_murein": initial_murein * 4,
            "shadow_murein": 0,
        },
        "wall_state": {},
        "cell_global": {"volume": initial_volume},
    }

    return Composite(
        {"processes": processes, "topology": topology, "state": initial_state}
    )


def output_data(data, filepath):
    variables = [
        ("concentrations", "ampicillin"),
        ("wall_state", "cracked"),
        ("bulk", "CPD-12261[p]"),
        ("bulk", "CPLX0-7717[m]"),
        ("bulk", "CPLX0-3951[i]"),
        ("murein_state", "incorporated_murein"),
        ("murein_state", "unincorporated_murein"),
        ("murein_state", "shadow_murein"),
        ("wall_state", "extension_factor"),
        ("wall_state", "lattice_cols"),
        ("listeners", "hole_size_distribution"),
        ("pbp_state", "active_fraction_PBP1A"),
        ("pbp_state", "active_fraction_PBP1B"),
        ("listeners", "porosity"),
    ]

    fig, axs = plt.subplots(len(variables), 1)
    T = sorted(data.keys())
    for i, path in enumerate(variables):
        # Get the data at the specified path for each timepoint t
        var_data = [
            remove_units(deserialize_value(get_value_from_path(data[t], path)))
            for t in T
        ]
        if path == ("listeners", "hole_size_distribution"):
            var_data = [len(data_at_time) for data_at_time in var_data]

        axs[i].plot(T, var_data)
        axs[i].set_title(path[-1])

    fig.set_size_inches(6, 1.5 * len(variables))
    fig.tight_layout()
    fig.savefig(filepath)


def test_cell_wall():
    total_time = 1200
    timeline_data = pd.read_csv(DATA, skipinitialspace=True)

    for antibiotics in [False, True]:
        # Create and run experiment
        composite = create_composite(timeline_data, antibiotics=antibiotics)
        plot_topology(
            composite,
            out_dir="out/processes/cell_wall/",
            filename="test_rig_topology.png",
        )

        # Get and plot data
        sim = Engine(composite=composite)
        sim.update(total_time)
        data = sim.emitter.get_data()
        output_data(
            data,
            "out/processes/cell_wall/test_cell_wall ["
            f"{'no' if not antibiotics else ''} antibiotics].png",
        )

        # Validate data
        time = sorted(data.keys())
        assert data[time[-1]]["wall_state"]["cracked"] == antibiotics

        del composite
        del sim
        del data


def main():
    test_cell_wall()


if __name__ == "__main__":
    main()
