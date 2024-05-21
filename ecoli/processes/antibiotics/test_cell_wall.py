import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vivarium.core.composer import Composite
from vivarium.core.engine import Engine
from vivarium.library.dict_utils import get_value_from_path
from vivarium.library.units import units
from vivarium.plots.topology import plot_topology

from ecoli.library.create_timeline import (
    add_computed_value_bulk,
    create_bulk_timeline_from_df,
)
from ecoli.processes.antibiotics.cell_wall import CellWall
from ecoli.processes.antibiotics.pbp_binding import PBPBinding
from ecoli.processes.antibiotics.murein_division import MureinDivision
from ecoli.processes.bulk_timeline import BulkTimelineProcess


DATA = "data/cell_wall/cell_wall_test_rig_17_09_2022_00_41_51.csv"


def parse_unit_string(unit_str):
    parse_result = re.search(r"!units\[(?P<value>\d+[.]\d+) (?P<units>\w+)\]", unit_str)
    return float(parse_result["value"]) * units.parse_expression(parse_result["units"])


def create_composite(timeline_data, antibiotics=True):
    # Create timeline process from saved simulation
    timeline = create_bulk_timeline_from_df(
        timeline_data,
        {
            "CPD-12261[p]": ("bulk", "CPD-12261[p]"),
            "CPLX0-7717[m]": ("bulk", "CPLX0-7717[m]"),
            "CPLX0-3951[i]": ("bulk", "CPLX0-3951[i]"),
            "Volume": ("cell_global", "volume"),
        },
    )

    # Add cell volume, and account for too much murein
    timeline = add_computed_value_bulk(
        timeline,
        lambda t, value: {
            ("cell_global", "volume"): parse_unit_string(
                value[("cell_global", "volume")]
            ),
            ("concentrations", "ampicillin"): (
                10 * units.micromolar if antibiotics and t > 0 else 0 * units.micromolar
            ),
            ("bulk", "CPD-12261[p]"): int(value[("bulk", "CPD-12261[p]")]),
        },
    )

    # Add cell wall process, pbp binding process, and timeline process into composite
    processes = {
        "cell_wall": CellWall({}),
        "pbp_binding": PBPBinding({}),
        "murein-division": MureinDivision({}),
        "bulk-timeline": BulkTimelineProcess(timeline),
    }
    topology = {
        "cell_wall": {
            "shape": ("cell_global",),
            "murein_state": ("murein_state",),
            "bulk": ("bulk",),
            "wall_state": ("wall_state",),
            "pbp_state": ("pbp_state",),
            "listeners": ("listeners",),
        },
        "pbp_binding": {
            "murein_state": ("murein_state",),
            "concentrations": ("concentrations",),
            "bulk": ("bulk",),
            "pbp_state": ("pbp_state",),
            "wall_state": ("wall_state",),
            "volume": ("cell_global", "volume"),
            "first_update": (
                "first_update",
                "pbp_binding",
            ),
        },
        "murein-division": {
            "bulk": ("bulk",),
            "murein_state": ("murein_state",),
            "wall_state": ("wall_state",),
            "first_update": (
                "first_update",
                "murein_division",
            ),
        },
        "bulk-timeline": {
            "bulk": ("bulk",),
            "cell_global": ("cell_global",),
            "concentrations": ("concentrations",),
            "global": ("global",),
        },
    }

    # Create initial state
    initial_murein = int(timeline_data.iloc[0]["CPD-12261[p]"])
    initial_PBP1A = int(timeline_data.iloc[0]["CPLX0-7717[m]"])
    initial_PBP1B = int(timeline_data.iloc[0]["CPLX0-3951[i]"])
    initial_volume = parse_unit_string(timeline_data.iloc[0]["Volume"])

    initial_state = {
        "bulk": np.array(
            [
                ("CPD-12261[p]", initial_murein),
                ("CPLX0-7717[m]", initial_PBP1A),
                ("CPLX0-3951[i]", initial_PBP1B),
                ("CPLX0-8300[c]", 0),
            ],
            dtype=[("id", "U40"), ("count", int)],
        ),
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
    for i, path in enumerate(variables):
        # Get the data at the specified path
        var_data = get_value_from_path(data, path)
        if path == ("listeners", "hole_size_distribution"):
            var_data = [len(data_at_time) for data_at_time in var_data]

        axs[i].plot(data["time"], var_data)
        axs[i].set_title(path[-1])

    fig.set_size_inches(6, 1.5 * len(variables))
    fig.tight_layout()
    fig.savefig(filepath)


def test_cell_wall():
    total_time = 600
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
        data = sim.emitter.get_timeseries()
        data["cell_global"]["volume"] = data["cell_global"].pop(
            ("volume", "femtoliter")
        )
        data["concentrations"]["ampicillin"] = data["concentrations"].pop(
            ("ampicillin", "micromolar")
        )
        data["pbp_state"]["active_fraction_PBP1A"] = data["pbp_state"].pop(
            ("active_fraction_PBP1A", "dimensionless")
        )
        data["pbp_state"]["active_fraction_PBP1B"] = data["pbp_state"].pop(
            ("active_fraction_PBP1B", "dimensionless")
        )
        bulk_array = np.array(data["bulk"])
        data["bulk"] = {
            bulk_id: bulk_array[:, i]
            for i, bulk_id in enumerate(composite.state["bulk"]["id"])
        }
        output_data(
            data,
            "out/processes/cell_wall/test_cell_wall ["
            f"{'no' if not antibiotics else ''} antibiotics].png",
        )

        # Validate data
        assert data["wall_state"]["cracked"][-1] == antibiotics

        del composite
        del sim
        del data


def main():
    test_cell_wall()


if __name__ == "__main__":
    main()
