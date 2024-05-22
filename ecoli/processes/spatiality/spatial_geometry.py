"""
================
Spatial Geometry
================

SpatialGeometry :term:`Process`  calculates geometric parameters of nodes a
nd edges utilized in diffusion_network.py. This process assumes E. coli is
a spherocylinder with constant radius. This process is currently unfinished.
This process assumes that there are properties of the nodes that specify
which type of compartment it is such that the geometry rules below apply.
This is not currently implemented in ecoli_spatial.py in vivarium-ecoli.
Much of the geometry rules have been specified, however it should be
checked to see if it leads to consistent results.

References:
 - width of 0.73 um: Cluzel et al., Nucleic Acids Research (2008)
 - strain: K-12 Frag1
 - periplasm width of 21 nm: Matias et al., J Bacteriology (2003)
 - cell density of 1100 g/L (from WCM): Baldwin et al., Arch microbiol. (1995)

.. WARNING:: This process is unfinished.
"""

import os
import numpy as np

from vivarium.core.process import Deriver
from vivarium.core.composition import (
    simulate_process,
    PROCESS_OUT_DIR,
)
from ecoli.library.schema import array_from


NAME = "spatial_geometry"
WIDTH = 0.73  # in um
PERIPLASM_WIDTH = 21  # in nm
DENSITY = 1100  # fg/um^3


class SpatialGeometry(Deriver):
    name = NAME
    defaults = {
        "nodes": [
            "nucleoid",
            "periplasm",
        ],
        "edges": {},
        "density": DENSITY,
        "mw": {},
        "width": WIDTH,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.nodes = np.asarray(self.parameters["nodes"])
        self.edges = self.parameters["edges"]
        self.mw = self.parameters["mw"]
        self.molecule_ids = self.parameters["mw"].keys()
        self.density = self.parameters["density"]

    def initial_state(self, config=None):
        pass

    def ports_schema(self):
        node_schema = {
            node_id: {
                "volume": {
                    "_default": 1.0,
                },
                "length": {
                    "_default": 1.0,
                },
                "radius": {
                    "_default": 1.0,
                },
                "molecules": {
                    "*": {
                        "_default": 0,
                    }
                },
            }
            for node_id in self.parameters["nodes"]
        }
        edge_schema = {
            edge_id: {
                "nodes": [],
                "cross_sectional_area": 1.0,
            }
            for edge_id in self.parameters["edges"].keys()
        }

        return {"nodes": node_schema, "edges": edge_schema}

    def next_update(self, timestep, states):
        volume = np.zeros(len(self.nodes))
        nodes = states["nodes"]
        length = [nodes[node_id]["length"] for node_id in self.nodes]
        cross_sectional_area = np.zeros(len(self.edges))
        inner_radius = nodes["nucleoid"]["radius"]
        outer_radius = nodes["periplasm"]["radius"] + inner_radius

        for i, node_id in enumerate(self.nodes):
            # if nucleoid or nucleoid region of membrane
            nucleoid = True
            volume[i] = sum(array_from(nodes[node_id]["molecules"])) * self.density
            if nucleoid:
                length[i] = volume[i] / (np.pi * inner_radius**2)
        for i, edge_id in enumerate(self.edges):
            nodes = states[edge_id]["nodes"]
            nucleoid = "nucleoid" in nodes
            cytosol = "cytosol_front" or "cytosol_rear" in nodes
            periplasm = "periplasm" in nodes
            cytosol_inner_membrane = False
            cytosol_outer_membrane = False
            nucleoid_inner_membrane = False
            nucleoid_outer_membrane = False

            # series of if statements for each interface
            if nucleoid and cytosol:
                cross_sectional_area[i] = np.pi * inner_radius**2
                continue
                # find if node is membrane
            if nucleoid and nucleoid_inner_membrane:
                node_id = "membrane"
                cross_sectional_area[i] = (
                    2 * np.pi * inner_radius * states["nodes"][node_id]["length"]
                )
                continue
            if cytosol and cytosol_inner_membrane:
                cross_sectional_area[i] = 2 * np.pi * inner_radius**2
                continue
            if periplasm and nucleoid_inner_membrane:
                cross_sectional_area[i] = (
                    2 * np.pi * inner_radius * states["nodes"][node_id]["length"]
                )
                continue
            if periplasm and nucleoid_outer_membrane:
                cross_sectional_area[i] = (
                    2 * np.pi * outer_radius * states["nodes"][node_id]["length"]
                )
                continue
            if periplasm and cytosol_inner_membrane:
                cross_sectional_area[i] = 2 * np.pi * inner_radius**2
                continue
            if periplasm and cytosol_outer_membrane:
                cross_sectional_area[i] = 2 * np.pi * outer_radius**2
                continue

        node_update = {
            node_id: {
                "volume": volume[np.where(self.nodes == node_id)][0],
                "length": volume[np.where(self.nodes == node_id)][0],
            }
            for node_id in self.nodes
        }
        edge_update = {}
        return {**node_update, **edge_update}


# functions to configure and run the process
def run_spatial_geometry_process():
    """
    Run a simulation of the process.

    Returns:
        The simulation output.
    """

    # initialize the process by passing in parameters
    parameters = {}
    spatial_geometry_process = SpatialGeometry(parameters)

    # declare the initial state, mirroring the ports structure
    initial_state = {}

    # run the simulation
    sim_settings = {"total_time": 10, "initial_state": initial_state}
    output = simulate_process(spatial_geometry_process, sim_settings)

    return output


def test_spatial_geometry_process(return_data=False):
    """
    Test that the process runs correctly.

    This will be executed by pytest.
    """
    output = run_spatial_geometry_process()
    # TODO(vivarium): Add assert statements to ensure correct performance.

    if return_data:
        return output


def main():
    """Simulate the process and plot results."""
    # make an output directory to save plots
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output = test_spatial_geometry_process(return_data=True)
    assert output is not None


if __name__ == "__main__":
    main()
