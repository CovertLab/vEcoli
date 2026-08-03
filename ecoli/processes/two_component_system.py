"""
====================
Two Component System
====================

This process models the phosphotransfer reactions of signal transduction pathways.

Specifically, phosphate groups are transferred from histidine kinases to response regulators
and back in response to counts of ligand stimulants.
"""

import numpy as np

from ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts

from wholecell.utils import units
from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = "ecoli-two-component-system"
TOPOLOGY = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
}
topology_registry.register(NAME, TOPOLOGY)

# --- BEGIN water-floor fix (added alongside the ArcA/PhoP WATER_REF_CONC
# calibration in reconstruction/ecoli/dataclasses/process/two_component_system.py) ---
#
# calculate_request()'s WATER[c] request is normally computed from the real,
# short timestep (~1-2 s), which legitimately rounds to ~0 net water
# turnover for ArcA/PhoP. But evolve_state()'s fallback branch (triggered
# whenever an unrelated molecule, e.g. a low-copy histidine kinase complex,
# is allocated less than requested) reuses that same allocation while
# integrating over a much longer 10000 s nominal timestep, where ArcA/PhoP's
# now-real dephosphorylation flux (see WATER_REF_CONC) needs non-negligible
# water. With 0 water allocated, that fallback integration can dip
# fractionally negative and raise "Solution to ODE ... has negative values."
#
# WATER_REQUEST_FLOOR reserves a small water buffer every timestep so the
# fallback is never starved of it. Trivial relative to the real WATER[c]
# pool (~1e9-1e10 molecules), so this does not meaningfully reduce water
# available to any other process.
WATER_REQUEST_FLOOR = 5000
# --- END water-floor fix ---


class TwoComponentSystem(PartitionedProcess):
    """Two Component System PartitionedProcess"""

    name = NAME
    topology = TOPOLOGY
    defaults = {
        "jit": False,
        "n_avogadro": 0.0,
        "cell_density": 0.0,
        "moleculesToNextTimeStep": (
            lambda counts, volume, avogadro, timestep, random, method, min_step, jit: (
                [],
                [],
            )
        ),
        "moleculeNames": [],
        "seed": 0,
    }

    # Constructor
    def __init__(self, parameters):
        super().__init__(parameters)

        # Simulation options
        self.jit = self.parameters["jit"]

        # Get constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Create method
        self.moleculesToNextTimeStep = self.parameters["moleculesToNextTimeStep"]

        # Build views
        self.moleculeNames = self.parameters["moleculeNames"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.molecule_idx = None

        # --- water-floor fix: locate WATER[c] within moleculeNames so
        # calculate_request() can apply WATER_REQUEST_FLOOR to it below.
        water_matches = np.where(np.asarray(self.moleculeNames) == "WATER[c]")[0]
        self.water_request_idx = water_matches[0] if water_matches.size else None
        # --- end water-floor fix ---

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "listeners": {"mass": {"cell_mass": {"_default": 0}}},
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        # At t=0, convert all strings to indices
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.moleculeNames, states["bulk"]["id"]
            )

        # Get molecule counts
        moleculeCounts = counts(states["bulk"], self.molecule_idx)

        # Get cell mass and volume
        cellMass = (states["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        self.cellVolume = cellMass / self.cell_density

        # Solve ODEs to next time step using the BDF solver through solve_ivp.
        # Note: the BDF solver has been empirically tested to be the fastest
        # solver for this setting among the list of solvers that can be used
        # by the scipy ODE suite.
        self.molecules_required, self.all_molecule_changes = (
            self.moleculesToNextTimeStep(
                moleculeCounts,
                self.cellVolume,
                self.n_avogadro,
                states["timestep"],
                self.random_state,
                method="BDF",
                jit=self.jit,
            )
        )

        # --- water-floor fix: see WATER_REQUEST_FLOOR comment above. Reserve
        # a small water buffer regardless of what this timestep's short-window
        # calculation requested, so evolve_state()'s fallback (which reuses
        # this allocation over a much longer nominal timestep) isn't starved.
        if self.water_request_idx is not None:
            self.molecules_required[self.water_request_idx] = max(
                self.molecules_required[self.water_request_idx], WATER_REQUEST_FLOOR
            )
        # --- end water-floor fix ---

        requests = {"bulk": [(self.molecule_idx, self.molecules_required.astype(int))]}
        return requests

    def evolve_state(self, timestep, states):
        moleculeCounts = counts(states["bulk"], self.molecule_idx)
        # Check if any molecules were allocated fewer counts than requested
        if (self.molecules_required > moleculeCounts).any():
            _, self.all_molecule_changes = self.moleculesToNextTimeStep(
                moleculeCounts,
                self.cellVolume,
                self.n_avogadro,
                10000,
                self.random_state,
                method="BDF",
                min_time_step=states["timestep"],
                jit=self.jit,
            )
        # Increment changes in molecule counts
        update = {"bulk": [(self.molecule_idx, self.all_molecule_changes.astype(int))]}

        return update
