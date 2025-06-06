"""
=========================
Replication Data Listener
=========================
"""

import numpy as np
from ecoli.library.schema import numpy_schema, listener_schema, attrs
from vivarium.core.process import Step

from ecoli.processes.registries import topology_registry


NAME = "replication_data_listener"
TOPOLOGY = {
    "listeners": ("listeners",),
    "oriCs": ("unique", "oriC"),
    "DnaA_boxes": ("unique", "DnaA_box"),
    "active_replisomes": ("unique", "active_replisome"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}
topology_registry.register(NAME, TOPOLOGY)


class ReplicationData(Step):
    """
    Listener for replication data.
    """

    name = NAME
    topology = TOPOLOGY

    defaults = {"time_step": 1, "emit_unique": False}

    def ports_schema(self):
        return {
            "listeners": {
                "replication_data": listener_schema(
                    {
                        "fork_coordinates": [],
                        "fork_domains": [],
                        "fork_unique_index": [],
                        "number_of_oric": [],
                        "free_DnaA_boxes": [],
                        "total_DnaA_boxes": [],
                    }
                )
            },
            "oriCs": numpy_schema("oriCs", emit=self.parameters["emit_unique"]),
            "active_replisomes": numpy_schema(
                "active_replisomes", emit=self.parameters["emit_unique"]
            ),
            "DnaA_boxes": numpy_schema(
                "DnaA_boxes", emit=self.parameters["emit_unique"]
            ),
            "global_time": {"_default": 0.0},
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def next_update(self, timestep, states):
        fork_coordinates, fork_domains, fork_unique_index = attrs(
            states["active_replisomes"], ["coordinates", "domain_index", "unique_index"]
        )

        (DnaA_box_bound,) = attrs(states["DnaA_boxes"], ["DnaA_bound"])

        fork_coords = np.zeros(1000, dtype=np.float64)
        fork_coords[: len(fork_coordinates)] = fork_coordinates
        fork_doms = np.zeros(1000, dtype=np.int64)
        fork_doms[: len(fork_domains)] = fork_domains
        fork_unique_idx = np.zeros(1000, dtype=np.int64)
        fork_unique_idx[: len(fork_unique_index)] = fork_unique_index
        update = {
            "listeners": {
                "replication_data": {
                    "fork_coordinates": fork_coords,
                    "fork_domains": fork_doms,
                    "fork_unique_index": fork_unique_idx,
                    "number_of_oric": states["oriCs"]["_entryState"].sum(),
                    "total_DnaA_boxes": len(DnaA_box_bound),
                    "free_DnaA_boxes": np.count_nonzero(np.logical_not(DnaA_box_bound)),
                }
            }
        }
        return update
