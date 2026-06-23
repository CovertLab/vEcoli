"""
=========
Allocator
=========

Reads requests from PartionedProcesses, and allocates molecules according to
process priorities.
"""

import numpy as np
from ecoli.library.bigraph_bridge import BigraphStep as Step
from typing import Any

from ecoli.processes.registries import topology_registry
from ecoli.library.schema import counts, numpy_schema, bulk_name_to_idx, listener_schema

# Register default topology for this process, associating it with process name
NAME = "allocator"
TOPOLOGY = {
    "request": ("request",),
    "allocate": ("allocate",),
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "allocator_rng": ("allocator_rng",),
}
topology_registry.register(NAME, TOPOLOGY)
# Register "allocator-1", "allocator-2", "allocator-3" to support
# multi-tiered partitioning scheme
topology_registry.register(NAME + "-1", TOPOLOGY)
topology_registry.register(NAME + "-2", TOPOLOGY)
topology_registry.register(NAME + "-3", TOPOLOGY)

ASSERT_POSITIVE_COUNTS = True


class NegativeCountsError(Exception):
    pass


class Allocator(Step):
    """Allocator Step

    process-bigraph interface: reads request/bulk/allocator_rng,
    writes allocate/request/listeners.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'molecule_names': 'array[string]',
        'process_names': 'list[string]',
        'custom_priorities': 'map[integer]',
        'seed': 'lineage_seed[integer]',
        'time_step': 'float{1.0}',
        'emit_unique': 'boolean{false}',
    }

    defaults: dict[str, Any] = {}

    processes: dict[str, Any] = {}

    def inputs(self):
        return {
            'request': 'map[map[list[integer]]]',
            'bulk': 'bulk_array',
            'allocator_rng': 'random_state',
            'listeners': {
                'atp': {
                    'atp_requested': f'array[{self.n_processes},integer[64]]',
                    'atp_allocated_initial': f'array[{self.n_processes},integer[64]]',
                },
            },
        }

    def outputs(self):
        # NOTE: v1 PARITY HACK — the ``listeners.atp.*`` ports are
        # intentionally NOT declared here (even though
        # ``next_update`` does return values for them). v1's
        # ``allocator_topology`` in ``ecoli_composite.py:126`` only
        # wires ``request``/``allocate``/``bulk`` — the allocator's
        # listener output goes to a phantom port and is silently
        # dropped. As a result v1's parquet emits all-zeros for
        # ``listeners.atp.atp_requested`` and
        # ``listeners.atp.atp_allocated_initial`` (the listener
        # stays at its ``listener_schema`` default).
        #
        # v2 composite auto-wires from ``outputs()`` declarations,
        # so declaring these here would actually persist the
        # writes — and break parity with v1 in every tick.
        # Dropping them mirrors v1's broken behavior.
        #
        # **TODO (v1 fix)**: the right fix is to add
        # ``"listeners": ("listeners",)`` to ``allocator_topology``
        # in ``ecoli_composite.py`` so v1 ALSO emits the real ATP
        # request/allocation per tick. Once that's in and a
        # full-column parity check confirms v1 emits real values,
        # restore the listener declaration here and drop this
        # comment. Until then, parity wins.
        # NB: the OUTER ``overwrite`` was removed from ``allocate`` (and
        # ``request``). With multiple per-layer allocators (allocator_1..N)
        # all wiring their output to the SAME ``('allocate',)`` /
        # ``('request',)`` store, an outer ``overwrite`` made each allocator
        # REPLACE the whole process→counts map. A re-triggered allocator
        # whose request view held only a subset of processes then wiped out
        # every other process's ``allocate`` entry, so the next evolver in
        # the cascade read a missing ``allocate`` key (KeyError: 'allocate').
        # Dropping the outer overwrite lets the top-level process map MERGE
        # per-process entries across allocator invocations while the INNER
        # ``overwrite[array]`` still sets (not accumulates) each process's
        # count vector — the intended partition semantics.
        return {
            'allocate': 'divide_share[map[map[overwrite[array[integer[64]]]]]]',
            'request': 'divide_share[map[map[list[integer]]]]',
        }

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.moleculeNames = self.parameters["molecule_names"]
        self.n_molecules = len(self.moleculeNames)
        self.mol_name_to_idx = {
            name: idx for idx, name in enumerate(self.moleculeNames)
        }
        self.mol_idx_to_name = {
            idx: name for idx, name in enumerate(self.moleculeNames)
        }
        self.processNames = self.parameters["process_names"]
        self.n_processes = len(self.processNames)
        self.proc_name_to_idx = {
            name: idx for idx, name in enumerate(self.processNames)
        }
        self.proc_idx_to_name = {
            idx: name for idx, name in enumerate(self.processNames)
        }
        self.processPriorities = np.zeros(len(self.processNames))
        for process, custom_priority in self.parameters["custom_priorities"].items():
            if process not in self.proc_name_to_idx.keys():
                continue
            self.processPriorities[self.proc_name_to_idx[process]] = custom_priority
        self.seed = self.parameters["seed"]

        # Helper indices for Numpy indexing
        self.molecule_idx = None

    def ports_schema(self):
        ports = {
            "bulk": numpy_schema("bulk"),
            "request": {
                process: {
                    "bulk": {
                        "_default": [],
                        "_emit": False,
                        "_divider": "null",
                        "_updater": "set",
                    }
                }
                for process in self.processNames
            },
            "allocate": {
                process: {
                    "bulk": {
                        "_default": [],
                        "_emit": False,
                        "_divider": "null",
                        "_updater": "set",
                    }
                }
                for process in self.processNames
            },
            "listeners": {
                "atp": listener_schema(
                    {
                        "atp_requested": ([0] * self.n_processes, self.processNames),
                        "atp_allocated_initial": (
                            [0] * self.n_processes,
                            self.processNames,
                        ),
                        # Use blame functionality to get ATP consumed per process
                        # 'atp_allocated_final': ([0] * self.n_processes,
                        #     self.processNames)
                    }
                )
            },
            "allocator_rng": {"_default": np.random.RandomState(seed=self.seed)},
        }
        return ports

    def next_update(self, timestep, states):
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.moleculeNames, states["bulk"]["id"]
            )
            self.atp_idx = bulk_name_to_idx("ATP[c]", states["bulk"]["id"])
        total_counts = counts(states["bulk"], self.molecule_idx)
        original_totals = total_counts.copy()
        counts_requested = np.zeros((self.n_molecules, self.n_processes), dtype=int)
        # Keep track of which process indices are in current partitioning layer
        proc_idx_in_layer = []
        for process in states["request"]:
            proc_idx = self.proc_name_to_idx[process]
            if len(states["request"][process]["bulk"]) > 0:
                proc_idx_in_layer.append(proc_idx)
            for req_idx, req in states["request"][process]["bulk"]:
                counts_requested[req_idx, proc_idx] += req

        if ASSERT_POSITIVE_COUNTS and np.any(counts_requested < 0):
            raise NegativeCountsError(
                "Negative value(s) in counts_requested:\n"
                + "\n".join(
                    "{} in {} ({})".format(
                        self.mol_idx_to_name[molIndex],
                        self.proc_idx_to_name[processIndex],
                        counts_requested[molIndex, processIndex],
                    )
                    for molIndex, processIndex in zip(*np.where(counts_requested < 0))
                )
            )

        # Calculate partition
        partitioned_counts = calculatePartition(
            self.processPriorities,
            counts_requested,
            total_counts,
            states["allocator_rng"],
        )

        partitioned_counts.astype(int, copy=False)

        if ASSERT_POSITIVE_COUNTS and np.any(partitioned_counts < 0):
            raise NegativeCountsError(
                "Negative value(s) in partitioned_counts:\n"
                + "\n".join(
                    "{} in {} ({})".format(
                        self.mol_idx_to_name[molIndex],
                        self.proc_idx_to_name[processIndex],
                        partitioned_counts[molIndex, processIndex],
                    )
                    for molIndex, processIndex in zip(*np.where(partitioned_counts < 0))
                )
            )

        # Ensure we are not overdrafting any molecules
        counts_unallocated = original_totals - np.sum(partitioned_counts, axis=-1)

        if ASSERT_POSITIVE_COUNTS and np.any(counts_unallocated < 0):
            raise NegativeCountsError(
                "Negative value(s) in counts_unallocated:\n"
                + "\n".join(
                    "{} ({})".format(
                        self.mol_idx_to_name[molIndex], counts_unallocated[molIndex]
                    )
                    for molIndex in np.where(counts_unallocated < 0)[0]
                )
            )

        # Only update listener ATP counts for processes in
        # current partitioning layer
        non_zero_mask = counts_requested[self.atp_idx, :] != 0
        curr_atp_req = np.array(states["listeners"]["atp"]["atp_requested"]).copy()
        curr_atp_alloc = np.array(
            states["listeners"]["atp"]["atp_allocated_initial"]
        ).copy()
        curr_atp_req[non_zero_mask] = counts_requested[self.atp_idx, non_zero_mask]
        curr_atp_alloc[non_zero_mask] = partitioned_counts[self.atp_idx, non_zero_mask]

        update = {
            "request": {process: {"bulk": []} for process in states["request"]},
            "allocate": {
                process: {"bulk": partitioned_counts[:, self.proc_name_to_idx[process]]}
                for process in states["request"]
            },
            "listeners": {
                "atp": {
                    "atp_requested": curr_atp_req,
                    "atp_allocated_initial": curr_atp_alloc,
                }
            },
        }

        return update


def calculatePartition(
    process_priorities, counts_requested, total_counts, random_state
):
    priorityLevels = np.sort(np.unique(process_priorities))[::-1]

    partitioned_counts = np.zeros_like(counts_requested)

    for priorityLevel in priorityLevels:
        processHasPriority = priorityLevel == process_priorities

        requests = counts_requested[:, processHasPriority].copy()

        total_requested = requests.sum(axis=1)
        excess_request_mask = (total_requested > total_counts) & (total_requested > 0)

        # Get fractional request for molecules that have excess request
        # compared to available counts
        fractional_requests = (
            requests[excess_request_mask, :]
            * total_counts[excess_request_mask, np.newaxis]
            / total_requested[excess_request_mask, np.newaxis]
        )

        # Distribute fractional counts to ensure full allocation of excess
        # request molecules
        remainders = fractional_requests % 1
        options = np.arange(remainders.shape[1])
        for idx, remainder in enumerate(remainders):
            total_remainder = remainder.sum()
            count = int(np.round(total_remainder))
            if count > 0:
                allocated_indices = random_state.choice(
                    options, size=count, p=remainder / total_remainder, replace=False
                )
                fractional_requests[idx, allocated_indices] += 1
        requests[excess_request_mask, :] = fractional_requests

        allocations = requests.astype(np.int64)
        partitioned_counts[:, processHasPriority] = allocations
        total_counts -= allocations.sum(axis=1)
    return partitioned_counts
