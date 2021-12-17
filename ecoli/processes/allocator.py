"""
=========
Allocator
=========

Reads requests from PartionedProcesses, and allocates molecules according to
process priorities.
"""
import numpy as np
from vivarium.core.process import Deriver

from ecoli.processes.registries import topology_registry


# Register default topology for this process, associating it with process name
NAME = 'allocator'
TOPOLOGY = {
    "molecules": ("bulk",),
    "listeners": ("listeners",)
}
topology_registry.register(NAME, TOPOLOGY)

ASSERT_POSITIVE_COUNTS = True

class NegativeCountsError(Exception):
	pass

class Allocator(Deriver):
    """ Allocator Deriver """
    name = NAME
    topology = TOPOLOGY

    defaults = {}

    processes = {}

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.moleculeNames = self.parameters['molecule_names']
        self.n_molecules = len(self.moleculeNames)
        self.mol_name_to_idx = {name: idx for idx, name in enumerate(self.moleculeNames)}
        self.mol_idx_to_name = {idx: name for idx, name in enumerate(self.moleculeNames)}
        self.processNames = self.parameters['process_names']
        self.n_processes = len(self.processNames)
        self.proc_name_to_idx = {name: idx for idx, name in enumerate(self.processNames)}
        self.proc_idx_to_name = {idx: name for idx, name in enumerate(self.processNames)}
        self.processPriorities = np.zeros(len(self.processNames))
        for process, custom_priority in self.parameters['custom_priorities'].items():
            if process not in self.proc_name_to_idx.keys():
                continue
            self.processPriorities[self.proc_name_to_idx[process]] = custom_priority
        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)

    def ports_schema(self):
        ports = {
            'bulk': {
                molecule: {'_default': 0}
                for molecule in self.moleculeNames},
            'request': {
                process: {
                    'bulk': {
                        '*': {'_default': 0, '_updater': 'set'}}}
                for process in self.processNames},
            'allocate': {
                process: {
                    'bulk': {
                        '*': {'_default': 0, '_updater': 'set'}}}
                for process in self.processNames},
        }
        return ports

    def next_update(self, timestep, states):
        total_counts = np.array([states['bulk'][molecule] for
                                 molecule in self.mol_idx_to_name.values()])
        original_totals = total_counts.copy()
        counts_requested = np.zeros((self.n_molecules, self.n_processes), dtype=int)
        for process in states['request']:
            proc_idx = self.proc_name_to_idx[process]
            for molecule, count in states['request'][process]['bulk'].items():
                mol_idx = self.mol_name_to_idx[molecule]
                counts_requested[mol_idx][proc_idx] += count

        if ASSERT_POSITIVE_COUNTS and np.any(counts_requested < 0):
            raise NegativeCountsError(
                "Negative value(s) in counts_requested:\n"
                + "\n".join(
                    "{} in {} ({})".format(
                        self.mol_idx_to_name[molIndex],
                        self.proc_idx_to_name[processIndex],
                        counts_requested[molIndex, processIndex]
                        )
                    for molIndex, processIndex in zip(*np.where(counts_requested < 0))
                    )
                )

        # Calculate partition
        partitioned_counts = calculatePartition(
            self.processPriorities,
            counts_requested,
            total_counts,
            self.random_state
            )

        partitioned_counts.astype(int, copy=False)

        if ASSERT_POSITIVE_COUNTS and np.any(partitioned_counts < 0):
            raise NegativeCountsError(
                    "Negative value(s) in partitioned_counts:\n"
                    + "\n".join(
                    "{} in {} ({})".format(
                        self.mol_idx_to_name[molIndex],
                        self.proc_idx_to_name[processIndex],
                        counts_requested[molIndex, processIndex]
                        )
                    for molIndex, processIndex in zip(*np.where(partitioned_counts < 0))
                    )
                )

        # Record unpartitioned counts for later merging
        counts_unallocated = original_totals - np.sum(
            partitioned_counts, axis=-1)

        # TODO (Cyrus) -- Reintroduce this later (or ignore it until allocation is removed)
        # if ASSERT_POSITIVE_COUNTS and np.any(counts_unallocated < 0):
        #     raise NegativeCountsError(
        #             "Negative value(s) in counts_unallocated:\n"
        #             + "\n".join(
        #             "{} ({})".format(
        #                 self.mol_idx_to_name[molIndex],
        #                 counts_unallocated[molIndex]
        #                 )
        #             for molIndex in np.where(counts_unallocated < 0)[0]
        #             )
        #         )


        update = {
            'request': {
                process: {
                    'bulk': {
                        molecule: 0
                        for molecule in states['request'][process]['bulk']}}
                for process in states['request']},
            'allocate': {
                process: {
                    'bulk': {molecule: partitioned_counts[
                        self.mol_name_to_idx[molecule],
                        self.proc_name_to_idx[process]]
                    for molecule in states['request'][process]['bulk']}}
                for process in states['request']}}

        return update

def calculatePartition(process_priorities, counts_requested, total_counts, random_state):
    priorityLevels = np.sort(np.unique(process_priorities))[::-1]

    partitioned_counts = np.zeros_like(counts_requested)

    for priorityLevel in priorityLevels:
        processHasPriority = (priorityLevel == process_priorities)

        requests = counts_requested[:, processHasPriority].copy()

        total_requested = requests.sum(axis=1)
        excess_request_mask = (total_requested > total_counts) & (total_requested > 0)

        # Get fractional request for molecules that have excess request
        # compared to available counts
        fractional_requests = (
            requests[excess_request_mask, :] * total_counts[excess_request_mask, np.newaxis]
            / total_requested[excess_request_mask, np.newaxis]
            )

        # requests_counts_product = requests[excess_request_mask, :] * total_counts[excess_request_mask, np.newaxis]
        # requests_with_mask = total_requested[excess_request_mask, np.newaxis]
        # test = np.true_divide(requests_counts_product, requests_with_mask,
        #                  out=np.zeros_like(requests_counts_product), where=requests_with_mask != [0], casting='unsafe')
        # TODO(Matt): Incorporate this fix into the line of code above. Commented code a start, but only returns ints.
        for lst_index in range(len(fractional_requests)):
            for value_index in range(len(fractional_requests[lst_index])):
                if np.isnan(fractional_requests[lst_index][value_index]):
                    fractional_requests[lst_index][value_index] = 0

        # Distribute fractional counts to ensure full allocation of excess
        # request molecules
        remainders = fractional_requests % 1
        options = np.arange(remainders.shape[1])
        for idx, remainder in enumerate(remainders):
            total_remainder = remainder.sum()
            count = int(np.round(total_remainder))
            if count > 0:
                allocated_indices = random_state.choice(options, size=count, p=remainder/total_remainder, replace=False)
                fractional_requests[idx, allocated_indices] += 1
        requests[excess_request_mask, :] = fractional_requests

        allocations = requests.astype(np.int64)
        partitioned_counts[:, processHasPriority] = allocations
        total_counts -= allocations.sum(axis=1)
    return partitioned_counts
