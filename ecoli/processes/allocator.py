"""
Allcoator

Updates bulk with process updates, runs metabolism, runs process requests, allocates molecules

**Currently NON-FUNCTIONAL**
"""
from vivarium.core.process import Deriver
import numpy as np

from wholecell.utils.constants import CUSTOM_PRIORITIES

ASSERT_POSITIVE_COUNTS = False

class NegativeCountsError(Exception):
	pass

class Allocator(Deriver):
    name = 'allocator'

    defaults = {}
    
    processes = {}

    # Constructor
    def __init__(self, parameters=None):
        super().__init__()
        self.moleculeNames = parameters['molecule_names']
        self.n_molecules = len(self.moleculeNames)
        self.mol_name_to_idx = {name: idx for idx, name in enumerate(self.moleculeNames)}
        self.mol_idx_to_name = {idx: name for idx, name in enumerate(self.moleculeNames)}
        self.processNames = list(parameters['processes'].keys())
        self.n_processes = len(self.processNames)
        self.proc_name_to_idx = {name: idx for idx, name in enumerate(self.processNames)}
        self.proc_idx_to_name = {idx: name for idx, name in enumerate(self.processNames)}
        self.processPriorities = np.zeros(len(self.processNames))
        for process, custom_priority in CUSTOM_PRIORITIES.items():
            self.processPriorities[self.proc_name_to_idx[process]] = custom_priority
        self.seed = parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)
        self.processes = parameters['processes']

    def ports_schema(self):
        ports = {}
        ports['bulk'] = {molecule: {'_default': 0} 
                         for molecule in self.moleculeNames}
        return ports

    def next_update(self, timestep, states):
        total_counts = np.array([states['totals'][molecule] for 
                                 molecule in self.mol_idx_to_name.values()])
        original_totals = total_counts.copy()
        counts_requested = np.zeros((self.n_molecules, self.n_processes))
        for process in self.processes:
            proc_idx = self.proc_name_to_idx[process]
            for molecule, molecule_requests in self.processes.requests:
                mol_idx = self.mol_name_to_idx[molecule]
                counts_requested[mol_idx][proc_idx] = molecule_requests

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

        if ASSERT_POSITIVE_COUNTS and np.any(counts_unallocated < 0):
            raise NegativeCountsError(
                    "Negative value(s) in counts_unallocated:\n"
                    + "\n".join(
                    "{} ({})".format(
                        self.mol_idx_to_name[molIndex],
                        counts_unallocated[molIndex]
                        )
                    for molIndex in np.where(counts_unallocated < 0)[0]
                    )
                )
        
        update = {
            #'requested': {process: {} for process in states['requested']},
            'allocated': {
                process: {
                    molecule: partitioned_counts[self.mol_name_to_idx[molecule], 
                                                    self.proc_name_to_idx[process]] 
                    for molecule in states['requested'][process]}
                for process in states['requested']}}
        
        return update

def calculatePartition(process_priorities, counts_requested, total_counts, random_state):
    priorityLevels = np.sort(np.unique(process_priorities))[::-1]
    
    partitioned_counts = np.zeros_like(counts_requested)

    for priorityLevel in priorityLevels:
        processHasPriority = (priorityLevel == process_priorities)

        requests = counts_requested[:, processHasPriority].copy()

        total_requested = requests.sum(axis=1)
        excess_request_mask = (total_requested > total_counts)

        # Get fractional request for molecules that have excess request
        # compared to available counts
        fractional_requests = (
            requests[excess_request_mask, :] * total_counts[excess_request_mask, np.newaxis]
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
                allocated_indices = random_state.choice(options, size=count, p=remainder/total_remainder, replace=False)
                fractional_requests[idx, allocated_indices] += 1
        requests[excess_request_mask, :] = fractional_requests

        allocations = requests.astype(np.int64)
        partitioned_counts[:, processHasPriority] = allocations
        total_counts -= allocations.sum(axis=1)
    return partitioned_counts
