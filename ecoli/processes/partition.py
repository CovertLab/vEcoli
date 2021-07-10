"""
Partition

Allocate molecules to each process by proportion requested and process priority
"""
from vivarium.core.process import Deriver

import copy

class Partition(Deriver):
    name = 'partition'

    defaults = {}

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.process_priorities = {
            10: ['protein_degradation'],
            0: ['transcript_initiation',
                'transcript_elongation',
                'rna_degradation',
                'polypeptide_initiation',
                'polypeptide_elongation',
                'complexation',
                'equilibrium',
                'chromosome_replication',
                'mass',
                'divide_condition',
                'partition'],
            -5: ['two_component_system'],
            -10:['tf_binding',
                'metabolism'],
        }

    def ports_schema(self):
        return {}
    
    def calculate_request(self, timestep, states):
        return {'timesteps': 1}

    def next_update(self, timestep, states):
        # Find intersection between two lists
        def intersection(lst1, lst2):
            temp = set(lst2)
            lst3 = [value for value in lst1 if value in temp]
            return lst3
        
        # Deepcopy is bad
        curr_state = copy.deepcopy(states)
        update = {'allocated': {}}
        for processes in self.process_priorities.values():
            for process in processes:
                update['allocated'][process] = {}
            # Get only processes that have requested molecules
            requesting_procs = intersection(processes, list(curr_state['requested'].keys()))
            total_requested = {name: 0 for name in curr_state['totals'].keys()}
            # Find total count requested for each molecule
            for process in requesting_procs:
                for (molecule, count) in curr_state['requested'][process].items():
                    total_requested[molecule] += count
            for process in requesting_procs:
                for (molecule, requested_count) in curr_state['requested'][process].items():
                    # Divy molecules by proportion requested if demand exceeds supply
                    if total_requested[molecule] > curr_state['totals'][molecule]:
                        update['allocated'][process][molecule] = requested_count \
                            / total_requested[molecule] * curr_state['totals'][molecule]
                    else:
                        update['allocated'][process][molecule] = requested_count
                    # Decrease molecules left for lower priority processes
                    curr_state['totals'][molecule] -= update['allocated'][process][molecule]
        update['timesteps'] = 1
        return update
