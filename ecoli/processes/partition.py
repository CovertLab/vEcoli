"""
Partition

Allocate molecules to each process by proportion requested and process priority
"""
from vivarium.core.process import Deriver
from ecoli.library.schema import array_from
import numpy as np

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
        ports = {}
        ports['totals'] = {'*': {'_default': 0}}
        return ports
    
    def calculate_request(self, timestep, states):
        # Ensure that time is incremented at the end of every
        # half time step
        return {'timesteps': 1}

    def evolve_state(self, timestep, states):
        def intersection(lst1, lst2):
            temp = set(lst2)
            lst3 = [value for value in lst1 if value in temp]
            return lst3
        update = {'allocated': {}}
        running_totals = {}
        for process in states['requested'].values():
            for molecule in process:
                running_totals[molecule] = states['totals'][molecule]
        for processes in self.process_priorities.values():
            for process in processes:
                update['allocated'][process] = {}
            # Get only processes that have requested molecules
            requesting_procs = intersection(processes, list(states['requested']))
            total_requested = {name: 0 for name in states['totals'].keys()}
            # Find total count requested for each molecule
            for process in requesting_procs:
                for (molecule, count) in states['requested'][process].items():
                    total_requested[molecule] += int(count)
            for process in requesting_procs:
                for (molecule, requested_count) in states['requested'][process].items():
                    if running_totals[molecule] <= 0:
                        update['allocated'][process][molecule] = 0
                    else:
                        # Divy molecules by proportion requested if demand exceeds supply
                        if total_requested[molecule] > running_totals[molecule]:
                            update['allocated'][process][molecule] = np.floor(requested_count 
                                / total_requested[molecule] * running_totals[molecule])
                        else:
                            update['allocated'][process][molecule] = requested_count
                        # Decrease molecules left for lower priority processes
                        running_totals[molecule] -= update['allocated'][process][molecule]
        update['timesteps'] = 1
        return update
    
    def next_update(self, timestep, states):
        return self.evolve_state(timestep, states)
