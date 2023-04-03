from vivarium.core.process import Step
from ecoli.library.schema import numpy_schema

class CacheUpdate(Step):
    """Placed before and after every Step to ensure that _cached_entryState
    is up to date (as if each Step was running in its own timestep)"""

    name = 'cache-update'

    def __init__(self, parameters=None):
        super().__init__(parameters)
        # Topology for all unique molecule ports (port: path)
        self.unique_dict = self.parameters['unique_dict']

    def ports_schema(self):
        return {
            unique_mol: numpy_schema(unique_mol)
            for unique_mol in self.unique_dict
        }
    
    def next_update(self, timestep, states):
        return {
            unique_mol: {'time': states[unique_mol]['time'][0] - 1}
            for unique_mol in self.unique_dict.keys()
        }
