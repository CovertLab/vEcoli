from vivarium.core.process import Step
from ecoli.library.schema import numpy_schema

class UniqueUpdate(Step):
    """Placed after before and after all Steps to ensure that
    unique molecules are completely up-to-date"""

    name = 'unique-update'

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
            unique_mol: {'update': True}
            for unique_mol in self.unique_dict.keys()
        }
