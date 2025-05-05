from typing import Optional

from vivarium.core.process import Process
from vivarium.core.engine import Engine, pp
from basico import *
from vivarium.core.types import State

from ecoli.processes.vkecoli.utils.basico_helper import _set_initial_concentrations, _get_transient_concentration
import matplotlib.pyplot as plt
from ecoli.processes.vkecoli.utils.updater import bulk_numpy_updater, get_bulk_counts, divide_bulk
import os


from ecoli.processes.registries import topology_registry

# Register default topology for this process, associating it with process name
NAME = "Kecoli"
# TOPOLOGY = {"bulk": ("bulk",), "listeners": ("listeners",), "timestep": ("timestep",)}

TOPOLOGY = {'species': ('species_store',)},


topology_registry.register(NAME, TOPOLOGY)



DEFAULT_MODEL_FILE = os.path.join('models','k-ecoli74.xml')


# defining a numpy array with custom data type to emulate vEcoli bulk structure
custom_dtype = np.dtype([
    ('id', '<U100'),  # String (Unicode, max 100 characters)
    ('count', '<f8'),  # Float (64-bit)
    ('rRNA_submass', '<f8'),  # Float (64-bit)
    ('tRNA_submass', '<f8'),
    ('mRNA_submass', '<f8'),
    ('miscRNA_submass', '<f8'),
    ('nonspecific_RNA_submass', '<f8'),
    ('protein_submass', '<f8'),
    ('metabolite_submass', '<f8'),
    ('water_submass', '<f8'),
    ('DNA_submass', '<f8')
])

# defining a vivarium process for ecoli kinetic metabolism models

class KecoliCell(Process):
    """
    note: initial env state is by convention

    """
    defaults = {
        'model_file': DEFAULT_MODEL_FILE, # path to the sbml model file
        'time_step': 1.0,
        'env_perturb': ["GLCx"], #TODO: dict, assert env species
        'env_conc': [1.0],
    }

    def __init__(self, parameters=None): #constructor
        super().__init__(parameters)

        self.copasi_model_object = load_model(self.parameters['model_file'])
        self.all_species = get_species(model=self.copasi_model_object).index.tolist()
        self.ic_default = get_species(model=self.copasi_model_object)["initial_concentration"].values
        for sp_idx,sp_name in enumerate(self.parameters['env_perturb']):
            self.ic_default[self.all_species.index(sp_name)] = self.parameters['env_conc'][sp_idx]

    def initial_state(self, config=None):

        num_species = len(self.all_species)

        # Create an empty array with default values of 0
        species_array = np.zeros(num_species, dtype=custom_dtype)

        # Fill in the 'id' and 'count' fields
        species_array['id'] = self.all_species  # retrieves species names from sbml
        species_array['count'] = self.ic_default  # retrieves ic from sbml

        return {'species':species_array }


    def ports_schema(self):

        ports = {
            'species': {
                '_default':[],
                '_updater': bulk_numpy_updater, # modified version of vEcoli bulk updater
                '_emit': True,
                "_divider": divide_bulk # modified version of vEcoli bulk divider
            }
        }

        return ports

    def next_update(self, endtime, states):

        species_levels = list(zip(states['species']['id'],states['species']['count'])) # retrieves current species levels

        _set_initial_concentrations(species_levels,self.copasi_model_object) # set species levels as ic

        # run time step simulation
        timecourse = run_time_course(duration=endtime, intervals=1, update_model=True, model=self.copasi_model_object)

        results = [(mol_id, _get_transient_concentration(name=mol_id, dm=self.copasi_model_object)) for mol_id in self.all_species] # reorganize results
        del_value = [] #TODO: rename to delta
        species_levels_values = states['species']['count']

        for idx,(mol_id,value_new) in enumerate(results):
            value = species_levels_values[idx]
            del_value.append((idx,value_new - value)) # get species level changes

        return {'species':del_value}

#%
def test_vkecoli():

    wd = os.getcwd()
    model_path = DEFAULT_MODEL_FILE

    total_time = 300

    config = {
        'model_file': model_path
    }

    kecoli_process = KecoliCell(parameters=config)
    kecoli_ports = kecoli_process.ports_schema()
    kecoli_initial_state = kecoli_process.initial_state()
    kecoli_initial_state['species_store'] = kecoli_initial_state.pop('species')

    sim = Engine(
        processes={'kecoli': kecoli_process},
        topology={'kecoli': {
            'species': ('species_store',)
        }},
        initial_state=kecoli_initial_state,
    )

    sim.update(total_time) # run process

    data = sim.emitter.get_timeseries() # retrive simulation outputs from in memory emitter



#
if __name__ == '__main__':
    test_vkecoli()

