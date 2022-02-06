from matplotlib import pyplot as plt
import numpy as np
from vivarium.core.composition import simulate_process

from vivarium.core.process import Process
from vivarium.plots.simulation_output import plot_variables

from ecoli.processes.registries import topology_registry


# Register default topology for this process, associating it with process name
NAME = 'ecoli-tf-binding'
TOPOLOGY = {
        
}
topology_registry.register(NAME, TOPOLOGY)


class CellWall(Process):

    name = NAME
    topology = TOPOLOGY
    defaults = {
        'cracked' : False,
        'murein' : 'CPD-12261[p]',
        'beta_lactam' : "", # TBD
        'SLT' : '', # TBD
        'PBP1A' : 'CPLX0-7717[i]', # transglycosylase-transpeptidase ~100
        'PBP1B' : 'CPLX0-3951[i]', # transglycosylase-transpeptidase ~100
        'PBP1C' : 'G7322-MONOMER[i]', # transglycosylase
        'PBP2' : 'EG10606-MONOMER[i]', # transpeptidase ~20
        'PBP3' : 'EG10341-MONOMER[i]', # transglycosylase-transpeptidase ~50
        'PBP4' : 'EG10202-MONOMER[p]', # DD-endopeptidase, DD-carboxypeptidase ~110
        'PBP5' : 'EG10201-MONOMER[i]', # DD-caroxypeptidase ~1,800
        'PBP6' : 'EG10203-MONOMER[i]', # DD-carbocypeptidase ~600
        'PBP7' : 'EG12015-MONOMER[p]', # DD-endopeptidase
        'SURFACE_AREA_PER_MOLECULE' : {		# in um^2 / molecule
            'LPS': 1.42E-6,
            'porins_and_ompA': 9E-6,
            'phospholipids': 4.71E-07,
            'lipoprotein': 7.14E-07,
        }
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)


    def ports_schema(self):
        schema = {

        }

        return schema


    def next_update(self, timestep, states):
        # Cell wall construction/destruction, beta-lactam activity

        # Crack detection
        
        # Cracking is irreversible
        cracked = self.cracked or False

        update = {
            "cracked" : cracked
        }
        return update


def main():
    params = {

    }

    cellwall = CellWall(params)
    data = simulate_process(cellwall, {'total_time': 10})
    # fig = plot_variables(
    #     data,
    #     variables=[
    #         # ('internal', 'antibiotic'),
    #         # ('internal', 'antibiotic_hydrolyzed'),
    #     ],
    # )


if __name__ == '__main__':
    main()
