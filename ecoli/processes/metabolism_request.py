"""
Metabolism

Metabolism sub-model. Encodes molecular simulation of microbial metabolism using flux-balance analysis.

TODO:
- option to call a reduced form of metabolism (assume optimal)
- handle oneSidedReaction constraints
"""

from os import symlink
from vivarium.core.process import Deriver
from ecoli.library.schema import bulk_schema

from ecoli.processes.metabolism_evolve import FluxBalanceAnalysisModel

USE_KINETICS = True

class MetabolismRequest(Deriver):
    name = 'ecoli-metabolism-request'

    defaults = {}

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.include_ppgpp = self.parameters['include_ppgpp']
        self.current_timeline = self.parameters['current_timeline']
        self.media_id = self.parameters['media_id']

        # Create model to use to solve metabolism updates
        self.model = FluxBalanceAnalysisModel(
            self.parameters,
            timeline=self.current_timeline,
            include_ppgpp=self.include_ppgpp)


    def ports_schema(self):
        return {
            'metabolites': bulk_schema(self.model.metaboliteNamesFromNutrients),
            'catalysts': bulk_schema(self.model.catalyst_ids),
            'kinetics_enzymes': bulk_schema(self.model.kinetic_constraint_enzymes),
            'kinetics_substrates': bulk_schema(self.model.kinetic_constraint_substrates),
        }
    
    def calculate_request(self, timestep, states):
        # Request counts of molecules needed
        requests = {}
        metabolites = {metabolite: count for metabolite, count 
                       in states['metabolites'].items()}
        catalysts = {catalyst: count for catalyst, count 
                     in states['catalysts'].items()}
        enzymes = {enzyme: count for enzyme, count 
                   in states['kinetics_enzymes'].items()}
        substrates = {substrate: count for substrate, count 
                      in states['kinetics_substrates'].items()}
        requests['requested'] = {**metabolites, **catalysts, **enzymes, **substrates}
        requests['requested'].update(states['catalysts'])
        requests['requested'].update(states['kinetics_enzymes'])
        requests['requested'].update(states['kinetics_substrates'])
        return requests
        
    def next_update(self, timestep, states):
        return {}  