'''
Kinetic rate law generation using the Convenience Kinetics formulation of Michaelis-Menten kinetics

Formulation provided in:
    Liebermeister, Wolfram, and Edda Klipp. "Bringing metabolic networks to life:
    convenience rate law and thermodynamic constraints."
    Theoretical Biology and Medical Modelling 3.1 (2006): 41.

# TODO -- make a vmax options if enzyme kcats not available
'''

import os

import numpy as np

from vivarium.library.dict_utils import tuplify_port_dicts



def get_molecules(reactions):
    '''
    Get a list of all molecules used by reactions

    Args:
           reaction (dict): all reactions that will be used by transport
    Returns:
           self.molecule_ids (list): all molecules used by these reactions
    '''

    molecule_ids = []
    for reaction_id, specs in reactions.items():
        stoichiometry = specs['stoichiometry']
        substrates = stoichiometry.keys()
        enzymes = specs['catalyzed by']
        # Add all relevant molecules_ids
        molecule_ids.extend(substrates)
        molecule_ids.extend(enzymes)
    return list(set(molecule_ids))

# Helper functions
def make_configuration(reactions):
    '''
    Make the rate law configuration, which tells the parameters where to be placed.

    Args:
        reactions (dict): all reactions that will be made into rate laws, in the same format as all_reactions (above).

    Returns:
        rate_law_configuration (dict): includes partition and reaction_cofactor entries for each reaction
    '''

    rate_law_configuration = {}
    # gets all potential interactions between the reactions
    for reaction_id, specs in reactions.items():
        enzymes = specs['catalyzed by']
        # initialize all enzymes
        for enzyme in enzymes:
            if enzyme not in rate_law_configuration:
                rate_law_configuration[enzyme] = {
                    'partition': [],
                    'reaction_cofactors': {},
                }

    # identify parameters for reactions
    for reaction_id, specs in reactions.items():
        stoich = specs.get('stoichiometry')
        enzymes = specs.get('catalyzed by', None)
        reversibility = specs.get('is reversible', False)

        # get sets of cofactors driving this reaction
        forward_cofactors = [mol for mol, coeff in stoich.items() if coeff < 0]
        cofactors = [forward_cofactors]

        if reversibility:
            reverse_cofactors = [mol for mol, coeff in stoich.items() if coeff > 0]
            cofactors.append(reverse_cofactors)

        # get partition, reactions, and parameter indices for each enzyme, and save to rate_law_configuration dictionary
        for enzyme in enzymes:

            # get competition for this enzyme from all other reactions
            competing_reactions = [rxn for rxn, specs2 in reactions.items() if
                    (rxn is not reaction_id) and (enzyme in specs2['catalyzed by'])]

            competitors = []
            for reaction2 in competing_reactions:
                stoich2 = reactions[reaction2]['stoichiometry']
                reactants2 = [mol for mol, coeff in stoich2.items() if coeff < 0]
                competitors.append(reactants2)

            # partition includes both competitors and cofactors.
            partition = competitors + cofactors
            rate_law_configuration[enzyme]['partition'] = partition
            rate_law_configuration[enzyme]['reaction_cofactors'][reaction_id] = cofactors

    return rate_law_configuration

def cofactor_numerator(concentration, km):
    return concentration / km if km else 0

def cofactor_denominator(concentration, km):
    return 1 + concentration / km if km else 1

def construct_convenience_rate_law(stoichiometry, enzyme, cofactors_sets, partition, parameters):
    '''
    Make a convenience kinetics rate law for one enzyme

    Args:
        stoichiometry (dict): the stoichiometry for the given reaction
        enzyme (str): the current enzyme
        cofactors_sets: a list of lists with the required cofactors, grouped by [[cofactor set 1], [cofactor set 2]], each pair needs a kcat.
        partition: a list of lists. each sublist is the set of cofactors for a given partition.
            [[C1, C2],[C3, C4], [C5]]
        parameters (dict): all the parameters with {parameter_id: value}

    Returns:
        a kinetic rate law for the reaction, with arguments for concentrations and parameters,
        and returns flux.
    '''

    kcat_f = parameters.get('kcat_f')
    kcat_r = parameters.get('kcat_r')

    # remove km parameters with None as their value
    for parameter, value in parameters.items():
        if 'kcat' not in parameter:
            if value is None:
                for part in partition:
                    if parameter in part:
                        part.remove(parameter)
                for cofactors_set in cofactors_sets:
                    if parameter in cofactors_set:
                        cofactors_set.remove(parameter)
                # print('removing parameter: {}'.format(parameter))

	# if reversible, determine direction by looking at stoichiometry
    if kcat_r:
        coeff = [stoichiometry[mol] for mol in cofactors]
        positive_coeff = [c > 0 for c in coeff]
        if all(c == True for c in positive_coeff):  # if all coeffs are positive
            kcat = -kcat_r  # use reverse rate
        elif all(c == False for c in positive_coeff):  # if all coeffs are negative
            kcat = kcat_f
    else:
        kcat = kcat_f

    def rate_law(concentrations):

        # construct numerator
        enzyme_concentration = concentrations[enzyme]

        numerator = 0
        for cofactors in cofactors_sets:
            # multiply the affinities of all cofactors
            term = np.prod([
                cofactor_numerator(
                    concentrations[molecule],
                    parameters[molecule])  # km of molecule
                for molecule in cofactors])
            numerator += kcat * term  # TODO (if there is no kcat, need an exception)
        numerator *= enzyme_concentration

        # construct denominator, with all competing terms in the partition
        # denominator starts at +1 for the unbound state
        denominator = 1
        for cofactors_set in partition:
            # multiply the affinities of all cofactors in this partition
            term = np.prod([
                cofactor_denominator(
                    concentrations[molecule],
                    parameters[molecule])
				for molecule in cofactors_set])
            denominator += term - 1
        flux = numerator / denominator

        return flux

    return rate_law

# Make rate laws
def make_rate_laws(reactions, rate_law_configuration, kinetic_parameters):
    '''
    Make a rate law for each reaction

    Args:
        reactions (dict): in the same format as all_reactions, described above

        rate_law_configuration (dict): with an embedded structure:
            {enzyme_id: {
                'reaction_cofactors': {
                    reaction_id: [cofactors list]
                    }
                'partition': [partition list]
                }
            }

        kinetic_parameters (dict): with an embedded structure:
            {reaction_id: {
                'enzyme_id': {
                    parameter_id: value
                    }
                }
            }

    Returns:
        rate_laws (dict): each reaction_id is a key and has sub-dictionary for each relevant enzyme,
            with kinetic rate law functions as their values
    '''

    rate_laws = {reaction_id: {} for reaction_id in list(reactions.keys())}
    for reaction_id, specs in reactions.items():
        stoichiometry = specs.get('stoichiometry')
        # reversible = specs.get('is reversible') # TODO (eran) -- add reversibility based on specs
        enzymes = specs.get('catalyzed by')

        # rate law for each enzyme
        for enzyme in enzymes:
            if enzyme not in kinetic_parameters[reaction_id]:
                print('{} not in reaction {}'.format(enzyme, reaction_id))
                continue

            cofactors_sets = rate_law_configuration[enzyme]["reaction_cofactors"][reaction_id]
            partition = rate_law_configuration[enzyme]["partition"]

            rate_law = construct_convenience_rate_law(
                stoichiometry,
                enzyme,
                cofactors_sets,
                partition,
                kinetic_parameters[reaction_id][enzyme])

            # save the rate law for each enzyme in this reaction
            rate_laws[reaction_id][enzyme] = rate_law

    return rate_laws


class KineticFluxModel(object):
    '''
    A kinetic rate law class

    Args:
        all_reactions (dict): all metabolic reactions, with:
            {reaction_id: {
                'catalyzed by': list,
                'is reversible': bool,
                'stoichiometry': dict,
                }}

        kinetic_parameters (dict): a dictionary of parameters a nested format:
            {reaction_id: {
                enzyme_id : {
                    param_id: param_value}}}

    Attributes:
        rate_laws: a dict, with a key for each reaction id, and then subdictionaries with each reaction's enzymes
            and their rate law function. These rate laws are used directly from within this dictionary
    '''

    def __init__(self, all_reactions, kinetic_parameters):

        self.kinetic_parameters = kinetic_parameters
        self.reaction_ids = list(self.kinetic_parameters.keys())
        self.reactions = {reaction_id: all_reactions[reaction_id] for reaction_id in all_reactions}
        self.molecule_ids = get_molecules(self.reactions)

        # make the rate laws
        self.rate_law_configuration = make_configuration(self.reactions)

        self.rate_laws = make_rate_laws(
            self.reactions,
            self.rate_law_configuration,
            self.kinetic_parameters)

    def get_fluxes(self, concentrations_dict):
        '''
        Use rate law functions to calculate flux

        Args:
            concentrations_dict (dict): all relevant molecules and their concentrations, in mmol/L.
                {molecule_id: concentration}

        Returns:
            reaction_fluxes (dict) - with fluxes for all reactions
        '''

        # Initialize reaction_fluxes and exchange_fluxes dictionaries
        reaction_fluxes = {reaction_id: 0.0 for reaction_id in self.reaction_ids}

        for reaction_id, enzymes in self.rate_laws.items():
            for enzyme, rate_law in enzymes.items():
                flux = rate_law(concentrations_dict)
                reaction_fluxes[reaction_id] += flux

        return reaction_fluxes



toy_reactions = {
    'ABC-13-RXN': {
        'stoichiometry': {
            ('cytoplasm', 'PI'): 1,
            ('cytoplasm', 'ADP'): 1,
            ('cytoplasm', 'GLT'): 1,
            ('cytoplasm', 'PROTON'): 1,
            ('cytoplasm', 'ATP'): -1,
            ('cytoplasm', 'WATER'): -1,
            ('periplasm', 'GLT'): -1},
        'is reversible': False,
        'catalyzed by': [
            ('membrane', 'ABC-13-CPLX')]},
    'TRANS-RXN-122': {
        'stoichiometry': {
            ('cytoplasm', 'GLT'): 1, 
            ('periplasm', 'GLT'): -1, 
            ('periplasm', 'NA+'): -2, 
            ('cytoplasm', 'NA+'): 2},
        'is reversible': False,
        'catalyzed by': [
            ('membrane', 'GLTP-MONOMER'),
            ('membrane', 'DCTA-MONOMER')]},
    }


toy_kinetics = {
    'ABC-13-RXN': {
        ('membrane', 'ABC-13-CPLX'): {
            ('cytoplasm', 'ATP'): None,
            ('periplasm', 'GLT'): 1e-3,
            ('cytoplasm', 'WATER'): None,
            'kcat_f': 1.0
        }},
    'TRANS-RXN-122': {
        ('membrane', 'DCTA-MONOMER'): {
            ('periplasm', 'GLT'): 1e-3,
            ('periplasm', 'NA+'): 1e-5,
            'kcat_f': 1.0
        },
        ('membrane', 'GLTP-MONOMER'): {
            ('periplasm', 'GLT'): 1e-3,
            ('periplasm', 'NA+'): 1e-5,
            ('periplasm', 'PROTON'): None,
            'kcat_f': 1.0
        }}
    }

toy_initial_state = {
    'cytoplasm': {
        'PI': 1.0,
        'ADP': 1.0,
        'GLT': 1.0,
        'PROTON': 1.0,
        'ATP': 1.0,
        'WATER': 1.0,
        'NA+': 1.0},
    'periplasm': {
        'GLT': 1.0,
        'NA+': 1.0
    },
    'membrane': {
        'ABC-13-CPLX': 1.0,
        'GLTP-MONOMER': 1.0,
        'DCTA-MONOMER': 1.0}
}


def test_kinetics():
    kinetic_rate_laws = KineticFluxModel(toy_reactions, toy_kinetics)
    flattened_toy_states = tuplify_port_dicts(toy_initial_state)
    flux = kinetic_rate_laws.get_fluxes(flattened_toy_states)

    print(flux)


if __name__ == '__main__':
    test_kinetics()
