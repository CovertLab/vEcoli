from six.moves import cPickle
import numpy as np

from vivarium.core.process import Generator
from vivarium.core.composition import simulate_compartment_in_experiment

from ecoli.processes.complexation import Complexation
from ecoli.processes.protein_degradation import ProteinDegradation
from ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from ecoli.processes.metabolism import Metabolism

from wholecell.utils import units
from wholecell.utils.fitting import normalize

RAND_MAX = 2**31

class Ecoli(Generator):

    defaults = {
        'seed': 0,
        'sim_data_path': '../wcEcoli/out/manual/kb/simData.cPickle'}

    def __init__(self, config):
        super(Ecoli, self).__init__(config)

        self.seed = np.uint32(self.config['seed'] % np.iinfo(np.uint32).max)
        self.random_state = np.random.RandomState(seed = self.seed)

    def initialize_complexation(self, sim_data):
        complexation_config = {
            'stoichiometry': sim_data.process.complexation.stoichMatrix().astype(np.int64).T,
            'rates': sim_data.process.complexation.rates,
            'molecule_names': sim_data.process.complexation.moleculeNames,
            'seed': self.random_state.randint(RAND_MAX)}

        complexation = Complexation(complexation_config)
        return complexation

    def initialize_protein_degradation(self, sim_data):
        protein_degradation_config = {
            'raw_degradation_rate': sim_data.process.translation.monomerData['degRate'].asNumber(1 / units.s),
            'shuffle_indexes': sim_data.process.translation.monomerDegRateShuffleIdxs if hasattr(
                sim_data.process.translation, "monomerDegRateShuffleIdxs") else None,
            'water_id': sim_data.moleculeIds.water,
            'amino_acid_ids': sim_data.moleculeGroups.amino_acids,
            'amino_acid_counts': sim_data.process.translation.monomerData["aaCounts"].asNumber(),
            'protein_ids': sim_data.process.translation.monomerData['id'],
            'protein_lengths': sim_data.process.translation.monomerData['length'].asNumber(),
            'seed': self.random_state.randint(RAND_MAX)}

        protein_degradation = ProteinDegradation(protein_degradation_config)
        return protein_degradation

    def initialize_polypeptide_initiation(self, sim_data):
        polypeptide_initiation_config = {
            'protein_lengths': sim_data.process.translation.monomerData["length"].asNumber(),
            'translation_efficiencies': normalize(sim_data.process.translation.translationEfficienciesByMonomer),
            'active_ribosome_fraction': sim_data.process.translation.ribosomeFractionActiveDict,
            'elongation_rates': sim_data.process.translation.ribosomeElongationRateDict,
            'variable_elongation': False,
            'make_elongation_rates': sim_data.process.translation.make_elongation_rates,
            'protein_index_to_TU_index': sim_data.relation.rnaIndexToMonomerMapping,
            'all_TU_ids': sim_data.process.transcription.rnaData['id'],
            'all_mRNA_ids': sim_data.process.translation.monomerData['rnaId'],
            'ribosome30S': sim_data.moleculeIds.s30_fullComplex,
            'ribosome50S': sim_data.moleculeIds.s50_fullComplex,
            'seed': self.random_state.randint(RAND_MAX),
            'shuffle_indexes': sim_data.process.translation.monomerDegRateShuffleIdxs if hasattr(
                sim_data.process.translation, "monomerDegRateShuffleIdxs") else None}

        polypeptide_initiation = PolypeptideInitiation(polypeptide_initiation_config)
        return polypeptide_initiation

    def initialize_metabolism(self, sim_data):
        metabolism_config = {
            'get_import_constraints': sim_data.external_state.get_import_constraints,
            'nutrientToDoublingTime': sim_data.nutrientToDoublingTime,
            'use_trna_charging': False,
            'include_ppgpp': False,
            'aa_names': sim_data.moleculeGroups.amino_acids,
            'current_timeline': None,
            'media_id': 'minimal',
            'condition': sim_data.condition,
            'nutrients': sim_data.conditions[sim_data.condition]['nutrients'],
            'metabolism': sim_data.process.metabolism,
            'non_growth_associated_maintenance': sim_data.constants.nonGrowthAssociatedMaintenance,
            'avogadro': sim_data.constants.nAvogadro,
            'cell_density': sim_data.constants.cellDensity,
            'dark_atp': sim_data.constants.darkATP,
            'cell_dry_mass_fraction': sim_data.mass.cellDryMassFraction,
            'get_biomass_as_concentrations': sim_data.mass.getBiomassAsConcentrations,
            'ppgpp_id': sim_data.moleculeIds.ppGpp,
            'getppGppConc': sim_data.growthRateParameters.getppGppConc,
            'exchange_data_from_media': sim_data.external_state.exchange_data_from_media,
            'getMass': sim_data.getter.getMass,
            'doubling_time': sim_data.conditionToDoublingTime[sim_data.condition],
            'amino_acid_ids': sorted(sim_data.amino_acid_code_to_id_ordered.values())}

        metabolism = Metabolism(metabolism_config)
        return metabolism

    def generate_processes(self, config):
        sim_data_path = config['sim_data_path']
        with open(sim_data_path, 'rb') as sim_data_file:
            sim_data = cPickle.load(sim_data_file)

        complexation = self.initialize_complexation(sim_data)
        protein_degradation = self.initialize_protein_degradation(sim_data)
        polypeptide_initiation = self.initialize_polypeptide_initiation(sim_data)
        metabolism = self.initialize_metabolism(sim_data)

        return {
            'complexation': complexation,
            'protein_degradation': protein_degradation,
            'polypeptide_initiation': polypeptide_initiation,
            'metabolism': metabolism}

    def generate_topology(self, config):
        return {
            'complexation': {
                'molecules': ('bulk',)},

            'protein_degradation': {
                'metabolites': ('bulk',),
                'proteins': ('bulk',)},

            'polypeptide_initiation': {
                'environment': ('environment',),
                'listeners': ('listeners',),
                'active_ribosomes': ('unique',),
                'RNAs': ('unique',),
                'subunits': ('bulk',)},

            'metabolism': {
                'metabolites': ('bulk',),
                'catalysts': ('bulk',),
                'kinetics_enzymes': ('bulk',),
                'kinetics_substrates': ('bulk',),
                'amino_acids': ('bulk',),
                'listeners': ('listeners',),
                'environment': ('environment',),
                'polypeptide_elongation': ('process_state', 'polypeptide_elongation')}}


def test_ecoli():
    ecoli = Ecoli({})

    initial_state = {
        'environment': {
            'media_id': 'minimal'},
        'listeners': {
            'mass': {
                'cell_mass': 1.0,
                'dry_mass': 1.0}},
        'bulk': {},
        'unique': {},
        'process_state': {
            'polypeptide_elongation': {}}}

    settings = {
        'timestep': 1,
        'total_time': 10,
        'initial_state': initial_state}

    data = simulate_compartment_in_experiment(ecoli, settings)

    print(data)


if __name__ == '__main__':
    test_ecoli()
