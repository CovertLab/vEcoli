from warnings import warn
import numpy as np

from vivarium.core.emitter import timeseries_from_data

from ecoli.composites.ecoli_master import run_ecoli

from ecoli.analysis.tablereader_utils import (
    replace_scalars, replace_scalars_2d, camel_case_to_underscored)

ANY_STRING = (bytes, str)

MAPPING = {
    'BulkMolecules': {
        # Use Blame to get ATP used per process
        'atpAllocatedFinal': None,
        'counts': ('bulk',),
        'atpRequested': ('listeners', 'atp_requested',),
        'atpAllocatedInitial': ('listeners', 'atp_allocated_initial',),
    },
    'EnzymeKinetics': {
        'actualFluxes': ('listeners', 'enzyme_kinetics', 'actual_fluxes'),
        'metaboliteCountsFinal': ('listeners', 'enzyme_kinetics', 'metabolite_counts_final'),
        'targetFluxesLower': ('listeners', 'enzyme_kinetics', 'target_fluxes_lower'),
        'metaboliteCountsInit': ('listeners', 'enzyme_kinetics', 'metabolite_counts_init'),
        'targetFluxesUpper': ('listeners', 'enzyme_kinetics', 'target_fluxes_upper'),
        'countsToMolar': ('listeners', 'enzyme_kinetics', 'counts_to_molar'),
        'simulationStep': None,
        'time': ('time',),
        'enzymeCountsInit': ('listeners', 'enzyme_kinetics', 'enzyme_counts_init'),
        'targetFluxes': ('listeners', 'enzyme_kinetics', 'target_fluxes'),
    },
    'GrowthLimits': {
        'aaAllocated': ('listeners', 'growth_limits', 'aa_allocated'),
        'aasUsed': ('listeners', 'growth_limits', 'aas_used'),
        'ntpRequestSize': ('listeners', 'growth_limits', 'ntp_request_size'),
        'aaPoolSize': None,
        'activeRibosomeAllocated': None,
        'ntpUsed': None,
        'aaRequestSize': None,
        'rela_syn': None,
        'aa_supply': None,
        'fraction_trna_charged': None,
        'simulationStep': None,
        'aa_supply_aa_conc': None,
        'net_charged': ('listeners', 'growth_limits', 'net_charged', replace_scalars),
        'spot_deg': None,
        'aa_supply_enzymes': None,
        'ntpAllocated': None,
        'spot_syn': None,
        'aa_supply_fraction': None,
        'ntpPoolSize': None,
        'time': ('time', ),
        'attributes': None
    },
    'mRNACounts': {
        'mRNA_counts': ('listeners', 'mRNA_counts',),
        'simulationStep': None,
        'full_mRNA_counts': None,
        'partial_mRNA_counts': None,
        'time': ('time', ),
        'attributes': None
    },
    'RnapData': {
        'active_rnap_coordinates': None,
        'active_rnap_domain_indexes': None,
        'active_rnap_n_bound_ribosomes': None,
        'active_rnap_unique_indexes': None,
        'actualElongations': ('listeners', 'rnap_data', 'actual_elongations'),
        'codirectional_collision_coordinates': (
            'listeners', 'rnap_data', 'codirectional_collision_coordinates', replace_scalars),
        'didInitialize': ('listeners', 'rnap_data', 'did_initialize'),
        'didStall': ('listeners', 'rnap_data', 'did_stall'),
        'didTerminate': ('listeners', 'rnap_data', 'did_terminate'),
        'headon_collision_coordinates': (
            'listeners', 'rnap_data', 'headon_collision_coordinates', replace_scalars),
        'n_codirectional_collisions': None,
        'n_headon_collisions': None,
        'n_removed_ribosomes': None,
        'n_total_collisions': None,
        'rnaInitEvent': ('listeners', 'rnap_data', 'rna_init_event', replace_scalars),
        'simulationStep': None,
        'terminationLoss': None,
        'time': ('time', ),
        'attributes': None
    },
    'UniqueMoleculeCounts': {
        'simulationStep': None,
        'time': ('time', ),
        'uniqueMoleculeCounts': None,
        'attributes': None
    },
    'ComplexationListener': {
        'complexationEvents': ('listeners', 'complexation_events', replace_scalars),
        'simulationStep': None,
        'time': ('time', ),
        'attributes': None
    },
    'EquilibriumListener': {
        'reactionRates': ('listeners', 'equilibrium_listener', 'reaction_rates', replace_scalars),
        'simulationStep': None,
        'time': ('time', ),
        'attributes': None
    },
    'Main': {
        'time': ('time', ),
        'timeStepSec': None,
        'attributes': None
    },
    'ReplicationData': {
        'fork_coordinates': None,
        'free_DnaA_boxes': None,
        'criticalInitiationMass': ('listeners', 'replication_data', 'critical_initiation_mass'),
        'fork_domains': None,
        'numberOfOric': None,
        'criticalMassPerOriC': ('listeners', 'replication_data', 'critical_mass_per_oriC'),
        'fork_unique_index': None,
        'total_DnaA_boxes': None,
        'attributes': {}
    },
    'RnaSynthProb': {
        'nActualBound': ('listeners', 'rna_synth_prob', 'n_actual_bound', replace_scalars),
        'rnaSynthProb': ('listeners', 'rna_synth_prob', 'rna_synth_prob', replace_scalars),
        'bound_TF_coordinates': None,
        'n_available_promoters': (
            'listeners', 'rna_synth_prob', 'n_available_promoters', replace_scalars),
        'simulationStep': None,
        'bound_TF_domains': None,
        'n_bound_TF_per_TU': ('listeners', 'rna_synth_prob', 'n_bound_TF_per_TU', replace_scalars_2d),
        'time': ('time', ),
        'bound_TF_indexes': None,
        'nPromoterBound': ('listeners', 'rna_synth_prob', 'n_promoter_bound', replace_scalars),
        'gene_copy_number': ('listeners', 'rna_synth_prob', 'gene_copy_number', replace_scalars),
        'pPromoterBound': ('listeners', 'rna_synth_prob', 'p_promoter_bound', replace_scalars),
        'attributes': None
    },
    'UniqueMolecules': {
        'attributes': None
    },
    'DnaSupercoiling': {
        'segment_domain_indexes': None,
        'segment_left_boundary_coordinates': None,
        'segment_right_boundary_coordinates': None,
        'segment_superhelical_densities': None,
        'simulationStep': None,
        'time': ('time', ),
        'attributes': {}
    },
    'EvaluationTime': {
        'append_times': None,
        'merge_times': None,
        'append_total': None,
        'merge_total': None,
        'partition_times': None,
        'calculate_mass_times': None,
        'partition_total': None,
        'calculate_mass_total': None,
        'simulationStep': None,
        'calculate_request_times': None,
        'time': ('time', ),
        'calculate_request_total': None,
        'update_queries_times': None,
        'clock_time': None,
        'update_queries_total': None,
        'evolve_state_times': None,
        'update_times': None,
        'evolve_state_total': None,
        'update_total': None,
        'attributes': None
    },
    'Mass': {
        'inner_membrane_mass': ('listeners', 'mass', 'inner_membrane_mass'),
        'proteinMass': ('listeners', 'mass', 'protein_mass'),
        'cellMass': ('listeners', 'mass', 'cell_mass'),
        'instantaneousGrowthRate': ('listeners', 'mass', 'instantaneous_growth_rate'),
        'rnaMass': ('listeners', 'mass', 'rna_mass'),
        'cellVolume': ('listeners', 'mass', 'volume'),
        'membrane_mass': ('listeners', 'mass', 'membrane_mass'),
        'rRnaMass': ('listeners', 'mass', 'rRna_mass'),
        'cytosol_mass': ('listeners', 'mass', 'cytosol_mass'),
        'mRnaMass': ('listeners', 'mass', 'mRna_mass'),
        'simulationStep': None,
        'dnaMass': ('listeners', 'mass', 'dna_mass'),
        'outer_membrane_mass': ('listeners', 'mass', 'outer_membrane_mass'),
        'smallMoleculeMass': ('listeners', 'mass', 'smallMolecule_mass'),
        'dryMass': ('listeners', 'mass', 'dry_mass'),
        'periplasm_mass': ('listeners', 'mass', 'periplasm_mass'),
        'time': ('time', ),
        'extracellular_mass': ('listeners', 'mass', 'extracellular_mass'),
        'pilus_mass': ('listeners', 'mass', 'pilus_mass'),
        'tRnaMass': ('listeners', 'mass', 'tRna_mass'),
        'flagellum_mass': ('listeners', 'mass', 'flagellum_mass'),
        'processMassDifferences': None,
        'waterMass': None,
        'growth': None,
        'projection_mass': ('listeners', 'mass', 'projection_mass'),
        'attributes': None
    },
    'RibosomeData': {
        'aaCountInSequence': (
            'listeners', 'ribosome_data', 'aa_count_in_sequence', replace_scalars),
        'aaCounts': (
            'listeners', 'ribosome_data', 'aa_counts', replace_scalars),
        'actualElongationHist': (
            'listeners', 'ribosome_data', 'actual_elongation_hist', replace_scalars),
        'actualElongations': ('listeners', 'ribosome_data', 'actual_elongations'),
        'didInitialize': None,
        'didTerminate': ('listeners', 'ribosome_data', 'did_terminate'),
        'effectiveElongationRate': ('listeners', 'ribosome_data', 'effective_elongation_rate'),
        'elongationsNonTerminatingHist': (
            'listeners', 'ribosome_data', 'elongations_non_terminating_hist', replace_scalars),
        'n_ribosomes_on_partial_mRNA_per_transcript': None,
        'n_ribosomes_per_transcript': None,
        'numTrpATerminated': ('listeners', 'ribosome_data', 'num_trpA_terminated'),
        'probTranslationPerTranscript': ('listeners',
                                         'ribosome_data',
                                         'prob_translation_per_transcript',
                                         replace_scalars),
        'processElongationRate': None,
        'rrn16S_init_prob': None,
        'rrn16S_produced': None,
        'rrn23S_init_prob': None,
        'rrn23S_produced': None,
        'rrn5S_init_prob': None,
        'rrn5S_produced': None,
        'simulationStep': None,
        'terminationLoss': None,
        'time': ('time', ),
        'total_rna_init': None,
        'translationSupply': (
            'listeners', 'ribosome_data', 'translation_supply'),
        'attributes': None
    },
    'Environment': {
        'media_concentrations': None,
        'media_id': None,
        'attributes': None
    },
    'FBAResults': {
        'objectiveValue': ('listeners', 'fba_results', 'objective_value', replace_scalars),
        'catalyst_counts': ('listeners', 'fba_results', 'catalyst_counts', replace_scalars),
        'reactionFluxes': ('listeners', 'fba_results', 'reaction_fluxes', replace_scalars),
        'coefficient': None,
        'reducedCosts': ('listeners', 'fba_results', 'reduced_costs', replace_scalars),
        'conc_updates': ('listeners', 'fba_results', 'conc_updates', replace_scalars),
        'shadowPrices': ('listeners', 'fba_results', 'shadow_prices', replace_scalars),
        'constrained_molecules': (
            'listeners', 'fba_results', 'constrained_molecules', replace_scalars),
        'simulationStep': None,
        'deltaMetabolites': (
            'listeners', 'fba_results', 'delta_metabolites', replace_scalars),
        'targetConcentrations': (
            'listeners', 'fba_results', 'target_concentrations', replace_scalars),
        'externalExchangeFluxes': (
            'listeners', 'fba_results', 'external_exchange_fluxes', replace_scalars),
        'time': ('time', ),
        'homeostaticObjectiveValues': (
            'listeners', 'fba_results', 'homeostatic_objective_values', replace_scalars),
        'translation_gtp': None,
        'kineticObjectiveValues': (
            'listeners', 'fba_results', 'kinetic_objective_values', replace_scalars),
        'unconstrained_molecules': (
            'listeners', 'fba_results', 'unconstrained_molecules', replace_scalars),
        'media_id': None,
        'uptake_constraints': (
            'listeners', 'fba_results', 'uptake_constraints', replace_scalars),
        'attributes': None
    },
    'MonomerCounts': {
        'monomerCounts': ('listeners', 'monomer_counts'),
        'simulationStep': None,
        'time': ('time', ),
        'attributes': None
    },
    'RnaDegradationListener': {
        'fragmentBasesDigested': None,
        'countRnaDegraded': (
            'listeners', 'rna_degradation_listener', 'count_rna_degraded', replace_scalars),
        'nucleotidesFromDegradation': None,
        'DiffRelativeFirstOrderDecay': None,
        'simulationStep': None,
        'FractEndoRRnaCounts': (
            'listeners', 'rna_degradation_listener', 'fract_endo_rrna_counts'),
        'time': ('time', ),
        'FractionActiveEndoRNases': (
            'listeners', 'rna_degradation_listener', 'fraction_active_endo_rrnases'),
        'attributes': None
    },
    'TranscriptElongationListener': {
        'attenuation_probability': (
            'listeners', 'transcript_elongation_listener', 'attentuation_probability', replace_scalars),
        'countRnaSynthesized': (
            'listeners', 'transcript_elongation_listener', 'count_rna_synthesized', replace_scalars),
        'time': ('time', ),
        'counts_attenuated': (
            'listeners', 'transcript_elongation_listener', 'counts_attenuated', replace_scalars),
        'countNTPsUSed': (
            'listeners', 'transcript_elongation_listener', 'count_NTPs_used'),
        'simulationStep': None,
        'attributes': None
    }
}

class TableReaderError(Exception):
	"""
	Base exception class for TableReader-associated exceptions.
	"""
	pass


class VersionError(TableReaderError):
	"""
	An error raised when the input files claim to be from a different format or
	version of the file specification.
	"""
	pass


class DoesNotExistError(TableReaderError):
	"""
	An error raised when a column or attribute does not seem to exist.
	"""
	pass


class VariableLengthColumnError(TableReaderError):
	"""
	An error raised when the user tries to access subcolumns of a variable
	length column.
	"""
	pass


class TableReader(object):
    """
    Fake TableReader. In wcEcoli, the TableReader class was used to access data saved from simulations.
    This class provides a bridge in order to port analyses over to vivarium-ecoli without significant modification.
    Given a path within the wcEcoli output structure and timeseries data from a vivarium-ecoli experiment,
    this class provides a way to retrieve data as if it were structured in the same way as it is in wcEcoli.

    Parameters:
        wc_path (str): Which wcEcoli table this TableReader would be reading from.
        data: timeseries data from a vivarium-ecoli experiment (to be read as if it were structured as in wcEcoli.)
    """

    def __init__(self, path, data, timeseries_data=False):
        # Strip down to table name, in case a full path is given
        path[(path.rfind('/')+1):]
        self._path = path

        # Store reference to the data
        if not timeseries_data:
            data = timeseries_from_data(data)
        self._data = data

        # List the column file names.
        self._mapping = MAPPING[path]
        self._columnNames = {
            k for k in self._mapping.keys() if k != "attributes"}

    @property
    def path(self):
        # type: () -> str
        return self._path

    def readColumn(self, name, indices=None, squeeze=True):
        # type: (str, Any, bool) -> np.ndarray
        """
        Load a full column (all rows). Each row entry is a 1-D NumPy array of
        subcolumns, so the initial result is a 2-D array row x subcolumn, which
        is optionally squeezed to arrays with lower dimensions if squeeze=True.
        In the case of fixed-length columns, this method can optionally read
        just a vertical slice of all those arrays -- the subcolumns at the
        given `indices`. For variable-length columns, np.nan is used as a
        filler value for the empty entries of each row.

        Parameters:
            name: The name of the column.
            indices: The subcolumn indices to select from each entry. This can
                be any value that works to index an ndarray along 1 dimension,
                or None for all the data. Specifying this argument
                for variable-length columns will throw an error.
            squeeze: If True, the resulting NumPy array is squeezed into a 0D,
                1D, or 2D array, depending on the number of rows and subcolumns
                it has.
                1 row x 1 subcolumn => 0D.
                n rows x 1 subcolumn or 1 row x m subcolumns => 1D.
                n rows x m subcolumns => 2D.

        Returns:
            ndarray: A writable 0D, 1D, or 2D array.
        """
        # Squeeze if flag is set to True
        viv_path = self._mapping[name]
        if callable(viv_path):
            result = viv_path(self._data)
        elif isinstance(viv_path, tuple):
            result = self._data
            for elem in viv_path:
                if callable(elem):
                    result = elem(result)
                else:
                    result = result[elem]
        else:
            # No explicit mapping defined, try heuristic mapping
            heuristic_path = ('listeners',
                              camel_case_to_underscored(self._path),
                              camel_case_to_underscored(name))

            warn(f'No explicit mapping defined from {self._path + "/" + name} to a path in vivarium data,\n'
                 f'trying heuristic mapping: {heuristic_path}.\n'
                 'If this works, consider adding an explicit mapping in tablereader.py!')

            result = self._data
            for elem in heuristic_path:
                result = result[elem]

        result = np.array(result).T

        # extract indices
        if indices is not None:
            result = result[:, indices]

        if squeeze:
            result = result.squeeze()

        return result

    def readSubcolumn(self, column, subcolumn_name):
        # type: (str, str) -> np.ndarray
        """Read in a subcolumn from a table by name

        Each column of a table is a 2D matrix. The SUBCOLUMNS_KEY attribute
        defines a map from column name to a name for an attribute that
        stores a list of names such that the i-th name describes the i-th
        subcolumn.

        Arguments:
            column: Name of the column.
            subcolumn_name: Name of the ID or object associated with the
                    desired subcolumn.

        Returns:
            The subcolumn, as a 1-dimensional array.
        """
        # subcol_name_map = self.readAttribute(SUBCOLUMNS_KEY)
        # subcols = self.readAttribute(subcol_name_map[column])
        # index = subcols.index(subcolumn_name)
        # return self.readColumn(column, [index], squeeze=False)[:, 0]
        raise NotImplementedError()

    def columnNames(self):
        """
        Returns the names of all columns.
        """
        return list(self._columnNames)

    def close(self):
        """
        Does nothing.
        """
        pass


def _check_bulk_inputs(mol_names):
    """
    Use to check and adjust mol_names inputs for functions that read bulk
    molecules to get consistent argument handling in both functions.
    """

    # Wrap an array in a tuple to ensure correct dimensions
    if not isinstance(mol_names, tuple):
        mol_names = (mol_names,)

    # Check for string instead of array since it will cause mol_indices lookup to fail
    for names in mol_names:
        if isinstance(names, ANY_STRING):
            raise Exception('mol_names tuple must contain arrays not strings like {!r}'.format(names))

    return mol_names


def read_bulk_molecule_counts(data, mol_names):
    '''
    Reads a subset of molecule counts from BulkMolecules using the indexing method
    of readColumn. Should only be called once per simulation being analyzed with
    all molecules of interest.
    '''

    mol_names = _check_bulk_inputs(mol_names)

    bulk_reader = TableReader('BulkMolecules', data)
    bulk_molecule_names = bulk_reader.readColumn("objectNames")
    mol_indices = {mol: i for i, mol in enumerate(bulk_molecule_names)}

    lengths = [len(names) for names in mol_names]
    indices = np.hstack([[mol_indices[mol] for mol in names] for names in mol_names])
    bulk_counts = bulk_reader.readColumn('counts', indices, squeeze=False)

    start_slice = 0
    for length in lengths:
        counts = bulk_counts[:, start_slice:start_slice + length].squeeze()
        start_slice += length
        yield counts


def test_table_reader():
    data = run_ecoli(total_time=4, time_series=False)

    # TODO actaully grab their values - they fail 'gracefully' rn because their keys are empty or arrays are empty
    equi_tb = TableReader("EquilibriumListener", data)
    equi_rxns = equi_tb.readColumn('reactionRates')

    fba_tb = TableReader("FBAResults", data)
    fba_rxns = fba_tb.readColumn('reactionFluxes')

    growth_lim_tb = TableReader("GrowthLimits", data)
    growth_lim_vals = growth_lim_tb.readColumn('net_charged')

    # i believe these are right
    dry_m_tb = TableReader("Mass", data)
    dry_m_vals = dry_m_tb.readColumn('dryMass')

    time_tb = TableReader("Main", data)
    time_vals = time_tb.readColumn('time')

    cell_m_tb = TableReader("Mass", data)
    cell_m_vals = cell_m_tb.readColumn('cellMass')

    bulk_tb = TableReader("BulkMolecules", data)
    bulk_counts = bulk_tb.readColumn("counts")

    rnap_tb = TableReader("RnapData", data)
    rna_init = rnap_tb.readColumn("rna_init_event")

    rna_synth_tb = TableReader("RnaSynthProb", data)
    tf_per_tu = rna_synth_tb.readColumn("n_bound_TF_per_TU")
    gene_copies = rna_synth_tb.readColumn("gene_copy_number")
    rna_synth_prob = rna_synth_tb.readColumn("rnaSynthProb")

    ribosome_tb = TableReader("RibosomeData", data)
    prob_trans = ribosome_tb.readColumn("probTranslationPerTranscript")


if __name__ == "__main__":
    test_table_reader()
