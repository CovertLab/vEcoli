from warnings import warn
import numpy as np

from vivarium.core.emitter import timeseries_from_data

from ecoli.library.schema import array_from, key_array_from
from ecoli.composites.ecoli_nonpartition import run_ecoli

from ecoli.analysis.tablereader_utils import (
    warn_incomplete, replace_scalars, replace_scalars_2d, camel_case_to_underscored)

ANY_STRING = (bytes, str)

MAPPING = {
    'BulkMolecules': {
        'atpAllocatedFinal': None,
        'atpRequested': None,
        'counts': ("bulk", array_from),
        'atpAllocatedInitial': None,
        'attributes': None,
        'objectNames': ("bulk", key_array_from),
    },
    'EnzymeKinetics': {
        'actualFluxes': None,
        'metaboliteCountsFinal': None,
        'targetFluxesLower': None,
        'metaboliteCountsInit': None,
        'targetFluxesUpper': None,
        'countsToMolar': None,
        'simulationStep': None,
        'time': ('time', ),
        'enzymeCountsInit': None,
        'targetFluxes': None,
        'attributes': None
    },
    'GrowthLimits': {
        'aaAllocated': None,
        'aasUsed': None,
        'ntpRequestSize': None,
        'aaPoolSize': None,
        'activeRibosomeAllocated': None,
        'ntpUsed': None,
        'aaRequestSize': None,
        'rela_syn': None,
        'aa_supply': None,
        'fraction_trna_charged': None,
        'simulationStep': None,
        'aa_supply_aa_conc': None,
        'net_charged': ('listeners', 'growth_limits', warn_incomplete),
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
        'actualElongations': None,
        'codirectional_collision_coordinates': None,
        'didInitialize': None,
        'didStall': None,
        'didTerminate': None,
        'headon_collision_coordinates': None,
        'n_codirectional_collisions': None,
        'n_headon_collisions': None,
        'n_removed_ribosomes': None,
        'n_total_collisions': None,
        'rnaInitEvent': ('listeners', 'rnap_data', 'rnaInitEvent', replace_scalars),
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
        'complexationEvents': None,
        'simulationStep': None,
        'time': ('time', ),
        'attributes': None
    },
    'EquilibriumListener': {
        'reactionRates': ('listeners', 'equilibrium_listener', warn_incomplete),
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
        'criticalInitiationMass': None,
        'fork_domains': None,
        'numberOfOric': None,
        'criticalMassPerOriC': None,
        'fork_unique_index': None,
        'total_DnaA_boxes': None,
        'attributes': {}
    },
    'RnaSynthProb': {
        'nActualBound': None,
        'rnaSynthProb': ('listeners', 'rna_synth_prob', 'rna_synth_prob', replace_scalars),
        'bound_TF_coordinates': None,
        'n_available_promoters': None,
        'simulationStep': None,
        'bound_TF_domains': None,
        'n_bound_TF_per_TU': ('listeners', 'rna_synth_prob', 'n_bound_TF_per_TU', replace_scalars_2d),
        'time': ('time', ),
        'bound_TF_indexes': None,
        'nPromoterBound': None,
        'gene_copy_number': None,
        'pPromoterBound': None,
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
        'proteinMass': ('listeners', 'mass', 'proteinMass'),
        'cellMass': ('listeners', 'mass', 'cell_mass'),
        'instantaniousGrowthRate': None,
        'rnaMass': ('listeners', 'mass', 'rnaMass'),
        'cellVolume': None,
        'membrane_mass': ('listeners', 'mass', 'membrane_mass'),
        'rRnaMass': ('listeners', 'mass', 'rRnaMass'),
        'cytosol_mass': ('listeners', 'mass', 'cytosol_mass'),
        'mRnaMass': ('listeners', 'mass', 'mRnaMass'),
        'simulationStep': None,
        'dnaMass': ('listeners', 'mass', 'dnaMass'),
        'outer_membrane_mass': ('listeners', 'mass', 'outer_membrane_mass'),
        'smallMoleculeMass': ('listeners', 'mass', 'smallMoleculeMass'),
        'dryMass': ('listeners', 'mass', 'dry_mass'),
        'periplasm_mass': ('listeners', 'mass', 'periplasm_mass'),
        'time': ('time', ),
        'extracellular_mass': ('listeners', 'mass', 'extracellular_mass'),
        'pilus_mass': ('listeners', 'mass', 'pilus_mass'),
        'tRnaMass': ('listeners', 'mass', 'tRnaMass'),
        'flagellum_mass': ('listeners', 'mass', 'flagellum_mass'),
        'processMassDifferences': None,
        'waterMass': None,
        'growth': None,
        'projection_mass': ('listeners', 'mass', 'projection_mass'),
        'attributes': None
    },
    'RibosomeData': {
        'aaCountInSequence': None,
        'aaCounts': None,
        'actualElongationHist': None,
        'actualElongations': None,
        'didInitialize': None,
        'didTerminate': None,
        'effectiveElongationRate': None,
        'elongationsNonTerminatingHist': None,
        'n_ribosomes_on_partial_mRNA_per_transcript': None,
        'n_ribosomes_per_transcript': None,
        'numTrpATerminated': None,
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
        'translationSupply': None,
        'attributes': None
    },
    'Environment': {
        'media_concentrations': None,
        'media_id': None,
        'attributes': None
    },
    'FBAResults': {
        'objectiveValue': None,
        'catalyst_counts': None,
        'reactionFluxes': ('listeners', 'fba_results', warn_incomplete),
        'coefficient': None,
        'reducedCosts': None,
        'conc_updates': None,
        'shadowPrices': None,
        'constrained_molecules': None,
        'simulationStep': None,
        'deltaMetabolites': None,
        'targetConcentrations': None,
        'externalExchangeFluxes': None,
        'time': ('time', ),
        'homeostaticObjectiveValues': None,
        'translation_gtp': None,
        'kineticObjectiveValues': None,
        'unconstrained_molecules': None,
        'media_id': None,
        'uptake_constraints': None,
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
        'countRnaDegraded': None,
        'nucleotidesFromDegradation': None,
        'DiffRelativeFirstOrderDecay': None,
        'simulationStep': None,
        'FractEndoRRnaCounts': None,
        'time': ('time', ),
        'FractionActiveEndoRNases': None,
        'attributes': None
    },
    'TranscriptElongationListener': {
        'attenuation_probability': None,
        'countRnaSynthesized': None,
        'time': ('time', ),
        'counts_attenuated': None,
        'countNTPsUSed': None,
        'simulationStep': None,
        'attributes': None
    }
}


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

        # Get attributes
        self._attributes = self._mapping['attributes']

    @property
    def path(self):
        # type: () -> str
        return self._path

    def readAttribute(self, name):
        # type: (str) -> Any
        """
        Return an attribute value.

        Parameters:
                name: The attribute name.

        Returns:
                value: The attribute value, JSON-deserialized from a string.
        """

        if name not in self._attributes:
            raise DoesNotExistError("No such attribute: {}".format(name))
        return self._attributes[name]

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

    def allAttributeNames(self):
        """
        Returns a list of all attribute names including Table metadata.
        """
        return list(self._attributes.keys())

    def attributeNames(self):
        """
        Returns a list of ordinary (client-provided) attribute names.
        """
        names = [key for key in self._attributes if not key.startswith('_')]
        return names

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
    rna_init = rnap_tb.readColumn("rnaInitEvent")

    rna_synth_tb = TableReader("RnaSynthProb", data)
    tf_per_tu = rna_synth_tb.readColumn("n_bound_TF_per_TU")
    #gene_copies = rna_synth_tb.readColumn("gene_copy_number")
    rna_synth_prob = rna_synth_tb.readColumn("rnaSynthProb")

    ribosome_tb = TableReader("RibosomeData", data)
    prob_trans = ribosome_tb.readColumn("probTranslationPerTranscript")


if __name__ == "__main__":
    test_table_reader()
