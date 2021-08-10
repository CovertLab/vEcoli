import numpy as np

from ecoli.library.schema import array_from
from ecoli.composites.ecoli_master import run_ecoli


def replace_scalars(array):
    for value in array:
        if value != [] and type(value) in {list, np.array}:
            array_len = len(value)
            break

    for i in range(len(array)):
        if array[i] == [] or type(array[i]) not in {list, np.array}:
            array[i] = [0 for i in range(array_len)]

    array = np.array(array)
    return array


def replace_scalars_2d(array):
    for value in array:
        if value != [] and type(value) in {list, np.array}:
            rows = len(value)
            cols = len(value[0])
            break

    for i in range(len(array)):
        if array[i] == [] or type(array[i]) not in {list, np.array}:
            array[i] = [[0 for i in range(cols)] for i in range(rows)]

    array = np.array(array)
    return array


MAPPING = {
    'BulkMolecules': {
        'atpAllocatedFinal': None,
        'atpRequested': None,
        'counts': ("bulk", array_from),
        'atpAllocatedInitial': None,
        'attributes': None
    },
    'EnzymeKinetics': {
        'actualFluxes': None,
        'metaboliteCountsFinal': None,
        'targetFluxesLower': None,
        'metaboliteCountsInit': None,
        'targetFluxesUpper': None,
        'countsToMolar': None,
        'simulationStep': None,
        'time': None,
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
        'net_charged': ('listeners', 'growth_limits'),
        'spot_deg': None,
        'aa_supply_enzymes': None,
        'ntpAllocated': None,
        'spot_syn': None,
        'aa_supply_fraction': None,
        'ntpPoolSize': None,
        'time': None,
        'attributes': None
    },
    'mRNACounts': {
        'mRNA_counts': None,
        'simulationStep': None,
        'full_mRNA_counts': None,
        'partial_mRNA_counts': None,
        'time': None,
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
        'time': None,
        'attributes': None
    },
    'UniqueMoleculeCounts': {
        'simulationStep': None,
        'time': None,
        'uniqueMoleculeCounts': None,
        'attributes': None
    },
    'ComplexationListener': {
        'complexationEvents': None,
        'simulationStep': None,
        'time': None,
        'attributes': None
    },
    'EquilibriumListener': {
        'reactionRates': ('listeners', 'equilibrium_listener'),
        'simulationStep': None,
        'time': None,
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
        'attributes': None
    },
    'RnaSynthProb': {
        'nActualBound': None,
        'rnaSynthProb': ('listeners', 'rna_synth_prob', 'rna_synth_prob', replace_scalars),
        'bound_TF_coordinates': None,
        'n_available_promoters': None,
        'simulationStep': None,
        'bound_TF_domains': None,
        'n_bound_TF_per_TU': ('listeners', 'rna_synth_prob', 'n_bound_TF_per_TU', replace_scalars_2d),
        'time': None,
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
        'time': None,
        'attributes': None
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
        'time': None,
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
        'inner_membrane_mass': None,
        'proteinMass': None,
        'cellMass': ('listeners', 'mass', 'cell_mass'),
        'instantaniousGrowthRate': None,
        'rnaMass': None,
        'cellVolume': None,
        'membrane_mass': None,
        'rRnaMass': None,
        'cytosol_mass': None,
        'mRnaMass': None,
        'simulationStep': None,
        'dnaMass': None,
        'outer_membrane_mass': None,
        'smallMoleculeMass': None,
        'dryMass': ('listeners', 'mass', 'dry_mass'),
        'periplasm_mass': None,
        'time': None,
        'extracellular_mass': None,
        'pilus_mass': None,
        'tRnaMass': None,
        'flagellum': None,
        'processMassDifferences': None,
        'waterMass': None,
        'growth': None,
        'projection_mass': None,
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
        'time': None,
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
        'reactionFluxes': ('listeners', 'fba_results'),
        'coefficient': None,
        'reducedCosts': None,
        'conc_updates': None,
        'shadowPrices': None,
        'constrained_molecules': None,
        'simulationStep': None,
        'deltaMetabolites': None,
        'targetConcentrations': None,
        'externalExchangeFluxes': None,
        'time': None,
        'homeostaticObjectiveValues': None,
        'translation_gtp': None,
        'kineticObjectiveValues': None,
        'unconstrained_molecules': None,
        'media_id': None,
        'uptake_constraints': None,
        'attributes': None
    },
    'MonomerCounts': {
        'monomerCounts': None,
        'simulationStep': None,
        'time': None,
        'attributes': None
    },
    'RnaDegradationListener': {
        'fragmentBasesDigested': None,
        'countRnaDegraded': None,
        'nucleotidesFromDegradation': None,
        'DiffRelativeFirstOrderDecay': None,
        'simulationStep': None,
        'FractEndoRRnaCounts': None,
        'time': None,
        'FractionActiveEndoRNases': None,
        'attributes': None
    },
    'TranscriptElongationListener': {
        'attenuation_probability': None,
        'countRnaSynthesized': None,
        'time': None,
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

    def __init__(self, path, data):
        # Strip down to table name, in case a full path is given
        path[(path.rfind('/')+1):]
        self._path = path

        # Store reference to the data
        self._data = data

        # Read the table's attributes file
        #attributes_filename = os.path.join(path, tw.FILE_ATTRIBUTES)

        #self._attributes = filepath.read_json_file(attributes_filename)

        # List the column file names. Ignore the 'attributes.json' file.
        self._mapping = MAPPING[path]
        self._columnNames = {
            k for k in self._mapping.keys() if k != "attributes"}

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

        raise NotImplementedError()
        # if name not in self._attributes:
        # 	raise DoesNotExistError("No such attribute: {}".format(name))
        # return self._attributes[name]

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

        # if name not in self._columnNames:
        #     raise DoesNotExistError("No such column: {}".format(name))

        # entry_blocks = []  # type: List[bytes]
        # row_size_blocks = []

        # # Read the header and read, decompress, and unpack all the blocks.
        # with open(os.path.join(self._path, name), 'rb') as dataFile:
        #     chunk = Chunk(dataFile, align=False)
        #     header = _ColumnHeader(chunk)
        #     variable_length = header.variable_length
        #     chunk.close()

        #     if variable_length and indices is not None:
        #         raise VariableLengthColumnError(
        #             'Attempted to access subcolumns of a variable-length column {}.'.format(name))

        #     # Variable-length columns should not be squeezed.
        #     if variable_length:
        #         squeeze = False

        #     if header.compression_type == tw.COMPRESSION_TYPE_ZLIB:
        #         def decompressor(data_bytes): return zlib.decompress(
        #             data_bytes)  # type: Callable[[bytes], bytes]
        #     else:
        #         def decompressor(data_bytes): return data_bytes

        #     while True:
        #         try:
        #             chunk = Chunk(dataFile, align=False)
        #         except EOFError:
        #             break

        #         if chunk.getname() == tw.BLOCK_CHUNK_TYPE:
        #             raw_entry = chunk.read()
        #             if len(raw_entry) != chunk.getsize():
        #                 raise EOFError('Data block cut short {}/{}'.format(
        #                     len(raw_entry), chunk.getsize()))
        #             entry_blocks.append(raw_entry)

        #         elif chunk.getname() == tw.ROW_SIZE_CHUNK_TYPE:
        #             row_sizes = chunk.read()
        #             if len(row_sizes) != chunk.getsize():
        #                 raise EOFError('Row sizes block cut short {}/{}'.format(
        #                     len(row_sizes), chunk.getsize()))
        #             row_size_blocks.append(row_sizes)

        #         chunk.close()  # skips to the next chunk

        # if variable_length and len(entry_blocks) != len(row_size_blocks):
        #     raise EOFError('Number of entry blocks ({}) does not match number of row size blocks ({}).'.format(
        #         len(entry_blocks), len(row_size_blocks)))

        # del raw_entry  # release the block ref

        # # Variable-length columns
        # if variable_length:
        #     # Concatenate row sizes array
        #     row_sizes_list = [
        #         np.frombuffer(block, tw.ROW_SIZE_CHUNK_DTYPE)
        #         for block in row_size_blocks]  # type: List[Iterable[int]]
        #     all_row_sizes = np.concatenate(row_sizes_list)

        #     # Initialize results array to NaNs
        #     result = np.full((len(all_row_sizes), all_row_sizes.max()), np.nan)

        #     row = 0
        #     for raw_entry, row_sizes_ in zip(entry_blocks, row_sizes_list):
        #         entries = decomp(raw_entry)
        #         entry_idx = 0

        #         # Fill each row with the length given by values in row_sizes_
        #         for row_size in row_sizes_:
        #             result[row, :row_size] = entries[entry_idx: (
        #                 entry_idx + row_size)]
        #             entry_idx += row_size
        #             row += 1

        # # Constant-length columns
        # else:
        #     # Decompress the last block to get its shape, then allocate the result.
        #     last_entries = decomp(entry_blocks.pop())
        #     last_num_rows = last_entries.shape[0]
        #     num_rows = len(entry_blocks) * \
        #         header.entries_per_block + last_num_rows
        #     num_subcolumns = header.elements_per_entry if indices is None else len(
        #         indices)
        #     result = np.zeros((num_rows, num_subcolumns), header.dtype)

        #     row = 0
        #     for raw_entry in entry_blocks:
        #         entries = decomp(raw_entry)
        #         additional_rows = entries.shape[0]
        #         result[row: (row + additional_rows)] = entries
        #         row += additional_rows

        #     result[row: (row + last_num_rows)] = last_entries

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
            raise NotImplementedError(f"No mapping implented from {self._path + '/' + name} to a path in vivarium data"
                                      f"(mapping is {viv_path}).")

        result = np.array(result)

        # TODO: indices

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


def test_table_reader():
    data = run_ecoli(total_time=4)

    #TODO actaully grab their values - they fail 'gracefully' rn because their keys are empty or arrays are empty
    equi_tb = TableReader("EquilibriumListener", data)
    equi_rxns = equi_tb.readColumn('reactionRates')

    fba_tb = TableReader("FBAResults", data)
    fba_rxns = fba_tb.readColumn('reactionFluxes')

    growth_lim_tb = TableReader("GrowthLimits", data)
    growth_lim_vals = growth_lim_tb.readColumn('net_charged')

    #i believe these are right
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
