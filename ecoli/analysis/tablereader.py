MAPPING = {
    'BulkMolecules': {
        'atpAllocatedFinal',
        'atpRequested',
        'counts',
        'atpAllocatedInitial',
        'attributes'
    },
    'EnzymeKinetics': {
        'actualFluxes',
        'metaboliteCountsFinal',
        'targetFluxesLower',
        'metaboliteCountsInit',
        'targetFluxesUpper',
        'countsToMolar',
        'simulationStep',
        'time',
        'enzymeCountsInit',
        'targetFluxes',
        'attributes'
    },
    'GrowthLimits': {
        'aaAllocated',
        'aasUsed',
        'ntpRequestSize',
        'aaPoolSize',
        'activeRibosomeAllocated',
        'ntpUsed',
        'aaRequestSize',
        'rela_syn',
        'aa_supply',
        'fraction_trna_charged',
        'simulationStep',
        'aa_supply_aa_conc'
        'net_charged',
        'spot_deg',
        'aa_supply_enzymes',
        'ntpAllocated',
        'spot_syn',
        'aa_supply_fraction',
        'ntpPoolSize',
        'time',
        'attributes'
    },
    'mRNACounts': {
        'mRNA_counts',
        'simulationStep',
        'full_mRNA_counts',
        'partial_mRNA_counts',
        'time',
        'attributes'
    },
    'RnapData': {
        'active_rnap_coordinates',
        'active_rnap_domain_indexes',
        'active_rnap_n_bound_ribosomes',
        'active_rnap_unique_indexes',
        'actualElongations',
        'codirectional_collision_coordinates',
        'didInitialize',
        'didStall',
        'didTerminate',
        'headon_collision_coordinates',
        'n_codirectional_collisions',
        'n_headon_collisions',
        'n_removed_ribosomes',
        'n_total_collisions',
        'rnaInitEvent',
        'simulationStep',
        'terminationLoss',
        'time',
        'attributes'
    },
    'UniqueMoleculeCounts': {
        'simulationStep',
        'time',
        'uniqueMoleculeCounts',
        'attributes'
    },
    'ComplexationListener': {
        'complexationEvents',
        'simulationStep',
        'time',
        'attributes'
    },
    'EquilibriumListener': {
        'reactionRates',
        'simulationStep',
        'time',
        'attributes'
    },
    'Main': {
        'time',
        'timeStepSec',
        'attributes'
    },
    'ReplicationData': {
        'fork_coordinates',
        'free_DnaA_boxes',
        'criticalInitiationMass',
        'fork_domains',
        'numberOfOric',
        'criticalMassPerOriC',
        'fork_unique_index',
        'total_DnaA_boxes',
        'attributes'
    },
    'RnaSynthProb': {
        'nActualBound',
        'rnaSynthProb',
        'bound_TF_coordinates'
        'n_available_promoters',
        'simulationStep',
        'bound_TF_domains',
        'n_bound_TF_per_TU',
        'time',
        'bound_TF_indexes',
        'nPromoterBound',
        'gene_copy_number',
        'pPromoterBound',
        'attributes'
    },
    'UniqueMolecules': {
        'attributes'
    },
    'DnaSupercoiling': {
        'segment_domain_indexes',
        'segment_left_boundary_coordinates',
        'segment_right_boundary_coordinates',
        'segment_superhelical_densities',
        'simulationStep',
        'time',
        'attributes'
    },
    'EvaluationTime': {
        'append_times',
        'merge_times',
        'append_total',
        'merge_total',
        'partition_times',
        'calculate_mass_times',
        'partition_total',
        'calculate_mass_total',
        'simulationStep',
        'calculate_request_times',
        'time',
        'calculate_request_total',
        'update_queries_times',
        'clock_time',
        'update_queries_total',
        'evolve_state_times',
        'update_times',
        'evolve_state_total',
        'update_total',
        'attributes'
    },
    'Mass': {
        'inner_membrane_mass',
        'proteinMass',
        'cellMass',
        'instantaniousGrowthRate',
        'rnaMass',
        'cellVolume',
        'membrane_mass',
        'rRnaMass',
        'cytosol_mass',
        'mRnaMass',
        'simulationStep',
        'dnaMass',
        'outer_membrane_mass',
        'smallMoleculeMass',
        'dryMass',
        'periplasm_mass',
        'time',
        'extracellular_mass',
        'pilus_mass',
        'tRnaMass',
        'flagellum',
        'processMassDifferences',
        'waterMass',
        'growth',
        'projection_mass',
        'attributes'
    },
    'RibosomeData': {
        'aaCountInSequence',
        'aaCounts',
        'actualElongationHist',
        'actualElongations',
        'didInitialize',
        'didTerminate',
        'effectiveElongationRate',
        'elongationsNonTerminatingHist',
        'n_ribosomes_on_partial_mRNA_per_transcript',
        'n_ribosomes_per_transcript',
        'numTrpATerminated',
        'probTranslationPerTranscript',
        'processElongationRate',
        'rrn16S_init_prob',
        'rrn16S_produced',
        'rrn23S_init_prob',
        'rrn23S_produced',
        'rrn5S_init_prob',
        'rrn5S_produced',
        'simulationStep',
        'terminationLoss',
        'time',
        'total_rna_init',
        'translationSupply',
        'attributes'
    },
    'Environment': {
        'media_concentrations',
        'media_id',
        'attributes'
    },
    'FBAResults': {
        'objectiveValue',
        'catalyst_counts',
        'reactionFluxes'
        'coefficient',
        'reducedCosts',
        'conc_updates',
        'shadowPrices',
        'constrained_molecules',
        'simulationStep',
        'deltaMetabolites',
        'targetConcentrations',
        'externalExchangeFluxes',
        'time',
        'homeostaticObjectiveValues',
        'translation_gtp',
        'kineticObjectiveValues',
        'unconstrained_molecules',
        'media_id',
        'uptake_constraints',
        'attributes'
    },
    'MonomerCounts': {
        'monomerCounts',
        'simulationStep',
        'time',
        'attributes'
    },
    'RnaDegradationListener': {
        'fragmentBasesDigested',
        'countRnaDegraded',
        'nucleotidesFromDegradation',
        'DiffRelativeFirstOrderDecay',
        'simulationStep',
        'FractEndoRRnaCounts',
        'time',
        'FractionActiveEndoRNases',
        'attributes'
    },
    'TranscriptElongationListener': {
        'attenuation_probability',
        'countRnaSynthesized',
        'time',
        'counts_attenuated',
        'countNTPsUSed',
        'simulationStep',
        'attributes'
    }
}


class TableReader(object):
    """
    Fake TableReader 

    Parameters:
            path (str): Path to the input location (a directory).
    """

    def __init__(self, path):
        self._path = path

        # Read the table's attributes file
        #attributes_filename = os.path.join(path, tw.FILE_ATTRIBUTES)

    #self._attributes = filepath.read_json_file(attributes_filename)

        # List the column file names. Ignore the 'attributes.json' file.
    #self._columnNames = {p for p in os.listdir(path) if '.json' not in p}

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

        if name not in self._columnNames:
            raise DoesNotExistError("No such column: {}".format(name))

        entry_blocks = []  # type: List[bytes]
        row_size_blocks = []

        # Read the header and read, decompress, and unpack all the blocks.
        with open(os.path.join(self._path, name), 'rb') as dataFile:
            chunk = Chunk(dataFile, align=False)
            header = _ColumnHeader(chunk)
            variable_length = header.variable_length
            chunk.close()

            if variable_length and indices is not None:
                raise VariableLengthColumnError(
                    'Attempted to access subcolumns of a variable-length column {}.'.format(name))

            # Variable-length columns should not be squeezed.
            if variable_length:
                squeeze = False

            if header.compression_type == tw.COMPRESSION_TYPE_ZLIB:
                def decompressor(data_bytes): return zlib.decompress(
                    data_bytes)  # type: Callable[[bytes], bytes]
            else:
                def decompressor(data_bytes): return data_bytes

            while True:
                try:
                    chunk = Chunk(dataFile, align=False)
                except EOFError:
                    break

                if chunk.getname() == tw.BLOCK_CHUNK_TYPE:
                    raw_entry = chunk.read()
                    if len(raw_entry) != chunk.getsize():
                        raise EOFError('Data block cut short {}/{}'.format(
                            len(raw_entry), chunk.getsize()))
                    entry_blocks.append(raw_entry)

                elif chunk.getname() == tw.ROW_SIZE_CHUNK_TYPE:
                    row_sizes = chunk.read()
                    if len(row_sizes) != chunk.getsize():
                        raise EOFError('Row sizes block cut short {}/{}'.format(
                            len(row_sizes), chunk.getsize()))
                    row_size_blocks.append(row_sizes)

                chunk.close()  # skips to the next chunk

        if variable_length and len(entry_blocks) != len(row_size_blocks):
            raise EOFError('Number of entry blocks ({}) does not match number of row size blocks ({}).'.format(
                len(entry_blocks), len(row_size_blocks)))

        del raw_entry  # release the block ref

        # Variable-length columns
        if variable_length:
            # Concatenate row sizes array
            row_sizes_list = [
                np.frombuffer(block, tw.ROW_SIZE_CHUNK_DTYPE)
                for block in row_size_blocks]  # type: List[Iterable[int]]
            all_row_sizes = np.concatenate(row_sizes_list)

            # Initialize results array to NaNs
            result = np.full((len(all_row_sizes), all_row_sizes.max()), np.nan)

            row = 0
            for raw_entry, row_sizes_ in zip(entry_blocks, row_sizes_list):
                entries = decomp(raw_entry)
                entry_idx = 0

                # Fill each row with the length given by values in row_sizes_
                for row_size in row_sizes_:
                    result[row, :row_size] = entries[entry_idx: (
                        entry_idx + row_size)]
                    entry_idx += row_size
                    row += 1

        # Constant-length columns
        else:
            # Decompress the last block to get its shape, then allocate the result.
            last_entries = decomp(entry_blocks.pop())
            last_num_rows = last_entries.shape[0]
            num_rows = len(entry_blocks) * \
                header.entries_per_block + last_num_rows
            num_subcolumns = header.elements_per_entry if indices is None else len(
                indices)
            result = np.zeros((num_rows, num_subcolumns), header.dtype)

            row = 0
            for raw_entry in entry_blocks:
                entries = decomp(raw_entry)
                additional_rows = entries.shape[0]
                result[row: (row + additional_rows)] = entries
                row += additional_rows

            result[row: (row + last_num_rows)] = last_entries

        # Squeeze if flag is set to True
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
