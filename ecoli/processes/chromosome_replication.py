"""Chromosome Replication

Process for chromosome replication. Performs initiation, elongation, and
termination of active partial chromosomes that replicate the chromosome.
"""

import uuid
import numpy as np

from vivarium.core.process import Process
from vivarium.library.dict_utils import deep_merge

from ecoli.library.schema import array_to, array_from, arrays_from, arrays_to, bulk_schema

from wholecell.utils import units
from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease


class ChromosomeReplication(Process):

    name = 'ecoli-chromosome_replication'

    defaults = {
        'max_time_step': 2.0,
        'get_dna_critical_mass': lambda doubling_time: units.Unum,
        'criticalInitiationMass': 975 * units.fg,
        'nutrientToDoublingTime': {},
        'replichore_lengths': np.array([]),
        'sequences': np.array([]),
        'polymerized_dntp_weights': [],
        'replication_coordinate': np.array([]),
        'D_period': np.array([]),
        'no_child_place_holder': -1,
        'basal_elongation_rate': 967,
        'make_elongation_rates': lambda random, replisomes, base, time_step: units.Unum,
        'mechanistic_replisome': True,

        # molecules
        'replisome_trimers_subunits': [],
        'replisome_monomers_subunits': [],
        'dntps': [],
        'ppi': [],

        # random seed
        'seed': 0,
        
        # partitioning flags
        'request_only': False,
        'evolve_only': False
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.max_time_step = self.parameters['max_time_step']

        # Load parameters
        self.get_dna_critical_mass = self.parameters['get_dna_critical_mass']
        self.criticalInitiationMass = self.parameters['criticalInitiationMass']
        self.nutrientToDoublingTime = self.parameters['nutrientToDoublingTime']
        self.replichore_lengths = self.parameters['replichore_lengths']
        self.sequences = self.parameters['sequences']
        self.polymerized_dntp_weights = self.parameters['polymerized_dntp_weights']
        self.replication_coordinate = self.parameters['replication_coordinate']
        self.D_period = self.parameters['D_period']
        self.no_child_place_holder = self.parameters['no_child_place_holder']
        self.basal_elongation_rate = self.parameters['basal_elongation_rate']
        self.make_elongation_rates = self.parameters['make_elongation_rates']

        # Sim options
        self.mechanistic_replisome = self.parameters['mechanistic_replisome']

        # random state
        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed=self.seed)

        self.emit_unique = self.parameters.get('emit_unique', True)

        self.request_only = self.parameters['request_only']
        self.evolve_only = self.parameters['evolve_only']

    def ports_schema(self):

        default_unique_schema = {
            '_default': 0, '_updater': 'set', '_emit': self.emit_unique}
        return {
            # bulk molecules
            'replisome_trimers': bulk_schema(self.parameters['replisome_trimers_subunits']),
            'replisome_monomers': bulk_schema(self.parameters['replisome_monomers_subunits']),
            'dntps': bulk_schema(self.parameters['dntps']),
            'ppi': bulk_schema(self.parameters['ppi']),
            'listeners': {
                'mass': {
                    'cell_mass': {'_default': 0.0, '_emit': True}},
                'replication_data': {
                    'criticalInitiationMass': {'_default': 0.0},
                    'criticalMassPerOriC': {'_default': 0.0},
                }
            },
            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'},
                },
            # unique molecules
            'active_replisomes': {
                '*': {
                    'domain_index': default_unique_schema,
                    'right_replichore': default_unique_schema,
                    'coordinates': default_unique_schema,
                    'dna_mass': {
                            '_default': 0, '_updater': 'accumulate'},
                }},
            'oriCs': {
                '*': {
                    'domain_index': default_unique_schema,
                }},
            'chromosome_domains': {
                '*': {
                    attr: default_unique_schema
                    for attr in ('domain_index', 'child_domains')
                }},
            'full_chromosomes': {
                '*': {
                    'domain_index': default_unique_schema,
                    'division_time': default_unique_schema,
                    'has_triggered_division': {
                        '_default': False, '_updater': 'set'},
                }}}

    def calculate_request(self, timestep, states):
        requests = {}
        # Get total count of existing oriC's
        n_oriC = len(states['oriCs'])
		# If there are no origins, return immediately
        if n_oriC == 0:
            return requests
        
        # Get current cell mass
        cellMass = (states['listeners']['mass']['cell_mass'] * units.fg)

        # Get critical initiation mass for current simulation environment
        current_media_id = states['environment']['media_id']
        self.criticalInitiationMass = self.get_dna_critical_mass(
            self.nutrientToDoublingTime[current_media_id])

        # Calculate mass per origin of replication, and compare to critical
        # initiation mass. If the cell mass has reached this critical mass,
        # the process will initiate a round of chromosome replication for each
        # origin of replication.
        massPerOrigin = cellMass / n_oriC
        self.criticalMassPerOriC = massPerOrigin / self.criticalInitiationMass

        # If replication should be initiated, request subunits required for
        # building two replisomes per one origin of replication, and edit
        # access to oriC and chromosome domain attributes
        if self.criticalMassPerOriC >= 1.0:
            requests['replisome_trimers'] = {rep_trimer: 6*n_oriC 
                                   for rep_trimer in states['replisome_trimers']}
            requests['replisome_monomers'] = {rep_monomer: 2*n_oriC for rep_monomer 
                                        in states['replisome_monomers']}

        # If there are no active forks return
        n_active_replisomes = len(states['active_replisomes'])
        if n_active_replisomes == 0:
            return requests

        # Get current locations of all replication forks
        fork_coordinates = arrays_from(
                states['active_replisomes'].values(),
                ['coordinates'])
        sequence_length = np.abs(np.repeat(fork_coordinates, 2))

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            len(self.sequences),
            self.basal_elongation_rate,
            timestep)

        sequences = buildSequences(
            self.sequences,
            np.tile(np.arange(4), n_active_replisomes//2),
            sequence_length,
            self.elongation_rates)

        # Count number of each dNTP in sequences for the next timestep
        sequenceComposition = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE], minlength=4)

        # If one dNTP is limiting then limit the request for the other three by
        # the same ratio
        dNtpsTotal = array_from(states['dntps'])
        maxFractionalReactionLimit = (np.fmin(1, dNtpsTotal / sequenceComposition)).min()

        # Request dNTPs
        requests['dntps'] = array_to(states['dntps'], maxFractionalReactionLimit
            * sequenceComposition)

        return requests
        
    def evolve_state(self, timestep, states):
        # Initialize the update dictionary
        update = {
            'replisome_trimers': {
                    mol: 0
                    for mol in self.parameters['replisome_trimers_subunits']},
            'replisome_monomers': {
                    mol: 0
                    for mol in self.parameters['replisome_monomers_subunits']},
            # 'oriCs': {},
            'active_replisomes': {},
            'listeners': {
                'replication_data': {},
            }}
        
        
        ## Module 1: Replication initiation
        # # Get number of existing replisomes and oriCs
        # n_active_replisomes = self.active_replisomes.total_count()
        # n_oriC = self.oriCs.total_count()
        n_active_replisomes = len(states['active_replisomes'])
        n_oriC = len(states['oriCs'])

        # If there are no origins, return immediately
        if n_oriC == 0:
            return update

        # Get attributes of existing chromosome domains
        domain_index_existing_domain, child_domains = arrays_from(
            states['chromosome_domains'].values(),
            ['domain_index', 'child_domains'])
        # domain_index_existing_domain, child_domains = self.chromosome_domains.attrs(
        #     'domain_index', 'child_domains')

        # # Get number of available replisome subunits
        # n_replisome_trimers = self.replisome_trimers.counts()
        # n_replisome_monomers = self.replisome_monomers.counts()
        
        # Initiate replication only when
        # 1) The cell has reached the critical mass per oriC
        # 2) If mechanistic replisome option is on, there are enough replisome
        # subunits to assemble two replisomes per existing OriC.
        # Note that we assume asynchronous initiation does not happen.
        initiate_replication = False
        if self.criticalMassPerOriC >= 1.0:
            n_replisome_trimers = array_from(states['replisome_trimers'])
            n_replisome_monomers = array_from(states['replisome_monomers'])
            initiate_replication = (not self.mechanistic_replisome or 
                                    (np.all(n_replisome_trimers == 6 * n_oriC) and
                                    np.all(n_replisome_monomers == 2 * n_oriC)))

        # If all conditions are met, initiate a round of replication on every
        # origin of replication
        if initiate_replication:
            # Get attributes of existing oriCs and domains
            domain_index_existing_oric, = arrays_from(
                states['oriCs'].values(),
                ['domain_index'])
            # domain_index_existing_oric = self.oriCs.attr('domain_index')

            # Get indexes of the domains that would be getting child domains
            # (domains that contain an origin)
            new_parent_domains = np.where(np.in1d(domain_index_existing_domain,
                                                  domain_index_existing_oric))[0]

            # Calculate counts of new replisomes and domains to add
            n_new_replisome = 2 * n_oriC
            n_new_domain = 2 * n_oriC

            # Calculate the domain indexes of new domains and oriC's
            max_domain_index = domain_index_existing_domain.max()
            domain_index_new = np.arange(
                max_domain_index + 1, max_domain_index + 2 * n_oriC + 1,
                dtype=np.int32)

            # Add new oriC's, and reset attributes of existing oriC's
            # All oriC's must be assigned new domain indexes
            update['oriCs'] = {
                '_add': [{
                    'key': str(uuid.uuid1()),
                    'state': {'domain_index': domain_index_new[index]}}
                    for index in range(n_oriC)],
                '_delete': [(key,) for key in states['oriCs'].keys()]}
            # self.oriCs.attrIs(domain_index=domain_index_new[:n_oriC])
            # self.oriCs.moleculesNew(
            #     n_oriC, domain_index=domain_index_new[n_oriC:])

            # Add and set attributes of newly created replisomes.
            # New replisomes inherit the domain indexes of the oriC's they
            # were initiated from. Two replisomes are formed per oriC, one on
            # the right replichore, and one on the left.
            coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
            right_replichore = np.tile(
                np.array([True, False], dtype=np.bool), n_oriC)
            domain_index_new_replisome = np.repeat(
                domain_index_existing_oric, 2)

            update['active_replisomes']['_add'] = [{
                    'key': str(uuid.uuid1()),
                    'state': {
                        'coordinates': coordinates_replisome[index],
                        'right_replichore': right_replichore[index],
                        'domain_index': domain_index_new_replisome[index],
                    }}
                    for index in range(n_new_replisome)]
            # self.active_replisomes.moleculesNew(
            #     n_new_replisome,
            #     coordinates=coordinates_replisome,
            #     right_replichore=right_replichore,
            #     domain_index=domain_index_new_replisome)

            # Add and set attributes of new chromosome domains. All new domains
            # should have have no children domains.
            new_child_domains = np.full(
                (n_new_domain, 2), self.no_child_place_holder, dtype=np.int32)
            new_domains_update = {
                '_add': [{
                    'key': str(uuid.uuid1()),
                    'state': {
                        'domain_index': domain_index_new[index].tolist(),
                        'child_domains': new_child_domains[index].tolist(),
                    }}
                    for index in range(n_new_domain)]}

            # Add new domains as children of existing domains
            child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)
            existing_domains_update = {
                domain: {'child_domains': child_domains[index].tolist()}
                for index, domain in enumerate(states['chromosome_domains'].keys())}
            update['chromosome_domains'] = {**new_domains_update, **existing_domains_update}

            # self.chromosome_domains.moleculesNew(
            #     n_new_domain,
            #     domain_index=domain_index_new,
            #     child_domains=np.full(
            #         (n_new_domain, 2), self.no_child_place_holder,
            #         dtype=np.int32))
            # child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)
            # self.chromosome_domains.attrIs(child_domains=child_domains)

            # Decrement counts of replisome subunits
            if self.mechanistic_replisome:
                for mol in self.parameters['replisome_trimers_subunits']:
                    update['replisome_trimers'][mol] -= 6 * n_oriC
                for mol in self.parameters['replisome_monomers_subunits']:
                    update['replisome_monomers'][mol] -= 2 * n_oriC

                # self.replisome_trimers.countsDec(6 * n_oriC)
                # self.replisome_monomers.countsDec(2 * n_oriC)


        # Write data from this module to a listener
        update['listeners']['replication_data']['criticalMassPerOriC'] = \
            self.criticalMassPerOriC
        update['listeners']['replication_data']['criticalInitiationMass'] = \
            self.criticalInitiationMass.asNumber(units.fg)
        # self.writeToListener("ReplicationData", "criticalMassPerOriC",
        #                      self.criticalMassPerOriC)
        # self.writeToListener("ReplicationData", "criticalInitiationMass",
        #                      self.criticalInitiationMass.asNumber(units.fg))

        ## Module 2: replication elongation
        # If no active replisomes are present, return immediately
        # Note: the new replication forks added in the previous module are not
        # elongated until the next timestep.
        if n_active_replisomes == 0:
            return update

        # Get allocated counts of dNTPs
        dNtpCounts = array_from(states['dntps'])
        # dNtpCounts = self.dntps.counts()

        # Get attributes of existing replisomes
        domain_index_replisome, right_replichore, coordinates_replisome, = arrays_from(
            states['active_replisomes'].values(),
            ['domain_index', 'right_replichore', 'coordinates'])
        # domain_index_replisome, right_replichore, coordinates_replisome = self.active_replisomes.attrs(
        #     "domain_index", "right_replichore", "coordinates")
        sequence_length = np.abs(np.repeat(coordinates_replisome, 2))

        # Build sequences to polymerize
        sequence_indexes = np.tile(np.arange(4), n_active_replisomes // 2)

        sequences = buildSequences(
            self.sequences,
            sequence_indexes,
            sequence_length,
            self.elongation_rates)

        # Use polymerize algorithm to quickly calculate the number of
        # elongations each fork catalyzes
        reactionLimit = dNtpCounts.sum()

        active_elongation_rates = self.elongation_rates[sequence_indexes]

        result = polymerize(
            sequences,
            dNtpCounts,
            reactionLimit,
            self.random_state,
            active_elongation_rates)

        sequenceElongations = result.sequenceElongation
        dNtpsUsed = result.monomerUsages

        # Compute mass increase for each elongated sequence
        mass_increase_dna = computeMassIncrease(
            sequences,
            sequenceElongations,
            self.polymerized_dntp_weights.asNumber(units.fg))

        # Compute masses that should be added to each replisome
        added_dna_mass = mass_increase_dna[0::2] + mass_increase_dna[1::2]

        # Update positions of each fork
        updated_length = sequence_length + sequenceElongations
        updated_coordinates = updated_length[0::2]

        # Reverse signs of fork coordinates on left replichore
        updated_coordinates[~right_replichore] = -updated_coordinates[~right_replichore]

        # Update attributes and submasses of replisomes
        active_replisomes_indexes = list(states['active_replisomes'].keys())
        active_replisomes_update = arrays_to(
            len(states['active_replisomes']),
            {
                'coordinates': updated_coordinates,
                'dna_mass': added_dna_mass,
             })
        update['active_replisomes'] = {
                active_replisomes_indexes[index]: active_replisomes
                for index, active_replisomes in enumerate(active_replisomes_update)}
        # self.active_replisomes.attrIs(coordinates=updated_coordinates)
        # self.active_replisomes.add_submass_by_name("DNA", added_dna_mass)

        # Update counts of polymerized metabolites
        update['dntps'] = array_to(self.parameters['dntps'], -dNtpsUsed)
        update['ppi'] = array_to(self.parameters['ppi'], [dNtpsUsed.sum()])
        # self.dntps.countsDec(dNtpsUsed)
        # self.ppi.countInc(dNtpsUsed.sum())

        ## Module 3: replication termination
        # Determine if any forks have reached the end of their sequences. If
        # so, delete the replisomes and domains that were terminated.
        terminal_lengths = self.replichore_lengths[
            np.logical_not(right_replichore).astype(np.int64)]
        terminated_replisomes = (np.abs(updated_coordinates) == terminal_lengths)

        # If any forks were terminated,
        if terminated_replisomes.sum() > 0:
            # Get domain indexes of terminated forks
            terminated_domains = np.unique(domain_index_replisome[terminated_replisomes])

            # Get attributes of existing domains and full chromosomes
            domain_index_domains, child_domains, = arrays_from(
                states['chromosome_domains'].values(),
                ['domain_index', 'child_domains'])
            domain_index_full_chroms, = arrays_from(
                states['full_chromosomes'].values(),
                ['domain_index'])
            # domain_index_domains, child_domains = self.chromosome_domains.attrs(
            #     "domain_index", "child_domains")
            # domain_index_full_chroms = self.full_chromosomes.attr("domain_index")

            # Initialize array of replisomes that should be deleted
            replisomes_to_delete = np.zeros_like(domain_index_replisome, dtype=np.bool)

            # Count number of new full chromosomes that should be created
            n_new_chromosomes = 0

            # Initialize array for domain indexes of new full chromosomes
            domain_index_new_full_chroms = []

            for terminated_domain_index in terminated_domains:
                # Get all terminated replisomes in the terminated domain
                terminated_domain_matching_replisomes = np.logical_and(
                    domain_index_replisome == terminated_domain_index,
                    terminated_replisomes)

                # If both replisomes in the domain have terminated, we are
                # ready to split the chromosome and update the attributes.
                if terminated_domain_matching_replisomes.sum() == 2:
                    # Tag replisomes and domains with the given domain index
                    # for deletion
                    replisomes_to_delete = np.logical_or(
                        replisomes_to_delete,
                        terminated_domain_matching_replisomes)

                    domain_mask = (
                            domain_index_domains == terminated_domain_index)

                    # Get child domains of deleted domain
                    child_domains_this_domain = child_domains[
                                                np.where(domain_mask)[0][0], :]

                    # Modify domain index of one existing full chromosome to
                    # index of first child domain
                    domain_index_full_chroms[
                        np.where(domain_index_full_chroms == terminated_domain_index)[0]
                    ] = child_domains_this_domain[0]

                    # Increment count of new full chromosome
                    n_new_chromosomes += 1

                    # Append chromosome index of new full chromosome
                    domain_index_new_full_chroms.append(child_domains_this_domain[1])

            # Delete terminated replisomes
            replisome_delete_update = [
                (key,) for index, key in enumerate(states['active_replisomes'].keys())
                if replisomes_to_delete[index]]
            if replisome_delete_update:
                update['active_replisomes']['_delete'] = replisome_delete_update
            # self.active_replisomes.delByIndexes(np.where(replisomes_to_delete)[0])

            # Generate new full chromosome molecules
            if n_new_chromosomes > 0:

                chromosome_add_update = {
                    '_add': [{
                        'key': str(uuid.uuid1()),
                        'state': {
                        'domain_index': domain_index_new_full_chroms[index],
                        'division_time': self.D_period,  #self.time() + self.D_period, TODO -- how is division_time used?
                        'has_triggered_division': False}}
                    for index in range(n_new_chromosomes)]}
                # self.full_chromosomes.moleculesNew(
                #     n_new_chromosomes,
                #     division_time=[self.time() + self.D_period] * n_new_chromosomes,
                #     has_triggered_division=[False] * n_new_chromosomes,
                #     domain_index=domain_index_new_full_chroms)

                # Reset domain index of existing chromosomes that have finished
                # replication
                chromosome_existing_update = {
                    key: {'domain_index': domain_index_full_chroms[index]}
                    for index, key in enumerate(states['full_chromosomes'].keys())}
                # self.full_chromosomes.attrIs(
                #     domain_index=domain_index_full_chroms)

                update['full_chromosomes'] = {**chromosome_add_update, **chromosome_existing_update}

            # Increment counts of replisome subunits
            if self.mechanistic_replisome:
                for mol in self.parameters['replisome_trimers_subunits']:
                    update['replisome_trimers'][mol] += 3 * replisomes_to_delete.sum()
                for mol in self.parameters['replisome_monomers_subunits']:
                    update['replisome_monomers'][mol] += replisomes_to_delete.sum()

                # update['replisome_trimers'] = 3 * replisomes_to_delete.sum()
                # update['replisome_monomers'] = replisomes_to_delete.sum()
                # self.replisome_trimers.countsInc(3 * replisomes_to_delete.sum())
                # self.replisome_monomers.countsInc(replisomes_to_delete.sum())

        return update

    def next_update(self, timestep, states):
        if self.request_only:
            update = self.calculate_request(timestep, states)
        elif self.evolve_only:
            update = self.evolve_state(timestep, states)
        else:
            requests = self.calculate_request(timestep, states)
            states = deep_merge(states, requests)
            update = self.evolve_state(timestep, states)
        return update
    

def test_chromosome_replication():
    test_config = {}
    process = ChromosomeReplication(test_config)


if __name__ == "__main__":
    test_chromosome_replication()
