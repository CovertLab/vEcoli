"""
Submodel for RNA degradation.

Mathematical formulation:

dr/dt = Kb - Kd * r
or,

dr/dt = Kb - kcatEndoRNase * EndoRNase * r / (Km + r)
or,

dr/dt = Kb - kcatEndoRNase * EndoRNase * r/Km / (1 + Sum(r/Km))

    where   r = RNA counts
            Kb = RNA production given a RNAP synthesis rate 
            tau = doubling time
            kcatEndoRNase = enzymatic activity for EndoRNases
            kd = RNA degradation rates 
            Km = Michaelis-Menten constants fitted to recapitulate first-order
            RNA decay:
                kd * r = kcatEndoRNase * EndoRNase * r / (Km + r),
                    non-cooperative EndoRNases
                kd * r = kcatEndoRNase * EndoRNase * r/Km / (1 + sum(r/Km)),
                    cooperation

This sub-model encodes molecular simulation of RNA degradation as two main
steps guided by RNases, "endonucleolytic cleavage" and "exonucleolytic
digestion":

1. Compute total counts of RNA to be degraded (D) and total capacity for
endo-cleavage (C) at each time point
2. Evaluate C and D. If C > D, then define a fraction of active endoRNases 
3. Dissect RNA degraded into different species (mRNA, tRNA, and rRNA) by
accounting endoRNases specificity
4. Update RNA fragments (assumption: fragments are represented as a pool of
nucleotides) created because of endonucleolytic cleavage
5. Compute total capacity of exoRNases and determine fraction of nucleotides
that can be digested
6. Update pool of metabolites (H and H2O) created because of exonucleolytic
digestion
"""

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process
from vivarium.library.dict_utils import deep_merge

from ecoli.library.schema import (
    array_from, array_to, arrays_from, arrays_to, listener_schema, bulk_schema, array_to_nonzero)

from wholecell.utils.constants import REQUEST_PRIORITY_DEGRADATION
from wholecell.utils import units
from six.moves import range, zip

class RnaDegradation(Process):
    name = 'ecoli-rna-degradation'

    defaults = {
        'rnaIds': [],
        'n_avogadro': 0.0,
        'cell_density': 1100 * units.g / units.L,
        'endoRnaseIds': [],
        'exoRnaseIds': [],
        'KcatExoRNase': np.array([]) / units.s,
        'KcatEndoRNases': np.array([]) / units.s,
        'charged_trna_names': [],
        'rnaDegRates': [],
        'shuffle_indexes': None,
        'is_mRNA': np.array([]),
        'is_rRNA': np.array([]),
        'is_tRNA': np.array([]),
        'rna_lengths': np.array([]),
        'polymerized_ntp_ids': [],
        'water_id': 'h2o',
        'ppi_id': 'ppi',
        'proton_id': 'h+',
        'counts_ACGU': np.array([[]]),
        'nmp_ids': [],
        'rrfaIdx': 0,
        'rrlaIdx': 0,
        'rrsaIdx': 0,
        'Km': np.array([]) * units.mol / units.L,
        'EndoRNaseCoop': 1,
        'EndoRNaseFunc': 1,
        'ribosome30S': 'ribosome30S',
        'ribosome50S': 'ribosome50S',
        'seed': 0}
    
    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.rnaIds = self.parameters['rnaIds']
        self.n_total_RNAs = len(self.rnaIds)

        # Load constants
        self.n_avogadro = self.parameters['n_avogadro']
        self.cell_density = self.parameters['cell_density']

        # Load RNase kinetic data
        self.endoRnaseIds = self.parameters['endoRnaseIds']
        self.exoRnaseIds = self.parameters['exoRnaseIds']
        self.KcatExoRNase = self.parameters['KcatExoRNase']
        self.KcatEndoRNases = self.parameters['KcatEndoRNases']

        # Load information about charged tRNA
        self.charged_trna_names = self.parameters['charged_trna_names']

        # Load first-order RNA degradation rates (estimated by mRNA half-life data)
        self.rnaDegRates = self.parameters['rnaDegRates']

        self.shuffle_indexes = self.parameters['shuffle_indexes']
        if self.shuffle_indexes:
            self.rnaDegRates = self.rnaDegRates[self.shuffle_indexes]

        self.is_mRNA = self.parameters['is_mRNA']
        self.is_rRNA = self.parameters['is_rRNA']
        self.is_tRNA = self.parameters['is_tRNA']

        self.rna_lengths = self.parameters['rna_lengths']

        # Build stoichiometric matrix
        self.polymerized_ntp_ids = self.parameters['polymerized_ntp_ids']
        self.nmp_ids = self.parameters['nmp_ids']
        self.water_id = self.parameters['water_id']
        self.ppi_id = self.parameters['ppi_id']
        self.proton_id = self.parameters['proton_id']
        self.counts_ACGU = self.parameters['counts_ACGU']

        self.endCleavageMetaboliteIds = self.polymerized_ntp_ids + [
            self.water_id, self.ppi_id, self.proton_id]
        nmpIdxs = list(range(4))
        h2oIdx = self.endCleavageMetaboliteIds.index(self.water_id)
        ppiIdx = self.endCleavageMetaboliteIds.index(self.ppi_id)
        hIdx = self.endCleavageMetaboliteIds.index(self.proton_id)
        self.endoDegradationSMatrix = np.zeros((len(self.endCleavageMetaboliteIds), self.n_total_RNAs), np.int64)
        self.endoDegradationSMatrix[nmpIdxs, :] = self.counts_ACGU
        self.endoDegradationSMatrix[h2oIdx, :] = 0
        self.endoDegradationSMatrix[ppiIdx, :] = 1
        self.endoDegradationSMatrix[hIdx, :] = 0

        self.ribosome30S = self.parameters['ribosome30S']
        self.ribosome50S = self.parameters['ribosome50S']
        self.rrfaIdx = self.parameters['rrfaIdx']
        self.rrlaIdx = self.parameters['rrlaIdx']
        self.rrsaIdx = self.parameters['rrsaIdx']

        # Load Michaelis-Menten constants fitted to recapitulate first-order RNA decay model
        self.Km = self.parameters['Km']

        # If set to True, assume cooperation in endoRNase activity
        self.EndoRNaseCoop = self.parameters['EndoRNaseCoop']

        # If set to False, assume RNAs degrade simply by first-order kinetics
        self.EndoRNaseFunc = self.parameters['EndoRNaseFunc']

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)
        
    def ports_schema(self):
        return {
            'charged_trna': bulk_schema(self.charged_trna_names),
            'bulk_RNAs': bulk_schema(self.rnaIds, updater='nonnegative_accumulate'),
            'nmps': bulk_schema(self.nmp_ids),
            'fragmentMetabolites': bulk_schema(self.endCleavageMetaboliteIds),
            'fragmentBases': bulk_schema(self.polymerized_ntp_ids),
            'endoRnases': bulk_schema(self.endoRnaseIds),
            'exoRnases': bulk_schema(self.exoRnaseIds),
            'subunits': bulk_schema([self.ribosome30S, self.ribosome50S]),
            'molecules': bulk_schema([self.water_id, self.proton_id]),
            'active_ribosome': {
                '*': {
                    'unique_index': {'_default': 0}}},
            'RNAs': {
                '*': {
                    'unique_index': {'_default': 0, '_updater': 'set'},
                    'TU_index': {'_default': 0, '_updater': 'set'},
                    'is_full_transcript': {'_default': 0, '_updater': 'set'},
                    'can_translate': {'_default': 0, '_updater': 'set'}}},
            'listeners': {
                'mass': {
                    'cell_mass': {'_default': 0.0},
                    'dry_mass': {'_default': 0.0}},
                'rna_degradation_listener': listener_schema({
                    'fraction_active_endo_rnases': 0,
                    'diff_relative_first_order_decay': 0,
                    'fract_endo_rrna_counts': 0,
                    'count_rna_degraded': 0,
                    'nucleotides_from_degradation': 0,
                    'fragment_bases_digested': 0})}}
            
    def calculate_request(self, timestep, states):
        # Compute factor that convert counts into concentration, and vice versa
        cell_mass = states['listeners']['mass']['cell_mass'] * units.fg
        cell_volume = cell_mass / self.cell_density
        counts_to_molar = 1 / (self.n_avogadro * cell_volume)

        # Get total counts of RNAs including rRNAs, charged tRNAs, and active
        # (translatable) unique mRNAs
        bulk_RNA_counts = array_from(states['bulk_RNAs'])
        bulk_RNA_counts[self.rrsaIdx] += states['subunits'][self.ribosome30S]
        bulk_RNA_counts[[self.rrlaIdx, self.rrfaIdx]] += states['subunits'][
            self.ribosome50S]
        bulk_RNA_counts[[self.rrlaIdx, self.rrfaIdx, self.rrsaIdx]] \
            += len(states['active_ribosome'])
        bulk_RNA_counts[self.is_tRNA.astype(np.bool)] \
            += array_from(states['charged_trna'])

        TU_index, can_translate, is_full_transcript = arrays_from(
            states['RNAs'].values(), 
            ['TU_index', 'can_translate', 'is_full_transcript'])
         
        TU_index_translatable_mRNAs = TU_index[can_translate]
        unique_RNA_counts = np.bincount(
            TU_index_translatable_mRNAs, minlength=self.n_total_RNAs)
        total_RNA_counts = bulk_RNA_counts + unique_RNA_counts

        # Compute RNA concentrations
        rna_conc_molar = counts_to_molar * total_RNA_counts

        # Get counts of endoRNases
        endornase_counts = array_from(states['endoRnases'])
        total_kcat_endornase = units.dot(self.KcatEndoRNases, endornase_counts)

        # Calculate the fraction of active endoRNases for each RNA based on
        # Michaelis-Menten kinetics
        if self.EndoRNaseCoop:
            frac_endornase_saturated = (
                rna_conc_molar / self.Km / (1 + units.sum(rna_conc_molar / self.Km))
            ).asNumber()
        else:
            frac_endornase_saturated = (
                rna_conc_molar / (self.Km + rna_conc_molar)
            ).asNumber()

        # Calculate difference in degradation rates from first-order decay
        # and the number of EndoRNases per one molecule of RNA
        total_endornase_counts = np.sum(endornase_counts)
        diff_relative_first_order_decay = units.sum(
            units.abs(self.rnaDegRates * total_RNA_counts -
                total_kcat_endornase * frac_endornase_saturated)
            )
        endornase_per_rna = total_endornase_counts / np.sum(total_RNA_counts)
        
        requests = {'listeners': {}}
        requests['listeners'].update({"rna_degradation_listener": {
                                    "fraction_active_endo_rnases": 
                                        np.sum(frac_endornase_saturated)}})
        requests['listeners'].update({"rna_degradation_listener": {
                                    "diff_relative_first_oder_decay": 
                                        diff_relative_first_order_decay.asNumber()}})
        requests['listeners'].update({"rna_degradation_listener": {
                                    "fract_endo_rrna_counts": 
                                        endornase_per_rna}})

        if self.EndoRNaseFunc:
            # Dissect RNAse specificity into mRNA, tRNA, and rRNA
            mrna_specificity = np.dot(frac_endornase_saturated, self.is_mRNA)
            trna_specificity = np.dot(frac_endornase_saturated, self.is_tRNA)
            rrna_specificity = np.dot(frac_endornase_saturated, self.is_rRNA)

            n_total_mrnas_to_degrade = self._calculate_total_n_to_degrade(
                timestep,
                mrna_specificity,
                total_kcat_endornase)
            n_total_trnas_to_degrade = self._calculate_total_n_to_degrade(
                timestep,
                trna_specificity,
                total_kcat_endornase)
            n_total_rrnas_to_degrade = self._calculate_total_n_to_degrade(
                timestep,
                rrna_specificity,
                total_kcat_endornase)

            # Compute RNAse specificity
            rna_specificity = frac_endornase_saturated / np.sum(frac_endornase_saturated)

            # Boolean variable that tracks existence of each RNA
            rna_exists = (total_RNA_counts > 0).astype(np.int64)

            # Compute degradation probabilities of each RNA: for mRNAs, this
            # is based on the specificity of each mRNA. For tRNAs and rRNAs,
            # this is distributed evenly.
            mrna_deg_probs = 1. / np.dot(rna_specificity, self.is_mRNA * rna_exists) * rna_specificity * self.is_mRNA * rna_exists
            trna_deg_probs = 1. / np.dot(self.is_tRNA, rna_exists) * self.is_tRNA * rna_exists
            rrna_deg_probs = 1. / np.dot(self.is_rRNA, rna_exists) * self.is_rRNA * rna_exists

            # Mask RNA counts into each class of RNAs
            mrna_counts = total_RNA_counts * self.is_mRNA
            trna_counts = total_RNA_counts * self.is_tRNA
            rrna_counts = total_RNA_counts * self.is_rRNA

            # Determine number of individual RNAs to be degraded for each class
            # of RNA.
            n_mrnas_to_degrade = self._get_rnas_to_degrade(
                n_total_mrnas_to_degrade, mrna_deg_probs, mrna_counts)

            n_trnas_to_degrade = self._get_rnas_to_degrade(
                n_total_trnas_to_degrade, trna_deg_probs, trna_counts)

            n_rrnas_to_degrade = self._get_rnas_to_degrade(
                n_total_rrnas_to_degrade, rrna_deg_probs, rrna_counts)

            n_RNAs_to_degrade = n_mrnas_to_degrade + n_trnas_to_degrade + n_rrnas_to_degrade

        # First order decay with non-functional EndoRNase activity 
        # Determine mRNAs to be degraded by sampling a Poisson distribution
        # (Kdeg * RNA)
        else:
            n_RNAs_to_degrade = np.fmin(
                self.random_state.poisson(
                    (self.rnaDegRates * total_RNA_counts).asNumber()
                    ),
                total_RNA_counts
                )

        # Bulk RNAs (tRNAs and rRNAs) are degraded immediately. Unique RNAs
        # (mRNAs) are immediately deactivated (becomes unable to bind
        # ribosomes), but not degraded until transcription is finished and the
        # mRNA becomes a full transcript to simplify the transcript elongation
        # process.
        n_bulk_RNAs_to_degrade = n_RNAs_to_degrade.copy()
        n_bulk_RNAs_to_degrade[self.is_mRNA] = 0
        self.n_unique_RNAs_to_deactivate = n_RNAs_to_degrade.copy()
        self.n_unique_RNAs_to_deactivate[np.logical_not(self.is_mRNA)] = 0
        
        requests['bulk_RNAs'] = array_to(states['bulk_RNAs'], n_bulk_RNAs_to_degrade)
        requests['endoRnases'] = states['endoRnases']
        requests['exoRnases'] = states['exoRnases']
        requests['fragmentBases'] = states['fragmentBases']

        # Calculate the amount of water required for total RNA hydrolysis by
        # endo and exonucleases. We first calculate the number of unique RNAs
        # that should be degraded at this timestep.
        self.unique_mRNAs_to_degrade = np.logical_and(
            np.logical_not(can_translate), is_full_transcript)
        self.n_unique_RNAs_to_degrade = np.bincount(
            TU_index[self.unique_mRNAs_to_degrade],
            minlength=self.n_total_RNAs)

        # Assuming complete hydrolysis for now. Note that one additional water
        # molecule is needed for each RNA to hydrolyze the 5' diphosphate.
        waterForNewRnas = np.dot(
            n_bulk_RNAs_to_degrade + self.n_unique_RNAs_to_degrade,
            self.rna_lengths)
        waterForLeftOverFragments = array_from(states['fragmentBases']).sum()
        requests[self.water_id] = waterForNewRnas + waterForLeftOverFragments
        return requests
        
    def evolve_state(self, timestep, states):
        ## wcEcoli evolveState
        # Get vector of numbers of RNAs to degrade for each RNA species
        n_degraded_bulk_RNA = array_from(states['bulk_RNAs'])
        n_degraded_unique_RNA = self.n_unique_RNAs_to_degrade
        n_degraded_RNA = n_degraded_bulk_RNA + n_degraded_unique_RNA
        
        # Deactivate and degrade unique RNAs
        TU_index, can_translate = arrays_from(
            states['RNAs'].values(),
            ['TU_index', 'can_translate'])
        n_deactivated_unique_RNA = self.n_unique_RNAs_to_deactivate

        # Deactive unique RNAs
        non_zero_deactivation = (n_deactivated_unique_RNA > 0)

        for index, n_degraded in zip(
                np.arange(n_deactivated_unique_RNA.size)[non_zero_deactivation],
                n_deactivated_unique_RNA[non_zero_deactivation]):
            # Get mask for translatable mRNAs belonging to the degraded species
            mask = np.logical_and(TU_index == index, can_translate)

            # Choose n_degraded indexes randomly to deactivate
            can_translate[self.random_state.choice(
                size=n_degraded, a=np.where(mask)[0], replace=False)] = False

        rnas_indexes = list(states['RNAs'].keys())
        rnas_update = arrays_to(
            len(states['RNAs']), {
            'can_translate': can_translate})

        update = {
            'listeners': {
                'rna_degradation_listener': {
                    'count_rna_degraded': n_degraded_RNA,
                    'nucleotides_from_degradation': np.dot(n_degraded_RNA, self.rna_lengths)}},
            # Degrade bulk RNAs
            'bulk_RNAs': array_to_nonzero(
                self.rnaIds,
                -n_degraded_bulk_RNA),
            'RNAs': {
                rna_index: rnas_update[index]
                for index, rna_index in enumerate(states['RNAs'].keys())}}

        # Degrade full mRNAs that are inactive
        update['RNAs']['_delete'] = [
            (rnas_indexes[delete_index],)
            for delete_index in np.where(self.unique_mRNAs_to_degrade)[0]]

        # Modeling assumption: Once a RNA is cleaved by an endonuclease its
        # resulting nucleotides are lumped together as "polymerized fragments".
        # These fragments can carry over from previous timesteps. We are also
        # assuming that during endonucleolytic cleavage the 5'terminal
        # phosphate is removed. This is modeled as all of the fragments being
        # one long linear chain of "fragment bases".

        # Example:
        # PPi-Base-PO4(-)-Base-PO4(-)-Base-OH
        # ==>
        # Pi-FragmentBase-PO4(-)-FragmentBase-PO4(-)-FragmentBase + PPi
        # Note: Lack of -OH on 3' end of chain
        metabolitesEndoCleavage = np.dot(
            self.endoDegradationSMatrix, n_degraded_RNA)

        # Increase polymerized fragment counts
        update['fragmentMetabolites'] = array_to(
            self.endCleavageMetaboliteIds,
            metabolitesEndoCleavage)

        # Check if exonucleolytic digestion can happen 
        fragmentBases = array_from(states['fragmentBases'])
        if fragmentBases.sum() == 0:
            return update

        # Calculate exolytic cleavage events

        # Modeling assumption: We model fragments as one long fragment chain of
        # polymerized nucleotides. We are also assuming that there is no
        # sequence specificity or bias towards which nucleotides are
        # hydrolyzed.

        # Example:
        # Pi-FragmentBase-PO4(-)-FragmentBase-PO4(-)-FragmentBase + 3 H2O
        # ==>
        # 3 NMP + 3 H(+)
        # Note: Lack of -OH on 3' end of chain

        n_exoRNases = array_from(states['exoRnases'])
        n_fragment_bases = fragmentBases
        n_fragment_bases_sum = n_fragment_bases.sum()

        exornase_capacity = n_exoRNases.sum() * self.KcatExoRNase * (
                units.s * timestep)

        if exornase_capacity >= n_fragment_bases_sum:
            update['nmps'] = array_to(self.nmp_ids, n_fragment_bases)
            update['molecules'] = {
                self.water_id: -n_fragment_bases_sum,
                self.proton_id: n_fragment_bases_sum}
            update['fragmentBases'] = array_to(self.polymerized_ntp_ids, -fragmentBases)
            total_fragment_bases_digested = n_fragment_bases_sum

        else:
            fragmentSpecificity = n_fragment_bases / n_fragment_bases_sum
            possibleBasesToDigest = self.random_state.multinomial(
                exornase_capacity, fragmentSpecificity)
            n_fragment_bases_digested = n_fragment_bases - np.fmax(
                n_fragment_bases - possibleBasesToDigest, 0)

            total_fragment_bases_digested = n_fragment_bases_digested.sum()

            update['nmps'] = array_to(self.nmp_ids, n_fragment_bases_digested)
            update['molecules'] = {
                self.water_id: -total_fragment_bases_digested,
                self.proton_id: total_fragment_bases_digested}
            update['fragmentBases'] = array_to(self.polymerized_ntp_ids, -n_fragment_bases_digested)

        update['listeners']['rna_degradation_listener']['fragment_bases_digested'] = total_fragment_bases_digested

        return update

    def next_update(self, timestep, states):
        requests = self.calculate_request(timestep, states)
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        update['listeners'].update(requests['listeners'])
        return update

    def _calculate_total_n_to_degrade(self, timestep, specificity, total_kcat_endornase):
        """
        Calculate the total number of RNAs to degrade for a specific class of
        RNAs, based on the specificity of endoRNases on that specific class and
        the total kcat value of the endoRNases.

        Args:
            specificity: Sum of fraction of active endoRNases for all RNAs
            in a given class
            total_kcat_endornase: The summed kcat of all existing endoRNases
        Returns:
            Total number of RNAs to degrade for the given class of RNAs
        """
        return np.round(
            (specificity * total_kcat_endornase
             * (units.s * timestep)).asNumber()
            )


    def _get_rnas_to_degrade(self, n_total_rnas_to_degrade, rna_deg_probs,
            rna_counts):
        """
        Distributes the total count of RNAs to degrade for each class of RNAs
        into individual RNAs, based on the given degradation probabilities
        of individual RNAs. The upper bound is set by the current count of the
        specific RNA.

        Args:
            n_total_rnas_to_degrade: Total number of RNAs to degrade for the
            given class of RNAs (integer, scalar)
            rna_deg_probs: Degradation probabilities of each RNA (vector of
            equal length to the total number of different RNAs)
            rna_counts: Current counts of each RNA molecule (vector of equal
            length to the total number of different RNAs)
        Returns:
            Vector of equal length to rna_counts, specifying the number of
            molecules to degrade for each RNA
        """
        n_rnas_to_degrade = np.zeros_like(rna_counts)
        remaining_rna_counts = rna_counts

        while n_rnas_to_degrade.sum() < n_total_rnas_to_degrade and remaining_rna_counts.sum() != 0:
            n_rnas_to_degrade += np.fmin(
                self.random_state.multinomial(
                    n_total_rnas_to_degrade - n_rnas_to_degrade.sum(),
                    rna_deg_probs
                    ),
                remaining_rna_counts
                )
            remaining_rna_counts = rna_counts - n_rnas_to_degrade

        return n_rnas_to_degrade
