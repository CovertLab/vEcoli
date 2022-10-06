import numpy as np
from scipy.constants import N_A
from vivarium.core.process import Step
from vivarium.library.units import units

from ecoli.library.schema import dict_value_schema


AVOGADRO = N_A / units.mol
#: The ribosomes in our initial state from wcEcoli have these keys, but
#: we don't use them in vivarium-ecoli.
RIBOSOME_KEYS_TO_REMOVE = ('_globalIndex', '_entryState')
# Calculated from Dong, Nilsson, and Kurland (1996). DOI:
# 10.1006/jmbi.1996.0428. See Chris's 2022-04-05-2 lab notebook entry
# for calculation details.
TRNAS_PER_RIBOSOME = 0.38953800000000005


class TetracyclineRibosomeEquilibrium(Step):

    name = 'tetracycline-ribosome-equilibrium'
    defaults = {
        # K_eq for tetracycline binding ribosome.
        # Source: Epe, B., & Woolley, P. (1984). The binding of
        # 6-demethylchlortetracycline to 70S, 50S and 30S ribosomal
        # particles: A quantitative study by fluorescence anisotropy.
        # The EMBO Journal, 3(1), 121–126.
        'K_Tc': 1e6,
        # K_eq for tRNA binding ribosome.
        # Source: Holschuh, K., Riesner, D., & Gassen, H. G. (1981).
        # Steps of mRNA translocation in protein biosynthesis. Nature,
        # 293(5834), 675–677. https://doi.org/10.1038/293675a0.
        'K_tRNA': 1e6,
        'seed': 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.random_state = np.random.RandomState(
            seed=self.parameters['seed'])

    def ports_schema(self):
        return {
            'tetracycline': {
                '_default': 0 * units.mM,
                '_emit': True,
            },
            '30s-free': {
                '_default': 0,
            },
            '50s': {
                '_default': 0,
            },
            '70s-free': dict_value_schema('active_ribosome'),
            '70s-tetracycline': dict_value_schema('active_ribosome'),
            '30s-tetracycline': {
                '_default': 0,
            },
            # Cytoplasm volume.
            'volume': {
                '_default': 0 * units.fL,
            },
            'evolvers_ran': {
                '_default': True,
            },
            'listeners': {
                'total_internal_tetracycline': {
                    '_default': 0 * units.mM,
                    '_updater': 'set',
                    '_emit': True,
                },
                'frac_ribosomes_bound_tetracycline': {
                    '_default': 0,
                    '_updater': 'set',
                    '_emit': True,
                },
            },
        }

    def update_condition(self, timestep, states):
        return states['evolvers_ran']

    def next_update(self, _, states):
        volume = states['volume']

        count_70s_free = len(states['70s-free'])
        conc_70s_free = count_70s_free / AVOGADRO / volume
        count_70s_tc = len(states['70s-tetracycline'])
        conc_70s_tc = count_70s_tc / AVOGADRO / volume
        count_70s_total = count_70s_free + count_70s_tc

        count_30s_free = states['30s-free']
        conc_30s_free = count_30s_free / AVOGADRO / volume
        count_30s_tc = states['30s-tetracycline']
        conc_30s_tc = count_30s_tc / AVOGADRO / volume
        count_30s_total = count_30s_free + count_30s_tc

        conc_ribo_total = (
            conc_70s_free + conc_70s_tc + conc_30s_free + conc_30s_tc)
        count_ribo_total = (
            count_70s_free + count_70s_tc + count_30s_free +
            count_30s_tc)
        conc_trna = TRNAS_PER_RIBOSOME * conc_ribo_total

        conc_tc_free = states['tetracycline']
        count_tc_free = states['tetracycline'] * AVOGADRO * volume

        conc_tc_total = conc_tc_free + conc_30s_tc + conc_70s_tc
        count_tc_total = count_tc_free + count_30s_tc + count_70s_tc

        if not conc_tc_total:
            # If there is no tetracycline, do nothing.
            return {}

        # The ratio between the concentration of ribosomes (30S or 70S)
        # bound to tetracycline (Tc) and the concentration bound to
        # tRNA.
        tc_trna_binding_ratio = (
            self.parameters['K_Tc'] * conc_tc_free
            / self.parameters['K_tRNA'] / conc_trna
        )
        # Why this works: Let r be the binding ratio, tc be the
        # tetracycline-bound concentration, and tr be the tRNA-bound
        # concentration. Note that this means that r = tc / tr. We want
        # to find tc / (tc + tr), which we can calculate as r / (1 + r).
        # Here's the proof:
        #
        #     r / (1 + r) = (tc / tr) / (1 + tc / tr)
        #                 = (tc / tr) / ((tr + tc) / tr)
        #                 = tc / (tr + tc)
        #
        # Note that here we are assuming that all ribosomes are bound by
        # either tetracycline or tRNA. This assumption is reasonable
        # because the binding constants are so large (10^6) compared to
        # the number of 30s and 70s ribosomes available (~16,000 total).
        ribo_frac_tc_bound = tc_trna_binding_ratio / (
            1 + tc_trna_binding_ratio)
        count_ribo_total_tc_target = min(
            (ribo_frac_tc_bound * count_ribo_total).to(
                units.count).magnitude,
            # Leave one free tetracycline molecule to account for
            # rounding errors when converting between counts and
            # concentrations.
            max(0, count_tc_total - 1),
        )

        # Divide Tc-bound ribosomes between 30s and 70s proportionally.
        count_70s_tc_target = int(count_ribo_total_tc_target * (
            count_70s_total / count_ribo_total))
        count_30s_tc_target = (
            count_ribo_total_tc_target - count_70s_tc_target)

        # For reporting, find the resulting fraction of ribosomes bound
        # to tetracycline.
        ribo_frac_tc_bound_expected = (count_70s_tc_target +
            count_30s_tc_target) / count_ribo_total

        # Determine what needs to be done.
        count_70s_to_inhibit = max(
            int(count_70s_tc_target - count_70s_tc),
            0,  # We inactivate bound ribosomes, so can't un-bind them.
        )
        count_30s_to_inhibit = int(count_30s_tc_target - count_30s_tc)
        count_ribos_to_inhibit = (
            count_70s_to_inhibit + count_30s_to_inhibit)
        delta_tc = -count_ribos_to_inhibit / AVOGADRO / volume

        assert delta_tc + conc_tc_free >= 0

        # Handle the easy updates: the concentrations and subunits.
        update = {
            'tetracycline': {
                '_value': delta_tc,
                '_updater': 'accumulate',
            },
            '30s-free': {
                '_value': -count_30s_to_inhibit,
                '_updater': 'accumulate',
            },
            '30s-tetracycline': {
                '_value': count_30s_to_inhibit,
                '_updater': 'accumulate',
            },
            'listeners': {
                'total_internal_tetracycline': conc_tc_total,
                'frac_ribosomes_bound_tetracycline': (
                    ribo_frac_tc_bound_expected),
            },
        }

        # Handle the more difficult updates: the active ribosomes.
        if count_70s_to_inhibit > 0:
            # Randomly select which active ribosomes to inhibit.
            ids_70s_to_inhibit = self.random_state.choice(
                np.array(list(states['70s-free'])),
                count_70s_to_inhibit,
                replace=False,
            )
            # Assume that when ribosomes are inhibited, they also become
            # inactive.
            update.update({
                '70s-free': {
                    '_delete': list(ids_70s_to_inhibit),
                },
                '50s': {
                    '_value': len(ids_70s_to_inhibit),
                    '_updater': 'accumulate',
                },
            })
            update['30s-tetracycline']['_value'] += len(
                ids_70s_to_inhibit)
        return update
