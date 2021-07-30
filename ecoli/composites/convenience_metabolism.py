"""
Composite model with Metabolism and Convenience Kinetics
"""
import numpy as np
from vivarium.core.composer import Composer
from vivarium.core.engine import pp, Engine
from vivarium.library.dict_utils import deep_merge

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import AA_MEDIA_ID
from ecoli.states.wcecoli_state import get_state_from_file

from ecoli.processes.metabolism import Metabolism
from ecoli.processes.convenience_kinetics import ConvenienceKinetics
from ecoli.processes.exchange_stub import Exchange

from vivarium.processes.growth_rate import GrowthRate


SIM_DATA_PATH = '/home/santiagomille/Desktop/vivarium-ecoli/reconstruction/sim_data/kb/simData_4.cPickle'


class ConvenienceMetabolism(Composer):
    defaults = {
        'convenience_kinetics': {},
        'metabolism': {
            'media_id': AA_MEDIA_ID,
            'use_trna_charging': True
        },
        'aa': True,
        'time_step': 1.0,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
    }

    def __init__(self, config=None):
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

    def initial_state(self, config=None, aa=False, initial_time=0):
        state = get_state_from_file(aa = aa)
        return state

    def generate_processes(self, config):
        time_step = config['time_step']

        # get the parameters
        metabolism_config = self.load_sim_data.get_metabolism_config(time_step = time_step, aa = config['aa'])
        metabolism_config = deep_merge(metabolism_config, config['metabolism'])

        convenience_kinetics_config = self.load_sim_data.get_convenience_kinetics_config(time_step=time_step)
        convenience_kinetics_config = deep_merge(convenience_kinetics_config, config['convenience_kinetics'])
        # make the processes

        growth_rate_config = {
            'variables': ['cell_mass', 'dry_mass'],
            'default_growth_noise': 0.00075,
            'default_growth_rate':  0.00555,
        }

        exchange_stub_config = {'exchanges': {
            'L-ALPHA-ALANINE[c]': -0.03403124877865654,
            'ARG[c]': -0.017753836578163255,
            'ASN[c]': -0.01429771440815284,
            'L-ASPARTATE[c]': -0.014929808202838315,
            'CYS[c]': -0.002552888024986289,
            'GLT[c]': -0.01120188293384097,
            'GLN[c]': -0.01179696847906752,
            'GLY[c]': -0.02701082747638407,
            'HIS[c]': -0.0066707297417078065,
            'ILE[c]': -0.018673042468834125,
            'LEU[c]': -0.03087184106850655,
            'LYS[c]': -0.026150530917024482,
            'MET[c]': -0.008305096921050314,
            'PHE[c]': -0.011617958494822556,
            'PRO[c]': -0.012687892464619566,
            'SER[c]': -0.016576827590343523,
            'THR[c]': -0.021396261902734137,
            'TRP[c]': 0.00013911767221120886,
            'TYR[c]': -0.009107424784777024,
            'L-SELENOCYSTEINE[c]': -1.1707177140693039e-05,
            'VAL[c]': -0.02839286830361277
        }
        }
        for mol, rate in exchange_stub_config['exchanges'].items():
            exchange_stub_config['exchanges'][mol] = rate/2

        return {
            'convenience_kinetics': ConvenienceKinetics(convenience_kinetics_config),
            'metabolism': Metabolism(metabolism_config),
            'exchange_stub': Exchange(exchange_stub_config),
            'growth_rate': GrowthRate(growth_rate_config)
        }

    def generate_topology(self, config):
        topology = {
            'metabolism': {
                'metabolites': ('bulk',),
                'catalysts': ('bulk',),
                'kinetics_enzymes': ('bulk',),
                'kinetics_substrates': ('bulk',),
                'amino_acids': ('bulk',),
                'listeners': ('listeners',),
                'environment': ('environment',),
                'polypeptide_elongation': ('process_state', 'polypeptide_elongation'),
                'exchange_constraints': ('fluxes',),
                'conc_diff': ('export',),
            },

            'convenience_kinetics': {
                'external': ('environment',),
                'fluxes': ('fluxes',),
                'listeners': ('listeners',),
                'global': ('global',),
                'internal':('bulk',),
            },

            'exchange_stub': {
                'molecules': ('bulk',),
                'listeners': ('listeners',),
                'export': ('export',),
            },

            'growth_rate': {
                'variables': ('listeners', 'mass'),
                'rates': ('rates',)
            }
        }
        return topology


def test_convenience_metabolism(
        total_time=100,
        progress_bar=True,
        aa = True
):
    composer = ConvenienceMetabolism()

    # get initial state
    initial_state = composer.initial_state(aa = aa)

    # generate the composite
    ecoli = composer.generate()

    # make the experiment
    ecoli_simulation = Engine({
        'processes': ecoli.processes,
        'topology': ecoli.topology,
        'initial_state': initial_state,
        'progress_bar': progress_bar,
    })

    # run the experiment
    ecoli_simulation.update(total_time)

    # retrieve the data
    output = ecoli_simulation.emitter.get_timeseries()

    import matplotlib.pyplot as plt

    rows = 6
    cols = 4
    fig = plt.figure(figsize=(8, 11.5))

    time = output['time']

    for plot_index, aa in enumerate(output['export'].keys()):
        ax = plt.subplot(rows, cols, plot_index + 1)

        # Get actual fluxes
        bulk = np.array([b/m for b,m in zip(output['bulk'][str(aa)], output['listeners']['mass']['dry_mass'])])
        bulk /= bulk[2]
        export = np.array([-e / m for e, m in zip(output['export'][str(aa)], output['listeners']['mass']['dry_mass'])])
        export /= export[2]
        tag='[p]'
        if 'L-SELE' in aa:
            tag = '[c]'
        aa_flux = np.array([(-1 * b) / m for b, m in zip(output['environment']['exchange'][str(aa)[: -3]+tag], output['listeners']['mass']['dry_mass'])])
        aa_flux /= aa_flux[2]

        fluxes = [b / m for b, m in zip(output['fluxes']['EX_'+str(aa)], output['listeners']['enzyme_kinetics']['countsToMolar'])]
        fluxes = np.array([f/m for f,m in zip(fluxes, output['listeners']['mass']['dry_mass'])])
        fluxes /= fluxes[2]
        # Plot, orange is target flux and blue is actual flux
        ax.plot(time, aa_flux, linewidth=1, label='Uptake')
        ax.plot(time, bulk, linewidth=1, label='Bulk')
        ax.plot(time, fluxes, linewidth=1, label='uptake2')
        ax.plot(time, export, linewidth=1, label='export')
        ax.set_xlabel("Time (min)", fontsize=6)
        ax.set_ylabel("counts/gDCW", fontsize=6)
        ax.set_title("%s" % aa, fontsize=6, y=1.1)
        ax.tick_params(which="both", direction="out", labelsize=6)

    plt.rc("font", size=6)
    plt.suptitle("External exchange fluxes of amino acids", fontsize=10)
    plt.savefig('metabolism.png', dpi=300)

    plt.close("all")



if __name__ == "__main__":
    test_convenience_metabolism()
