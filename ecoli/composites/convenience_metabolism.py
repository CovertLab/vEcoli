"""
Composite model with Metabolism and Convenience Kinetics
"""
import os
import argparse

import numpy as np
from vivarium.core.composer import Composer
from vivarium.core.engine import pp, Engine
from vivarium.library.dict_utils import deep_merge
from ecoli.processes.nonspatial_environment import NonSpatialEnvironment

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import AA_MEDIA_ID
from ecoli.states.wcecoli_state import get_state_from_file

from ecoli.processes.metabolism import Metabolism
from ecoli.processes.convenience_kinetics import ConvenienceKinetics
from ecoli.processes.exchange_stub import Exchange
from ecoli.processes.local_field import LocalField

from vivarium.processes.growth_rate import GrowthRate
from vivarium.library.units import units

from ecoli.plots.topology import get_ecoli_master_topology_settings
from vivarium.plots.topology import plot_topology


SIM_DATA_PATH = '/home/santiagomille/Desktop/vivarium-ecoli/reconstruction/sim_data/kb/simData.cPickle'


class ConvenienceMetabolism(Composer):
    defaults = {
        'convenience_kinetics': {},
        'metabolism': {
            'media_id': 'minimal',
            'use_trna_charging': False
        },
        'aa': False,
        'time_step': 2.0,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'fields_on': False,
        'field_deriver': {
            'initial_external': {
                'exchanges': {
                    'A': 20
                },
                'location': [0.5, 0.5],
                'fields': {
                    'A': np.ones((1, 1), dtype=np.float64)
                },
                'dimensions': {
                    'bounds': [1,1],
                    'n_bins': [1,1],
                    'depth': 1,
                }
            },
            'nonspatial': True,
            'bin_volume': 0.050 * units.L,
        }
    }

    def __init__(self, config=None):
        config = deep_merge(self.defaults, config)
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

    def initial_state(self, config=None, initial_time=0):
        state = get_state_from_file(aa = config['aa'])
        return state

    def generate_processes(self, config):
        time_step = config['time_step']

        metabolism_config = self.load_sim_data.get_metabolism_config(time_step = time_step, aa = config['aa'])
        metabolism_config = deep_merge(metabolism_config, config['metabolism'])

        convenience_kinetics_config = self.load_sim_data.get_convenience_kinetics_config(time_step=time_step)
        convenience_kinetics_config = deep_merge(convenience_kinetics_config, config['convenience_kinetics'])

        growth_rate_config = {
            'variables': ['cell_mass', 'dry_mass'],
            'default_growth_noise': 0.00055,
            'default_growth_rate':  0.00105,
        }

        # mmolar/s 
        exchange_stub_config = {
            'exchanges': {
                'L-ALPHA-ALANINE[c]': -0.11891556567246558,
                'ARG[c]': -7.401981971305775e-02,
                'ASN[c]': -4.6276520958890666e-02,
                'L-ASPARTATE[c]': -6.134566727946733e-02,
                'CYS[c]': -7.859120955223963e-03,
                'GLT[c]': -7.431882467718336e-02,
                'GLN[c]': -4.182943019840645e-02,
                'GLY[c]': -8.661320475286021e-02,
                'HIS[c]': -2.0571058098381066e-02,
                'ILE[c]': -6.468540757057489e-02,
                'LEU[c]': -9.168525399920554e-02,
                'LYS[c]': -0.16771317383763713,
                'MET[c]': -2.810545691942153e-02,
                'PHE[c]': -3.6612722850633775e-02,
                'PRO[c]': -3.975203946118958e-02,
                'SER[c]': -5.7714042266331883e-02,
                'THR[c]': -6.261960148640104e-02,
                'TRP[c]': -3.186528021336261e-02,
                'TYR[c]': -3.25238094669149e-02,
                'L-SELENOCYSTEINE[c]': -2.0531845573797217e-8,
                'VAL[c]': -8.51282850382732e-02,
            }
        }

        processes = {
            'convenience_kinetics': ConvenienceKinetics(convenience_kinetics_config),
            'metabolism': Metabolism(metabolism_config),
            'exchange_stub': Exchange(exchange_stub_config),
            'growth_rate': GrowthRate(growth_rate_config)
        }

        # TODO -- plug in local field if you want to have environment exchnages applied to change external concentrations
        if config['fields_on']:
            fields_config = config['field_deriver']
            processes.update({
                'field_deriver': LocalField(fields_config)
            })

        return processes

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
                'external': ('environment',), #check this
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
                'variables': {
                    '_path': ('listeners', 'mass'),
                    # 'transporter_id': ('..', '..', 'bulk', 'transporter_id')  # connect a variable manually
                },
                'rates': ('rates',)
            }
        }
        if config['fields_on']:
            topology.update({
                'field_deriver': {
                    'exchanges': ('environment', 'exchange',),
                    'location': ('global', 'location',),
                    'fields': ('fields',),
                    'dimensions': ('dimensions',),
                    }
            })
        return topology


def test_convenience_metabolism(
        total_time=100,
        progress_bar=True,
        aa = True
):
    config = {
        'fields_on': False, 
        'aa': True,
        'metabolism': {
            'media_id': 'minimal_plus_amino_acids',
            'use_trna_charging': True
        },
        'time_step': 1.0,
    }

    composer = ConvenienceMetabolism(config)

    # get initial state
    initial_state = composer.initial_state(config=config)

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
    import ipdb; ipdb.set_trace()
    return output


def run_in_environment(
        total_time=100,
        progress_bar=True,
        aa=True
):
    config = {
        'fields_on': True, 
        'aa': True,
        'metabolism': {
            'media_id': 'minimal_plus_amino_acids',
            'use_trna_charging': True
        },
        'time_step': 1.0,
    }

    composer = ConvenienceMetabolism(config)

    # get initial state
    initial_state = composer.initial_state(config=config)

    # generate the composite
    ecoli = composer.generate()
    
    # configure the environment
    environment_config = {
        "volume": 0.050 * units.L,
        "concentrations": { # mol/L
           "2-3-DIHYDROXYBENZOATE[c]":0.000138,
           "2-KETOGLUTARATE[c]":0.000353,
           "2-PG[c]":9.18e-05,
           "2K-4CH3-PENTANOATE[c]":0.000138,
           "4-AMINO-BUTYRATE[c]":0.000304,
           "4-hydroxybenzoate[c]":5.22e-05,
           "ACETOACETYL-COA[c]":2.18e-05,
           "ACETYL-COA[c]":0.001058,
           "ACETYL-P[c]":0.00107,
           "ADENINE[c]":5.225e-06,
           "ADENOSINE[c]":1.31e-07,
           "ADP-D-GLUCOSE[c]":4.27e-06,
           "ADP[c]":0.0004956666666666667,
           "AMP[c]":0.00017585,
           "ANTHRANILATE[c]":3.48e-06,
           "APS[c]":6.63e-06,
           "ARG[c]":0.0005733333333333334,
           "ASN[c]":0.0008629999999999999,
           "ATP[c]":0.00963,
           "BIOTIN[c]":2.0549850734573224e-05,
           "CA+2[c]":0.002580604099410439,
           "CAMP[c]":1.339e-05,
           "CARBAMYUL-L-ASPARTATE[c]":0.00059,
           "CARBON-DIOXIDE[c]":7.52e-05,
           "CDP[c]":3.07e-05,
           "CHORISMATE[c]":0.00010183250713119245,
           "CIS-ACONITATE[c]":1.3049999999999999e-05,
           "CIT[c]":0.0018349999999999998,
           "CL-[c]":0.0025807328784961917,
           "CMP[c]":0.000574,
           "CO+2[c]":1.2508227422015942e-05,
           "CO-A[c]":0.0007375,
           "CPD-12115[c]":0.00010183428219093554,
           "CPD-12261[p]":0.0018815522242916387,
           "CPD-12575[c]":0.0025,
           "CPD-12819[c]":0.0419472129328885,
           "CPD-12824[c]":0.0027960412767205713,
           "CPD-13469[c]":0.00115,
           "CPD-2961[c]":0.002077,
           "CPD-8260[c]":0.010066442281334775,
           "CPD-9956[c]":0.00010183425364623143,
           "CPD0-939[c]":0.0036543621940603238,
           "CTP[c]":0.0017959999999999999,
           "CYS[c]":4.34e-05,
           "CYTIDINE[c]":1.8400000000000002e-06,
           "CYTOSINE[c]":9.475e-06,
           "D-ALA-D-ALA[c]":0.000103,
           "D-SEDOHEPTULOSE-7-P[c]":0.0003786666666666667,
           "DAMP[c]":8.84e-06,
           "DATP[c]":1.55e-05,
           "DCTP[c]":3.45e-05,
           "DEOXY-RIBOSE-5P[c]":0.000303,
           "DEOXYADENOSINE[c]":2.82e-06,
           "DEOXYGUANOSINE[c]":5.22e-07,
           "DGMP[c]":5.07e-05,
           "DGTP[c]":1.55e-05,
           "DI-H-OROTATE[c]":0.0019459500000000001,
           "DIHYDROXY-ACETONE-PHOSPHATE[c]":0.0018859999999999999,
           "DPG[c]":1.575e-05,
           "ENTEROBACTIN[c]":0.0001018331114857719,
           "ERYTHROSE-4P[c]":4.87e-05,
           "FAD[c]":0.000173,
           "FE+2[c]":0.0033289844989973167,
           "FMN[c]":5.37e-05,
           "FRUCTOSE-16-DIPHOSPHATE[c]":0.006373333333333332,
           "FRUCTOSE-6P[c]":0.0014190000000000001,
           "FUM[c]":0.000749,
           "G3P[c]":0.00154,
           "GDP[c]":0.00041663333333333334,
           "GLC-6-P[c]":0.0044335,
           "GLN[c]":0.00804,
           "GLT[c]":0.0498,
           "GLUCONATE[c]":4.16e-05,
           "GLUTATHIONE[c]":0.01176,
           "GLYCERATE[c]":0.00141,
           "GLYCEROL-3P[c]":0.00017449999999999999,
           "GLY[c]":0.0012469999999999998,
           "GMP[c]":1.761e-05,
           "GTP[c]":0.002135,
           "GUANINE[c]":9.6015e-05,
           "GUANOSINE[c]":1.071e-06,
           "HISTIDINOL[c]":1.175e-05,
           "HIS[c]":0.00014653333333333334,
           "HOMO-CYS[c]":0.00037,
           "HOMO-SER[c]":0.000238,
           "HYPOXANTHINE[c]":7.7e-07,
           "ILE[c]":0.000304,
           "IMP[c]":0.0007708,
           "INOSINE[c]":7e-07,
           "K+[c]":0.09677227144748746,
           "L-ALPHA-ALANINE[c]":0.0022313333333333334,
           "L-ARGININO-SUCCINATE[c]":0.000176,
           "L-ASPARTATE[c]":0.005066,
           "L-CITRULLINE[c]":0.000763,
           "L-ORNITHINE[c]":1.01e-05,
           "L-SELENOCYSTEINE[c]":4.34e-05,
           "LEU[c]":0.000304,
           "LL-DIAMINOPIMELATE[c]":0.000108,
           "LYS[c]":0.0017413333333333332,
           "MALONYL-COA[c]":3.54e-05,
           "MAL[c]":0.0019833333333333335,
           "METHYLENE-THF[c]":9.544070555637273e-05,
           "MET[c]":0.00016233333333333334,
           "MG+2[c]":0.0043010477476229944,
           "MN+2[c]":0.00034291644467901737,
           "N-ACETYL-D-GLUCOSAMINE-1-P[c]":8.19e-05,
           "N-ALPHA-ACETYLORNITHINE[c]":4.33e-05,
           "NADH[c]":5.475e-05,
           "NADPH[c]":0.0001505,
           "NADP[c]":0.00011054000000000001,
           "NAD[c]":0.0023350000000000003,
           "NI+2[c]":0.00015997385165092946,
           "OROTATE[c]":0.00547,
           "OXALACETIC_ACID[c]":4.87e-07,
           "OXIDIZED-GLUTATHIONE[c]":0.0013475000000000002,
           "PANTOTHENATE[c]":4.6450000000000004e-05,
           "PHENYL-PYRUVATE[c]":8.98e-05,
           "PHE[c]":0.0001481,
           "PHOSPHO-ENOL-PYRUVATE[c]":0.00012566666666666667,
           "PPI[c]":0.0005,
           "PROPIONYL-COA[c]":5.32e-06,
           "PROTOHEME[c]":0.00010183358139097685,
           "PROTON[c]":6.30957344480193e-08,
           "PRO[c]":0.00081,
           "PRPP[c]":0.000258,
           "PUTRESCINE[c]":0.01519315196347875,
           "PYRIDOXAL_PHOSPHATE[c]":9.224334550801772e-05,
           "PYRUVATE[c]":0.00366,
           "Pi[c]":0.0005,
           "QUINOLINATE[c]":1.15e-05,
           "REDUCED-MENAQUINONE[c]":0.00010183440689415692,
           "RIBOFLAVIN[c]":1.9e-05,
           "RIBOSE-1P[c]":3e-06,
           "RIBOSE-5P[c]":0.00049,
           "RIBULOSE-5P[c]":0.000228,
           "S-ADENOSYLMETHIONINE[c]":0.000368,
           "SER[c]":0.0016446666666666665,
           "SHIKIMATE[c]":1.41e-05,
           "SIROHEME[c]":0.00010183296736491462,
           "SPERMIDINE[c]":0.003079723994298178,
           "SUC-COA[c]":0.000233,
           "SUC[c]":0.0004716666666666667,
           "TDP[c]":0.000378,
           "THF[c]":9.452740861349238e-05,
           "THIAMINE-PYROPHOSPHATE[c]":9.270169102051287e-05,
           "THREO-DS-ISO-CITRATE[c]":0.00027184999999999997,
           "THR[c]":0.000586,
           "TRP[c]":4.34e-05,
           "TTP[c]":0.00462,
           "TYR[c]":0.00026245,
           "UDP-GLUCURONATE[c]":0.000566,
           "UDP-N-ACETYL-D-GLUCOSAMINE[c]":0.0049555,
           "UDP[c]":0.0009555,
           "UMP[c]":7.09e-05,
           "UNDECAPRENYL-DIPHOSPHATE[c]":2.511605507103185e-05,
           "URIDINE[c]":0.00209,
           "UTP[c]":0.0047150000000000004,
           "VAL[c]":0.00061,
           "WATER[c]":42.74215931168471,
           "XYLULOSE-5-PHOSPHATE[c]":0.0001695,
           "ZN+2[c]":0.00016887830059873836,
           "glycogen-monomer[c]":0.06602839123840608,
           "NI+2[p]":0.0001,
           "CO+2[p]":0.0001,
           "FE+2[p]":0.0001,
           "ZN+2[p]":0.0001,
           "MN+2[p]":0.0001,
           "NA+[p]":0.0001,
           "OXYGEN-MOLECULE[p]":0.0001,
           "CA+2[p]":0.0001,
           "Pi[p]":0.0001,
        }
    }

    environment_process = NonSpatialEnvironment(environment_config)
    ecoli.processes.update({
        environment_process.name: environment_process})

    # add topology
    environment_topology = environment_process.generate_topology({
        'topology': {
            'external': ('environment',),
            'fields': ('fields',),
            'dimensions': ('dimensions',),
        }})[environment_process.name]
    ecoli.topology.update({
        environment_process.name: environment_topology})

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
    return output


def run_convenience_metabolism():
    output = test_convenience_metabolism()
    plot_output(output)


def plot_output(output):
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
        ax.plot(time, aa_flux, linewidth=1, label='Uptake', color='blue')
        ax.plot(time, bulk, linewidth=1, label='Bulk', color='orange')
        ax.plot(time, fluxes, linewidth=1, label='uptake2', color='red')
        ax.plot(time, export, linewidth=1, label='export', color='black')
        ax.set_xlabel("Time (min)", fontsize=6)
        ax.set_ylabel("counts/gDCW", fontsize=6)
        ax.set_title("%s" % aa, fontsize=6, y=1.1)
        ax.tick_params(which="both", direction="out", labelsize=6)

    plt.rc("font", size=6)
    plt.suptitle("External exchange fluxes of amino acids", fontsize=10)
    plt.savefig('metabolism.png', dpi=300)

    plt.close("all")


def ecoli_topology_plot(config={}, filename=None, out_dir=None):
    """Make a topology plot of Ecoli"""
    agent_id_config = {'agent_id': '1'}
    ecoli = ConvenienceMetabolism({**agent_id_config, **config})
    settings = get_ecoli_master_topology_settings()
    topo_plot = plot_topology(
        ecoli,
        filename=filename,
        out_dir=out_dir,
        settings=settings)
    return topo_plot


test_library = {
    '0': run_convenience_metabolism,
    '1': run_in_environment,
}

if __name__ == "__main__":
    out_dir = os.path.join('out', 'ecoli_master')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = argparse.ArgumentParser(description='convenience metabolism')
    parser.add_argument(
        '--name', '-n', default=[], nargs='+', help='test ids to run')
    parser.add_argument(
        '--topology', '-t', action='store_true', default=False,
        help='save a topology plot of ecoli master')
    args = parser.parse_args()

    if args.topology:
        ecoli_topology_plot(filename='ecoli_master', out_dir=out_dir)

    run_all = not args.name

    for name in args.name:
        test_library[name]()


