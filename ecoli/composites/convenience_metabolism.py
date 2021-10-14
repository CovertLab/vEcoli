"""
======================
Convenience Metabolism
======================

Composite model with Metabolism and Convenience Kinetics
"""
import os
import argparse

import numpy as np
from vivarium.core.composer import Composer
from vivarium.core.engine import pp, Engine
from vivarium.library.dict_utils import deep_merge
from vivarium.processes.growth_rate import GrowthRate
from vivarium.library.units import units
from vivarium.plots.topology import plot_topology

from ecoli.processes.metabolism import Metabolism
from ecoli.processes.convenience_kinetics import ConvenienceKinetics
from ecoli.processes.stubs.exchange_stub import Exchange
from ecoli.processes.local_field import LocalField
from ecoli.processes.nonspatial_environment import NonSpatialEnvironment
from ecoli.library.sim_data import LoadSimData
from ecoli.states.wcecoli_state import get_state_from_file

from ecoli.plots.topology import get_ecoli_master_topology_settings

SIM_DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..', '..', 'reconstruction',
        'sim_data_wcEcoli-cffc309', 'kb', 'simData.cPickle',
    )
)



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
                'location': [50, 50],
                'dimensions': {
                    'bounds': [100,100],
                    'n_bins': [1,1]
                }
            },
            'nonspatial': True,
            'bin_volume': 50 * units.fL,
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
        fields_initial = {
            'location': [50, 50],
            'dimensions': {
                'bounds': [100,100],
                'n_bins': [1,1],
            }
        }
        state = deep_merge(state, fields_initial)
        return state

    def generate_processes(self, config):
        time_step = config['time_step']

        metabolism_config = self.load_sim_data.get_metabolism_config(time_step=time_step, aa=config['aa'])
        metabolism_config = deep_merge(metabolism_config, config['metabolism'])

        convenience_kinetics_config = self.load_sim_data.get_convenience_kinetics_config(time_step=time_step)
        convenience_kinetics_config = deep_merge(convenience_kinetics_config, config['convenience_kinetics'])

        growth_rate_config = {
            'variables': ['cell_mass', 'dry_mass'],
            'default_growth_noise': 0.0000015,
            'default_growth_rate':  0.0005,
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
                'external': ('local_environment',),
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
        total_time=1000,
        progress_bar=True,
        aa=True
):
    config = {
        'fields_on': False, 
        'aa': True,
        'metabolism': {
            'media_id': 'minimal_plus_amino_acids',
            'use_trna_charging': True
        },
        'time_step': 2.0,
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
        total_time=500,
        progress_bar=True,
        aa=True,
        out_dir=''
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
    
    # configure the environment [mol/L]
    environment_config = {
        "volume": 50 * units.fL,
        "concentrations": {
            "L-ALPHA-ALANINE[p]": 4.0/1000,
            "ARG[p]": 26.0/1000,
            "ASN[p]": 2.0/1000,
            "L-ASPARTATE[p]": 2.0/1000,
            "CYS[p]": 0.5/1000,
            "GLT[p]": 3.0/1000,
            "GLN[p]": 3.1/1000,
            "GLY[p]": 4.0/1000,
            "HIS[p]": 1.0/1000,
            "ILE[p]": 2.0/1000,
            "LEU[p]": 4.0/1000,
            "LYS[p]": 2.0/1000,
            "MET[p]": 1.0/1000,
            "PHE[p]": 2.0/1000,
            "PRO[p]": 2.0/1000,
            "SER[p]": 50.0/1000,
            "THR[p]": 2.0/1000,
            "TRP[p]": 0.5/1000,
            "TYR[p]": 1.0/1000,
            "VAL[p]": 3.0/1000
        }
    }

    environment_process = NonSpatialEnvironment(environment_config)
    ecoli.processes.update({
        environment_process.name: environment_process})

    # add topology
    environment_topology = environment_process.generate_topology({
        'topology': {
            'external': ('local_environment',),
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
    plot_output(output)
    import ipdb; ipdb.set_trace()
    return output


def run_convenience_metabolism():
    output = test_convenience_metabolism()
    plot_output(output)


def plot_output(output):
    import matplotlib.pyplot as plt

    rows = 6
    cols = 4
    fig = plt.figure(figsize=(8, 11.5))
    aa_no_in_media = 'L-SELENOCYSTEINE'

    time = output['time']

    for plot_index, aa in enumerate(output['export'].keys()):
        ax = plt.subplot(rows, cols, plot_index + 1)

        # Change in bulk concentrations
        raw_bulk = output['bulk'][str(aa)]
        bulk = [0]
        for i, v in enumerate(raw_bulk[1:]):
            bulk.append(v - raw_bulk[i])

        bulk = np.array([b/m for b, m in zip(bulk, output['listeners']['mass']['dry_mass'])])
        bulk /= bulk[4]
        
        # Change in AA used in translation/tRNA/etc.
        export = np.array([-e / m for e, m in zip(output['export'][str(aa)], output['listeners']['mass']['dry_mass'])])
        export /= export[2]

        # Change in imported amino acid fluxes (FBA)
        tag='[p]'
        if aa_no_in_media in aa:
            tag = '[c]'
        aa_flux = np.array([b/m for b, m in zip(output['environment']['exchange'][aa[:-3]+tag], output['listeners']['mass']['dry_mass'])])
        aa_flux /= aa_flux[3]
        
        # Change in imported aa fluxes (Convinience Kinetics)
        fluxes = [b / m for b, m in zip(output['fluxes']['EX_'+str(aa)], output['listeners']['enzyme_kinetics']['countsToMolar'])]
        fluxes = np.array([f/m for f,m in zip(fluxes, output['listeners']['mass']['dry_mass'])])
        fluxes /= fluxes[2]

        # Fields change
        if aa_no_in_media in aa:
            fields = np.ones(len(time))
        else:
            fields = np.array([f[0][0] for f in output['fields'][aa[:-3]+tag]])
        fields /= fields[0]

        # Plot, orange is target flux and blue is actual flux
        ax.plot(time, aa_flux, linewidth=1, label='Uptake - FBA', color='blue', alpha=0.8)
        ax.plot(time, bulk, linewidth=1, label='Bulk', color='orange', alpha=0.8)
        ax.plot(time, fluxes, linewidth=1, label='Uptake - CK', color='red',  alpha=0.8)
        ax.plot(time, export, linewidth=1, label='Used in Translation', color='green', alpha=0.8)
        ax.plot(time, fields, linewidth=1, label='Change in fields', color='pink', alpha=0.8)
        ax.set_xlabel("Time (min)", fontsize=4, labelpad=-0.5)
        ax.set_ylabel("counts/gDCW", fontsize=4, labelpad=-0.5)
        ax.set_title("%s" % aa, fontsize=5.5, y=1., pad=2.)
        ax.tick_params(which="both", direction="in", labelsize=4)
        
        handles, labels = ax.get_legend_handles_labels()
    
    fig.legend(handles, labels, loc='lower center')
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
        test_library[name](out_dir=out_dir)


