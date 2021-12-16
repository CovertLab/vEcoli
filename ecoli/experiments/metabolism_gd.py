"""
===========================================
Metabolism using Gradient Descent-based FBA
===========================================
"""
import argparse

# vivarium-core imports
from vivarium.core.engine import Engine
from vivarium.core.composer import Composer
from vivarium.library.dict_utils import deep_merge

# vivarium-ecoli imports
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.library.sim_data import LoadSimData
from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.metabolism_gd import MetabolismGD
from ecoli.processes import Exchange
from ecoli.processes.registries import topology_registry


# get topology from ecoli_master
metabolism_topology = topology_registry.access('ecoli-metabolism')


# make a composite with Exchange
class MetabolismExchange(Composer):
    defaults = {
        'metabolism': {
            'kinetic_rates': [],
        },
        'exchanger': {},
        'sim_data_path': SIM_DATA_PATH,
        'seed': 0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

    def generate_processes(self, config):

        # configure metabolism
        metabolism_config = self.load_sim_data.get_metabolism_gd_config()
        metabolism_config = deep_merge(metabolism_config, config['metabolism'])
        metabolism_process = MetabolismGD(metabolism_config)

        example_update = {'2-3-DIHYDROXYBENZOATE[c]': 0, '2-KETOGLUTARATE[c]': 0, '2-PG[c]': 0, '2K-4CH3-PENTANOATE[c]': 0,
         '4-AMINO-BUTYRATE[c]': 0, '4-hydroxybenzoate[c]': 0, 'ACETOACETYL-COA[c]': 1, 'ACETYL-COA[c]': 2495,
         'ACETYL-P[c]': 0, 'ADENINE[c]': 0, 'ADENOSINE[c]': 0, 'ADP-D-GLUCOSE[c]': 0, 'ADP[c]': 0, 'AMP[c]': 0,
         'ANTHRANILATE[c]': 0, 'APS[c]': 0, 'ARG[c]': 0, 'ASN[c]': 0, 'ATP[c]': 9064, 'BIOTIN[c]': 0, 'CA+2[c]': 0,
         'CA+2[p]': 0, 'CAMP[c]': 1, 'CARBAMYUL-L-ASPARTATE[c]': 0, 'CARBON-DIOXIDE[c]': 0, 'CDP[c]': 0,
         'CHORISMATE[c]': 0, 'CIS-ACONITATE[c]': 0, 'CIT[c]': 0, 'CL-[c]': 0, 'CMP[c]': 0, 'CO+2[c]': 0, 'CO+2[p]': 0,
         'CO-A[c]': 0, 'CPD-12115[c]': 0, 'CPD-12261[p]': 64, 'CPD-12575[c]': 0, 'CPD-12819[c]': 0, 'CPD-12824[c]': 0,
         'CPD-13469[c]': 0, 'CPD-2961[c]': 0, 'CPD-8260[c]': 0, 'CPD-9956[c]': 0, 'CPD0-939[c]': 0, 'CTP[c]': 16292,
         'CYS[c]': 0, 'CYTIDINE[c]': 0, 'CYTOSINE[c]': 0, 'D-ALA-D-ALA[c]': 0, 'D-SEDOHEPTULOSE-7-P[c]': 0,
         'DAMP[c]': 0, 'DATP[c]': 2222, 'DCTP[c]': 1649, 'DEOXY-RIBOSE-5P[c]': 0, 'DEOXYADENOSINE[c]': 0,
         'DEOXYGUANOSINE[c]': 0, 'DGMP[c]': 0, 'DGTP[c]': 1647, 'DI-H-OROTATE[c]': 0,
         'DIHYDROXY-ACETONE-PHOSPHATE[c]': 0, 'DPG[c]': 0, 'ENTEROBACTIN[c]': 109, 'ERYTHROSE-4P[c]': 0, 'FAD[c]': 171,
         'FE+2[c]': 0, 'FE+2[p]': 0, 'FMN[c]': 0, 'FRUCTOSE-16-DIPHOSPHATE[c]': 0, 'FRUCTOSE-6P[c]': 0, 'FUM[c]': 0,
         'G3P[c]': 0, 'GDP[c]': 0, 'GLC-6-P[c]': 0, 'GLN[c]': 0, 'GLT[c]': 0, 'GLUCONATE[c]': 2, 'GLUTATHIONE[c]': 0,
         'GLYCERATE[c]': 0, 'GLYCEROL-3P[c]': 0, 'GLY[c]': 0, 'GMP[c]': 0, 'GTP[c]': 20122, 'GUANINE[c]': 0,
         'GUANOSINE[c]': 0, 'HISTIDINOL[c]': 0, 'HIS[c]': 27, 'HOMO-CYS[c]': 0, 'HOMO-SER[c]': 0, 'HYPOXANTHINE[c]': 0,
         'ILE[c]': 0, 'IMP[c]': 0, 'INOSINE[c]': 0, 'K+[c]': 0, 'L-ALPHA-ALANINE[c]': 0, 'L-ARGININO-SUCCINATE[c]': 0,
         'L-ASPARTATE[c]': 0, 'L-CITRULLINE[c]': 0, 'L-ORNITHINE[c]': 0, 'L-SELENOCYSTEINE[c]': 0, 'LEU[c]': 325,
         'LL-DIAMINOPIMELATE[c]': 0, 'LYS[c]': 0, 'MALONYL-COA[c]': 0, 'MAL[c]': 0, 'METHYLENE-THF[c]': 223,
         'MET[c]': 0, 'MG+2[c]': 0, 'MN+2[c]': 0, 'MN+2[p]': 0, 'N-ACETYL-D-GLUCOSAMINE-1-P[c]': 0,
         'N-ALPHA-ACETYLORNITHINE[c]': 0, 'NA+[p]': 0, 'NADH[c]': 0, 'NADPH[c]': 0, 'NADP[c]': 0, 'NAD[c]': 769,
         'NI+2[c]': 0, 'NI+2[p]': 0, 'OROTATE[c]': 0, 'OXALACETIC_ACID[c]': 0, 'OXIDIZED-GLUTATHIONE[c]': 0,
         'OXYGEN-MOLECULE[p]': 0, 'PANTOTHENATE[c]': 0, 'PHENYL-PYRUVATE[c]': 996, 'PHE[c]': 0,
         'PHOSPHO-ENOL-PYRUVATE[c]': 0, 'PPI[c]': 0, 'PROPIONYL-COA[c]': 0, 'PROTOHEME[c]': 0, 'PROTON[c]': 0,
         'PRO[c]': 0, 'PRPP[c]': 0, 'PUTRESCINE[c]': 0, 'PYRIDOXAL_PHOSPHATE[c]': 0, 'PYRUVATE[c]': 0, 'Pi[c]': 0,
         'Pi[p]': 0, 'QUINOLINATE[c]': 0, 'REDUCED-MENAQUINONE[c]': 240, 'RIBOFLAVIN[c]': 0, 'RIBOSE-1P[c]': 0,
         'RIBOSE-5P[c]': 0, 'RIBULOSE-5P[c]': 0, 'S-ADENOSYLMETHIONINE[c]': 8, 'SER[c]': 0, 'SHIKIMATE[c]': 0,
         'SIROHEME[c]': 0, 'SPERMIDINE[c]': 0, 'SUC-COA[c]': 0, 'SUC[c]': 0, 'TDP[c]': 0, 'THF[c]': 0,
         'THIAMINE-PYROPHOSPHATE[c]': 0, 'THREO-DS-ISO-CITRATE[c]': 0, 'THR[c]': 0, 'TRP[c]': 0, 'TTP[c]': 0,
         'TYR[c]': 0, 'UDP-GLUCURONATE[c]': 0, 'UDP-N-ACETYL-D-GLUCOSAMINE[c]': 0, 'UDP[c]': 0, 'UMP[c]': 0,
         'UNDECAPRENYL-DIPHOSPHATE[c]': 0, 'URIDINE[c]': 0, 'UTP[c]': 14648, 'VAL[c]': 0,
         'XYLULOSE-5-PHOSPHATE[c]': 0, 'ZN+2[c]': 0, 'ZN+2[p]': 0, 'glycogen-monomer[c]': 10}

        # configure exchanger stub process
        # TODO -- this needs a dictionary with {mol_id: exchanged counts/sec}
        exchanger_config = {'exchanges': example_update, 'time_step': metabolism_config['time_step']}
        exchanger_process = Exchange(exchanger_config)

        return {
            'metabolism': metabolism_process,
            'exchanger': exchanger_process,
        }

    def generate_topology(self, config):
        return {
            'metabolism': metabolism_topology,
            'exchanger': {
                'molecules': ('bulk',),
            }
        }


def run_metabolism():
    # load the sim data
    load_sim_data = LoadSimData(
        sim_data_path=SIM_DATA_PATH,
        seed=0)

    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_metabolism_gd_config()
    metabolism_process = MetabolismGD(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # TODO -- add perturbations to initial_state to test impact on metabolism

    metabolism_composite = metabolism_process.generate()
    experiment = Engine({
        'processes': metabolism_composite['processes'],
        'topology': {metabolism_process.name: metabolism_topology},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_timeseries()


def run_metabolism_composite():
    composer = MetabolismExchange()
    metabolism_composite = composer.generate()

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')


    experiment = Engine(
        processes= metabolism_composite['processes'],
        topology= metabolism_composite['topology'],
        initial_state= initial_state
    )

    experiment.update(10)

    data = experiment.emitter.get_data()


def test_ecoli_with_metabolism_gd(
        filename='fba_gd_swap',
        total_time=100,
        divide=False,
        progress_bar=True,
        log_updates=False,
        emitter='timeseries',
):
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + '.json')
    sim.total_time = total_time
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter

    # assert that the processes were swapped
    sim.build_ecoli()
    assert 'ecoli-metabolism-gradient-descent' in sim.ecoli['processes']
    assert 'ecoli-metabolism' not in sim.ecoli['processes']
    assert 'ecoli-metabolism-gradient-descent' in sim.ecoli['topology']
    assert 'ecoli-metabolism' not in sim.ecoli['topology']

    # run simulation and add asserts to output
    output = sim.run()

    # put asserts here to make sure it is behaving as expected
    assert output['listeners']['fba_results']



experiment_library = {
    '0': run_metabolism,
    '1': run_metabolism_composite,
    '2': test_ecoli_with_metabolism_gd,
}


# run experiments with command line arguments: python ecoli/experiments/metabolism_gd.py -n exp_id
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='metabolism with gd')
    parser.add_argument('--name', '-n', default=[], nargs='+', help='test ids to run')
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        experiment_library[name]()
    if run_all:
        for name, test in experiment_library.items():
            test()
