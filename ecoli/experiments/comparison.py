"""
==================================
ecoli_master comparison to wcEcoli
==================================

run by calling a workflow from the command line. For example workflow 1:
```
$ python ecoli/experiments/comparison.py -w 1
```

"""

from six.moves import cPickle

from vivarium.core.composition import simulate_compartment_in_experiment
from vivarium.core.control import Control

# composites
from ecoli.composites.ecoli_master import (
    Ecoli,
    SIM_DATA_PATH,
)

# plots
from ecoli.plots.comparison import (
    mrna_scatter_comparison,
    fluxome_scatter_comparison,
    protein_counts_scatter_comparison,
    production_rate_plot,
    mass_fractions,
    mass_fractions_summary,
)


# get sim_data_config for plots
with open(SIM_DATA_PATH, 'rb') as sim_data_file:
    sim_data = cPickle.load(sim_data_file)
sim_data_config = {'sim_data': sim_data}


def run_experiment():
    agent_id = '1'
    ecoli = Ecoli({'agent_id': agent_id})
    initial_state = ecoli.initial_state()
    settings = {
        'experiment_name': 'run-ecoli',
        'description': 'testing vivarium-ecoli',
        'time_step': 1,
        'total_time': 30,
        'initial_state': initial_state,
        # 'emitter': {'type': 'database'}
        }
    return simulate_compartment_in_experiment(ecoli, settings)


# libraries for control
experiments_library = {
    '1': run_experiment,
}
plots_library = {
    'mrna': {
        'plot': mrna_scatter_comparison,
        'config': sim_data_config,
    },
    'fluxome': {
        'plot': fluxome_scatter_comparison,
        'config': sim_data_config,
    },
    'protein_count': {
        'plot': protein_counts_scatter_comparison,
        'config': sim_data_config,
    },
    'production_rate': {
        'plot': production_rate_plot,
        'config': sim_data_config,
    },
    'mass_fractions': {
        'plot': mass_fractions,
        'config': sim_data_config,
    },
    'mass_fractions_summary': {
        'plot': mass_fractions_summary,
        'config': sim_data_config,
    }
}
workflow_library = {
    '1': {
        'name': 'main',
        'experiment': '1',
        'plots': [
            'mrna',
            'fluxome',
            'protein_count',
            'production_rate',
            'mass_fractions',
            'mass_fractions_summary',
        ],
    }
}

if __name__ == '__main__':
    Control(
        experiments=experiments_library,
        plots=plots_library,
        workflows=workflow_library,
        )
