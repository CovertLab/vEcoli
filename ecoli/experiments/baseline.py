"""
========================
ecoli_master experiments
========================
"""

from vivarium.core.composition import simulate_compartment_in_experiment
from vivarium.core.control import Control

# composites
from ecoli.composites.ecoli_master import (
    Ecoli,
    get_initial_state
)

# plots
from ecoli.plots.mRNA_comparison import mrna_comparison_plot

def run_experiment():
    agent_id = '1'
    ecoli = Ecoli({'agent_id': agent_id})
    initial_state = get_initial_state()
    settings = {
        'timestep': 1,
        'total_time': 10,
        'initial_state': initial_state}

    data = simulate_compartment_in_experiment(ecoli, settings)

    return data


experiments_library = {
    '1': run_experiment,
}

plots_library = {
    'mrna': mrna_comparison_plot
}

workflow_library = {
    '1': {
        'name': 'main',
        'experiment': '1',
        'plots': ['mrna'],
    }
}

if __name__ == '__main__':
    Control(
        experiments=experiments_library,
        plots=plots_library,
        workflows=workflow_library,
        )
