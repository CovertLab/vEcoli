import os

from vivarium.core.control import run_library_cli
from vivarium.core.composition import EXPERIMENT_OUT_DIR
from vivarium.core.engine import pf
from vivarium.library.topology import get_in
from vivarium.core.serialize import deserialize_value
from vivarium.library.units import remove_units

from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.plots.snapshots import plot_snapshots


def test_antibiotics_nitrocefin():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'antibiotics_nitrocefin.json')
    sim.emitter = 'timeseries'
    sim.total_time = 2
    sim.run()
    data = sim.query()


def test_antibiotics_tetracycline_cephaloridine():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'antibiotics_tetracycline_cephaloridine.json')
    sim.emitter = 'timeseries'
    sim.total_time = 2
    sim.run()
    data = sim.query()


def test_lysis_rxn_dff_environment():
    beta_lactamase = 'EG10040-MONOMER[p]'
    beta_lactam = 'beta-lactam'
    lysis_time = 4

    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'lysis_trigger.json')
    sim.emitter = 'timeseries'
    sim.total_time = 10

    # add to the timeline, triggering burst
    sim.process_configs['timeline'] = {
        'timeline': [
            (0, {
                ('bulk', beta_lactamase): 100,
                ('bulk', beta_lactam): 100,
            }),
            (lysis_time, {
                ('burst',): True
            })
        ]
    }
    sim.run()

    # retrieve data and pre-process for plotting
    query = [
        ('dimensions', 'bounds'),
        ('fields', beta_lactamase),
        ('fields', beta_lactam),
        ('agents', '0', 'boundary'),
        ('agents', '0', 'burst'),
        ('agents', '0', 'bulk', beta_lactamase),
        ('agents', '0', 'bulk', beta_lactam),
    ]
    data = sim.query(query=query)
    data = deserialize_value(data)
    data = remove_units(data)
    for t, v in data.items():
        if 'agents' not in v:
            data[t]['agents'] = {}  # add empty agents back in

    assert '0' in data[0.0]['agents']  # agent 0 is present at time=0
    assert not data[lysis_time+2.0]['agents']  # not agents after lysis_time

    # plot
    out_dir = os.path.join(EXPERIMENT_OUT_DIR, 'lysis_environment')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plot_snapshots(
        n_snapshots=6,
        bounds=get_in(data, (max(data), 'dimensions', 'bounds')),
        agents={
            time: d['agents']
            for time, d in data.items()
        },
        fields={
            time: d['fields']
            for time, d in data.items()
        },
        out_dir=out_dir,
        filename='snapshots',
    )


library = {
    '0': test_antibiotics_nitrocefin,
    '1': test_antibiotics_tetracycline_cephaloridine,
    '2': test_lysis_rxn_dff_environment,
}

# python ecoli/experiments/antibiotics_tests.py -n library_id
if __name__ == "__main__":
    run_library_cli(library)
