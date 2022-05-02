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
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'lysis.json')
    sim.emitter = 'timeseries'
    sim.total_time = 10
    sim.run()

    # retrieve data
    query = [
        ('dimensions', 'bounds'),
        ('fields', 'beta-lactam'),
        ('fields', 'beta-lactamase'),
        ('agents', '0', 'boundary')
    ]
    data = sim.query(query=query)
    data = deserialize_value(data)
    data = remove_units(data)

    print(data[0.0]['fields'])
    print(pf(data[0.0]['agents']))
    # print(data[10.0]['dimensions'])

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
