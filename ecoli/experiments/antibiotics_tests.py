import os

from vivarium.core.control import run_library_cli
from vivarium.core.composition import EXPERIMENT_OUT_DIR
from vivarium.core.engine import pf
from vivarium.library.topology import get_in
from vivarium.core.serialize import deserialize_value
from vivarium.library.units import remove_units

from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.plots.snapshots import plot_snapshots


def test_antibiotics_tetracycline_cephaloridine():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'antibiotics_tetracycline_cephaloridine.json')
    sim.emitter = 'timeseries'
    sim.total_time = 2
    sim.run()
    data = sim.query()


def remove_empty_values(d):
    """remove {key: value} pairs with values that are None"""
    to_delete = []
    for k, v in d.items():
        if not v:
            to_delete.append(k)
        elif isinstance(v, dict):
            v2 = remove_empty_values(v)
            if not v2:
                to_delete.append(k)
            else:
                d[k] = v2
    for k in to_delete:
        del d[k]
    return d


def test_lysis_rxn_dff_environment(total_time = 10):
    beta_lactamase = 'EG10040-MONOMER[p]'
    beta_lactam = 'beta-lactam'
    hydrolyzed_beta_lactam = 'hydrolyzed-beta-lactam'
    lysis_time = 4

    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'lysis_trigger.json')
    sim.emitter = 'timeseries'
    sim.total_time = total_time

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
        ('fields', hydrolyzed_beta_lactam),
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

    assert '0' in data[0.0]['agents'] \
           and len(data[0.0]['agents']) == 1  # agent 0 is present at time=0
    after_lysis = remove_empty_values(data[lysis_time + 2.0]['agents'])
    assert len(after_lysis) == 0  # no agents after lysis_time

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
        colorbar_decimals=8,
    )


library = {
    '1': test_antibiotics_tetracycline_cephaloridine,
    '2': test_lysis_rxn_dff_environment,
}

# python ecoli/experiments/antibiotics_tests.py -n library_id
if __name__ == "__main__":
    run_library_cli(library)
