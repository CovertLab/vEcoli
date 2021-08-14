"""
==================
Colony Simulations
==================
"""
from vivarium.core.engine import Engine, pf
from vivarium.core.composition import simulate_composite
from vivarium.library.units import units

# vivarium-multibody imports
from vivarium_multibody.composites.lattice import (
    Lattice, make_lattice_config)
from vivarium_multibody.composites.grow_divide import GrowDivide



def get_lattice_composite(
        diffusion_rate=0.001,
        initial_concentration=1.0,
        bins=[10, 10],
        bounds=[10, 10],
        growth_rate=0.03,
        growth_noise=10**-6,
        depth=10
):
    lattice_config = make_lattice_config(
            bounds=bounds,
            n_bins=bins,
            concentrations={
                'glc': initial_concentration},
            diffusion=diffusion_rate,
            depth=depth)

    lattice_composer = Lattice(lattice_config)
    lattice_composite = lattice_composer.generate()

    growth_config = {'default_growth_rate': growth_rate, 'default_growth_noise': growth_noise}
    grow_divide_composer = GrowDivide({'agent_id': '0', 'growth': growth_config})

    agent_id = '0'
    grow_divide_composite = grow_divide_composer.generate(path=('agents', agent_id))
    lattice_composite.merge(composite=grow_divide_composite)

    return lattice_composite

def simulate_grow_divide_lattice(
        lattice_composite,
        total_time=100,
        initial_state=None,
):

    agent_id = '0'
    if initial_state is None:
        initial_state = {
            'agents': {
                agent_id: {
                    'global': {
                        'mass': 1000 * units.femtogram}
                }}}

    sim_settings = {
        'total_time': total_time,
        'initial_state': initial_state,
        # 'return_raw_data': True,
    }

    lattice_grow_divide_data = simulate_composite(
        lattice_composite, sim_settings)

    return lattice_grow_divide_data


def main():
    lattice_composite = get_lattice_composite()

    print('INITIAL CONFIG:')
    print(pf(lattice_composite))

    data = simulate_grow_divide_lattice(lattice_composite, total_time=20)

    print('FINAL CONFIG:')
    print(pf(lattice_composite))

    print('AGENT 0:')
    print(pf(data['agents']['0']))

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
