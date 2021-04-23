
from vivarium.core.composition import process_in_experiment

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH, get_state_from_file

from ecoli.processes.complexation import Complexation


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

def test_ecoli_complexation():
    time_step=2
    total_time=10

    config = load_sim_data.get_complexation_config(time_step=time_step)
    complexation = Complexation(config)

    initial_state = get_state_from_file(path='data/wcecoli_t1000.json')
    experiment = process_in_experiment(complexation, initial_state=initial_state)
    experiment.update(total_time)

    data = experiment.emitter.get_data()

    # TODO -- perform test here

    # # another option
    # update = complexation.next_update(time_step, initial_state)
    # import ipdb; ipdb.set_trace()




if __name__=='__main__':
    test_ecoli_complexation()
