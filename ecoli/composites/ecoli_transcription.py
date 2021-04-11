from vivarium.core.process import Composer
from vivarium.core.experiment import Experiment

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import get_state_from_file, SIM_DATA_PATH

from ecoli.processes.transcript_initiation import TranscriptInitiation
from ecoli.processes.transcript_elongation import TranscriptElongation

class Transcription(Composer):

    defaults = {
        'initiation' : {},
        'elongation' : {}
    }

    def __init__(self, config=None):
        super().__init__(config)

    def generate_processes(self, config):
        initiation = TranscriptInitiation(config['initiation'])
        elongation = TranscriptElongation(config['elongation'])

        return {'initiation' : initiation, 'elongation' : elongation}

    def generate_topology(self, config):
        topology = {
            'initiation': {
                'environment': ('environment',),
                'full_chromosomes': ('unique', 'full_chromosome'),
                'RNAs': ('unique', 'RNA'),
                'active_RNAPs': ('unique', 'active_RNAP'),
                'promoters': ('unique', 'promoter'),
                'molecules': ('bulk',),
                'listeners': ('listeners',)},

            'elongation': {
                'environment': ('environment',),
                'RNAs': ('unique', 'RNA'),
                'active_RNAPs': ('unique', 'active_RNAP'),
                'molecules': ('bulk',),
                'bulk_RNAs': ('bulk',),
                'ntps': ('bulk',),
                'listeners': ('listeners',)},
        }

        return topology

def test_transcription(total_time=10):
    load_sim_data = LoadSimData(sim_data_path=SIM_DATA_PATH,
                                seed=0)

    config = {
        'initiation' : load_sim_data.get_transcript_initiation_config(),
        'elongation' : load_sim_data.get_transcript_elongation_config()
    }
    transcription_composer = Transcription(config)
    transcription_composite = transcription_composer.generate()
    transcription_composite.processes['elongation'].request_on = True

    initial_state = get_state_from_file(path=f'data/wcecoli_t0.json')

    experiment_config = {
        'processes' : transcription_composite.processes,
        'topology' : transcription_composite.topology,
        'initial_state' : initial_state
    }
    transcription_experiment = Experiment(experiment_config)
    transcription_experiment.update(total_time)

    data = transcription_experiment.emitter.get_timeseries()

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_transcription()