import argparse
import os
import pickle
import time

from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
from reconstruction.ecoli.fit_sim_data_1 import fitSimData_1
from validation.ecoli.validation_data_raw import ValidationDataRawEcoli
from validation.ecoli.validation_data import ValidationDataEcoli
from wholecell.utils import constants
import wholecell.utils.filepath as fp


def main():
    parser = argparse.ArgumentParser(description='run_parca')
    parser.add_argument('-c', '--cpus', type=int, default=1,
        help='The number of CPU processes to use. Default = 1.')
    parser.add_argument('-o', '--outdir', type=str,
        default='reconstructions/sim_data', help='Directory to hold ParCa'
        ' output kb folder. Default = reconstructions/sim_data')
    parser.add_argument('--operons', type=bool, default=False,
        help='Turn operons on (polycistronic). Default = False.')
    parser.add_argument('--ribosome_fitting', type=bool, default=True,
        help='Fit ribosome expression to protein synthesis demands.')
    parser.add_argument('--rnapoly_fitting', type=bool, default=True,
        help='Fit RNA polymerase expression to protein synthesis demands.')
    parser.add_argument('--remove_rrna_operons', type=bool, default=False,
        help='Remove the seven rRNA operons. Does not have any effect if'
        ' operon flag is False (default).')
    parser.add_argument('--remove_rrff', type=bool, default=False,
        help='Remove the rrfF gene. If operon flag is True (default: False),'
        ' removes the rrfF gene from the rrnD operon.')
    parser.add_argument('--debug_parca', type=bool, default=False,
        help='Make Parca calculate only one arbitrarily-chosen transcription'
        ' factor condition when adjusting gene expression levels, leaving'
        ' the other TFs at their input levels for faster Parca debugging.'
        ' DO NOT USE THIS FOR A MEANINGFUL SIMULATION.')
    parser.add_argument('--load_intermediate', default=None,
        help='The function in the parca to load (skips functions that would'
        ' have run before the function). Must run with --save-intermediates'
        ' first.')
    parser.add_argument('--save_intermediates', action='store_true',
        help='If set, saves sim_data and cell_specs at intermediate'
        ' function calls in the parca.')
    parser.add_argument('--intermediates_directory', default=None,
        help='Directory to save or load intermediate sim_data and cell_specs'
        ' results from if --load-intermediate or --save-intermediates'
        ' are set.')
    parser.add_argument('--variable_elongation_transcription', type=bool,
        default=True, help='Use a different elongation rate for different'
        'transcripts (currently increases rates for rRNA). Usually set this'
        ' consistently between runParca and runSim.')
    parser.add_argument('--variable_elongation_translation', type=bool,
        default=False, help='Use a different elongation rate for different'
        'polypeptides (currently increases rates for ribosomal proteins). '
        'Usually set this consistently between runParca and runSim.')
    
    args = parser.parse_args()

    # Make output directory
    kb_directory = fp.makedirs(fp.ROOT_PATH, args.outdir, constants.KB_DIR)

    raw_data_file = os.path.join(kb_directory, constants.SERIALIZED_RAW_DATA)
    sim_data_file = os.path.join(
        kb_directory, constants.SERIALIZED_SIM_DATA_FILENAME)
    metrics_data_file = os.path.join(
        kb_directory, constants.SERIALIZED_METRICS_DATA_FILENAME)
    raw_validation_data_file = os.path.join(
        kb_directory, constants.SERIALIZED_RAW_VALIDATION_DATA)
    validation_data_file = os.path.join(
        kb_directory, constants.SERIALIZED_VALIDATION_DATA)
    intermediates_dir = kb_directory


    print(f"{time.ctime()}: Instantiating raw_data with operons={args.operons}")
    raw_data = KnowledgeBaseEcoli(
        operons_on=args.operons,
        remove_rrna_operons=args.remove_rrna_operons,
        remove_rrff=args.remove_rrff)
    print(f"{time.ctime()}: Saving raw_data")
    with open(raw_data_file, 'wb') as f:
        pickle.dump(raw_data, f)
    
    print(f"{time.ctime()}: Instantiating sim_data with operons={args.operons}")
    sim_data = fitSimData_1(
        raw_data=raw_data, cpus=args.cpus, debug=args.debug_parca,
        load_intermediate=args.load_intermediate,
        save_intermediates=args.save_intermediates,
        intermediates_directory=args.intermediates_directory,
        variable_elongation_transcription=args.variable_elongation_transcription,
        variable_elongation_translation=args.variable_elongation_translation,
        disable_ribosome_capacity_fitting=(not args.ribosome_fitting),
        disable_rnapoly_capacity_fitting=(not args.rnapoly_fitting)
    )
    print(f"{time.ctime()}: Saving sim_data")
    with open(sim_data_file, 'wb') as f:
        pickle.dump(sim_data, f)

    print(f"{time.ctime()}: Instantiating raw_validation_data")
    raw_validation_data = ValidationDataRawEcoli()
    print(f"{time.ctime()}: Saving raw_validation_data")
    with open(raw_validation_data_file, 'wb') as f:
        pickle.dump(raw_validation_data, f)
    
    print(f"{time.ctime()}: Instantiating validation_data")
    validation_data = ValidationDataEcoli()
    validation_data.initialize(raw_validation_data, raw_data)
    print(f"{time.ctime()}: Saving validation_data")
    with open(validation_data_file, 'wb') as f:
        pickle.dump(validation_data, f)


if __name__ == '__main__':
    main()
