import argparse
import json
import os
import pickle
import shutil
import time

from configs import CONFIG_DIR_PATH
from ecoli.experiments.ecoli_master_sim import SimConfig
from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
from reconstruction.ecoli.fit_sim_data_1 import fitSimData_1
from validation.ecoli.validation_data_raw import ValidationDataRawEcoli
from validation.ecoli.validation_data import ValidationDataEcoli
from wholecell.utils import constants
import wholecell.utils.filepath as fp


def run_parca(config):
    # Make output directory
    kb_directory = fp.makedirs(config["outdir"], constants.KB_DIR)

    raw_data_file = os.path.join(kb_directory, constants.SERIALIZED_RAW_DATA)
    sim_data_file = os.path.join(kb_directory, constants.SERIALIZED_SIM_DATA_FILENAME)
    raw_validation_data_file = os.path.join(
        kb_directory, constants.SERIALIZED_RAW_VALIDATION_DATA
    )
    validation_data_file = os.path.join(
        kb_directory, constants.SERIALIZED_VALIDATION_DATA
    )

    print(f"{time.ctime()}: Instantiating raw_data with operons={config['operons']}")
    raw_data = KnowledgeBaseEcoli(
        operons_on=config["operons"],
        remove_rrna_operons=config["remove_rrna_operons"],
        remove_rrff=config["remove_rrff"],
        stable_rrna=config["stable_rrna"],
        new_genes_option=config["new_genes"],
    )
    print(f"{time.ctime()}: Saving raw_data")
    with open(raw_data_file, "wb") as f:
        pickle.dump(raw_data, f)

    print(f"{time.ctime()}: Instantiating sim_data with operons={config['operons']}")
    sim_data = fitSimData_1(
        raw_data=raw_data,
        cpus=config["cpus"],
        debug=config["debug_parca"],
        load_intermediate=config["load_intermediate"],
        save_intermediates=config["save_intermediates"],
        intermediates_directory=config["intermediates_directory"],
        variable_elongation_transcription=config["variable_elongation_transcription"],
        variable_elongation_translation=config["variable_elongation_translation"],
        disable_ribosome_capacity_fitting=(not config["ribosome_fitting"]),
        disable_rnapoly_capacity_fitting=(not config["rnapoly_fitting"]),
        cache_dir=config["cache_dir"],
    )
    print(f"{time.ctime()}: Saving sim_data")
    with open(sim_data_file, "wb") as f:
        pickle.dump(sim_data, f)

    print(f"{time.ctime()}: Instantiating raw_validation_data")
    raw_validation_data = ValidationDataRawEcoli()
    print(f"{time.ctime()}: Saving raw_validation_data")
    with open(raw_validation_data_file, "wb") as f:
        pickle.dump(raw_validation_data, f)

    print(f"{time.ctime()}: Instantiating validation_data")
    validation_data = ValidationDataEcoli()
    validation_data.initialize(raw_validation_data, raw_data)
    print(f"{time.ctime()}: Saving validation_data")
    with open(validation_data_file, "wb") as f:
        pickle.dump(validation_data, f)


def main():
    parser = argparse.ArgumentParser(description="run_parca")
    default_config = os.path.join(CONFIG_DIR_PATH, "default.json")
    parser.add_argument(
        "--config",
        action="store",
        default=default_config,
        help=(
            "Path to configuration file for the simulation. "
            "All key-value pairs in this file will be applied on top "
            f"of the options defined in {default_config}."
        ),
    )
    parser.add_argument(
        "-c",
        "--cpus",
        type=int,
        help="The number of CPU processes to use. Default = 1.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Directory to hold ParCa output kb folder. "
        "Default = reconstruction/sim_data",
    )
    parser.add_argument(
        "--operons",
        action=argparse.BooleanOptionalAction,
        help="Turn operons on (polycistronic).",
    )
    parser.add_argument(
        "--ribosome-fitting",
        action=argparse.BooleanOptionalAction,
        help="Fit ribosome expression to protein synthesis demands.",
    )
    parser.add_argument(
        "--rnapoly-fitting",
        action=argparse.BooleanOptionalAction,
        help="Fit RNA polymerase expression to protein synthesis demands.",
    )
    parser.add_argument(
        "--remove-rrna-operons",
        action=argparse.BooleanOptionalAction,
        help="Remove the seven rRNA operons. Does not have any effect if"
        " --no-operons specified.",
    )
    parser.add_argument(
        "--remove-rrff",
        action=argparse.BooleanOptionalAction,
        help="Remove the rrfF gene. If operons are enabled,"
        " removes the rrfF gene from the rrnD operon.",
    )
    parser.add_argument(
        "--debug-parca",
        action=argparse.BooleanOptionalAction,
        help="Make Parca calculate only one arbitrarily-chosen transcription"
        " factor condition when adjusting gene expression levels, leaving"
        " the other TFs at their input levels for faster Parca debugging."
        " DO NOT USE THIS FOR A MEANINGFUL SIMULATION.",
    )
    parser.add_argument(
        "--load-intermediate",
        type=str,
        help="The function in the parca to load (skips functions that would"
        " have run before the function). Must run with --save-intermediates"
        " first.",
    )
    parser.add_argument(
        "--save-intermediates",
        action=argparse.BooleanOptionalAction,
        help="If set, saves sim_data and cell_specs at intermediate"
        " function calls in the parca.",
    )
    parser.add_argument(
        "--intermediates-directory",
        type=str,
        help="Directory to save or load intermediate sim_data and cell_specs"
        " results from if --load-intermediate or --save-intermediates"
        " are set.",
    )
    parser.add_argument(
        "--variable-elongation-transcription",
        action=argparse.BooleanOptionalAction,
        help="Use a different elongation rate for different transcripts"
        " (currently increases rates for rRNA). Usually set this"
        " consistently between runParca and runSim.",
    )
    parser.add_argument(
        "--variable-elongation-translation",
        action=argparse.BooleanOptionalAction,
        help="Use a different elongation rate for different polypeptides"
        " (currently increases rates for ribosomal proteins)."
        " Usually set this consistently between runParca and runSim.",
    )

    config_file = os.path.join(CONFIG_DIR_PATH, "default.json")
    args = parser.parse_args()
    with open(config_file, "r") as f:
        config = json.load(f)
    if args.config is not None:
        config_file = args.config
        with open(os.path.join(args.config), "r") as f:
            SimConfig.merge_config_dicts(config, json.load(f))
    # ParCa options are defined under `parca_options` key in config JSON
    # Merge these with CLI arguments, which take precedence
    parca_options = config.pop("parca_options")
    for k, v in vars(args).items():
        if v is not None:
            parca_options[k] = v
    # Expand outdir to absolute path
    parca_options["outdir"] = os.path.abspath(parca_options["outdir"])
    # Set cache directory for ParCa to outdir/cache
    parca_options["cache_dir"] = os.path.join(parca_options["outdir"], "cache")
    os.makedirs(parca_options["cache_dir"], exist_ok=True)
    # If config defines a sim_data_path, skip ParCa
    if config["sim_data_path"] is not None:
        out_kb = os.path.join(parca_options["outdir"], "kb")
        if not os.path.exists(out_kb):
            os.makedirs(out_kb)
        print(f"{time.ctime()}: Skipping ParCa. Using {config['sim_data_path']}")
        shutil.copy(config["sim_data_path"], out_kb)
    else:
        run_parca(parca_options)


if __name__ == "__main__":
    main()
