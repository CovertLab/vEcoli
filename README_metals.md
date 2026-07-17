# Regnerate Protein counts 

## Running on an HPC (recommended for large n_init_sims and generations #)
If running on an High Performance Compute (HPC) environment. See the following links for instructions to set up vEcoli on an HPC. Note that config files may need to be adjusted depending on your HPC cluster, all of which should be discussed in the following links

[Sherlock](https://covertlab.github.io/vEcoli/hpc.html#sherlock),
[other HPC cluster](https://covertlab.github.io/vEcoli/hpc.html#other-clusters)

1. Adjust config file 'configs/protein_counts_cofactor_HPC.json' before running the simulations to fit your HPC environment. If you run without adjusting the following, the sims may not run nor be saved where you expect them to be saved:
* experiment_id: 'name_your_output'
* out_dir: 'path for outputs'

### note that the current version of 'protein_counts_cofactor_HPC.json' is set to fit Stanford's sherlock HPC, much of the config may change depending on your HPC setup, see links above for more guidance on how to adjust the config file for your needs.

2. After setting up vEcoli with your HPC ensure the configs/protein_counts_cofactor_HPC.json has values "n_init_sims": 32 and "generations": 8. This will determine the number of 'cells' you simulate. Each n_init_sim is an indepent occurance of a cell, and generations is the number of cell cycles vEcoli will simulate for that seed. 

2. Running the sims will require you to run the following command: 

    python3 runscripts/workflow.py --config configs/protein_counts_cofactor_HPC.json

 Note your out_dir in configs/protein_counts_cofactor_HPC.json, this is where the output of all sims will reside.  

3. Once the sims are finished running, analysis results will be found at:
out_dir/experiment_id/analyses --> there should be two folders labeled 'variant=0' and 'variant=1'. variant 0 are the simulations run under minimal media and variant 1 are the sims run under rich media conditions. Each will have there own output at 
out_dir/experiment_id/analyses/variant=i/plots/analysis=protein_counts_cofactor/


## Running locally (not advised for large n_init_sims and generation #)
If running on your local machine follow the instructions in README.md to set up vEcoli

1. After vEcoli is setup on your machine ensure 'protein_counts_cofactor_local.json' has the correct number of "n_init_sims" and "generations" you want to run. Your hardware is the limiting factor in both time and size. 32 seeds X 8 generations will generate a folder of ~ 240 gb

2. Outputs will be found within the repo in a new 'out' folder. Analysis results will be found at 'out_dir/experiment_id/analyses/variant=i/plots/analysis=protein_counts_cofactor/'. Variant 0 are the simulations run under minimal media and variant 1 are the sims run under rich media conditions. Each will have there own output