# Vivarium *E. coli*

![vivarium](doc/_static/ecoli_master_topology.png)

## Background

Vivarium *E. coli* (vEcoli) is a port of the Covert Lab's 
[E. coli Whole Cell Model](https://github.com/CovertLab/wcEcoli) (wcEcoli)
to the [Vivarium framework](https://github.com/vivarium-collective/vivarium-core).
Its main benefits over the original model are:

1. **Modular processes:** easily add/remove processes that interact with
    existing or new simulation state
2. **Unified configuration:** all configuration happens through JSON files,
    making it easy to run simulations/analyses with different options
3. **Parquet output:** simulation output is in a widely-supported columnar
    file format that enables fast, larger-than-RAM analytics with DuckDB
4. **Google Cloud support:** workflows too large to run on a local machine
    can be easily run on Google Cloud

As in wcEcoli, [raw experimental data](reconstruction/ecoli/flat) is first processed
by the parameter calculator or [ParCa](reconstruction/ecoli/fit_sim_data_1.py) to calculate 
model parameters (e.g. transcription probabilities). These parameters are used to configure
[processes](ecoli/processes) that are linked together into a
[complete simulation](ecoli/experiments/ecoli_master_sim.py).

## Installation

> **Note:** The following instructions assume a Linux or MacOS system. Windows users can
> attempt to follow the same instructions after setting up 
> [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).

> **Note:** Refer to the following pages for non-local setups: 
> [Sherlock](https://covertlab.github.io/vEcoli/workflows.html#sherlock),
> [other HPC cluster](https://covertlab.github.io/vEcoli/workflows.html#other-hpc-clusters),
> [Google Cloud](https://covertlab.github.io/vEcoli/gcloud.html).

pyenv lets you install and switch between multiple Python releases and multiple "virtual 
environments", each with its own pip packages. Using pyenv, create a virtual environment 
and install Python 3.11.3. For a tutorial on how to install pyenv and other dependencies,
follow the instructions [here](https://github.com/CovertLab/wcEcoli/blob/master/docs/dev-tools.md).
Then, run the following command in your terminal:

    pyenv virtualenv 3.11.3 viv-ecoli && pyenv local viv-ecoli

Update `pip`, `setuptools` and `wheel` to avoid issues with these:

    pip install --upgrade pip setuptools==73.0.1 wheel

Now, install numpy (check `requirements.txt` for the exact version):

    pip install numpy==1.26.4

Then install the remaining requirements:

    pip install -r requirements.txt

And build the Cython components:

    make clean compile

Install `nextflow` following the instructions [here](https://www.nextflow.io/docs/latest/install.html).
After installing Java with SDKMAN!, close and reopen your terminal, then run
the following commands and compare their output to ensure that the Java compiler
and JVM were properly installed and the same version.

    javac -version
    java -version

After verifying your Java installation, you can proceed with installing Nextflow,
starting with `curl -s https://get.nextflow.io | bash`. If this fails, try
prepending `export CAPSULE_LOG=verbose` and re-run, checking for failed downloads.
If any downloads failed, re-run this command until it succeeds.

## Test Installation

To test your installation, from the top-level of the cloned repository, invoke:

    # Must set PYTHONPATH and OMP_NUM_THREADS for every new shell (can add to .bashrc/.zshrc)
    export PYTHONPATH=.
    export OMP_NUM_THREADS=1
    python runscripts/workflow.py --config ecoli/composites/ecoli_configs/test_installation.json

This will run the following basic simulation workflow:

1. Run the [parameter calculator](runscripts/parca.py) to generate simulation data.
2. Run the [simulation](ecoli/experiments/ecoli_master_sim.py)
    for a single generation, saving output in `out` folder.
3. [Analyze simulation output](runscripts/analysis.py) by creating a
    [mass fraction plot](ecoli/analysis/single/mass_fraction_summary.py).


## Next Steps
Check out the [user guide](https://covertlab.github.io/vEcoli/) for a high-level
tutorial of model development, details on key model components, and low-level API documentation.
