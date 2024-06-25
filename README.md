# Vivarium *E. coli*

![vivarium](doc/_static/ecoli_master_topology.png)

## Background

Vivarium *E. coli* (vEcoli) is a port of the Covert Lab's 
[E. coli Whole Cell Model](https://github.com/CovertLab/wcEcoli) (wcEcoli)
to the [Vivarium framework](https://github.com/vivarium-collective/vivarium-core). Its main benefits over the old model are:

1. Modular processes: easily add/remove processes that interact with
    existing or new simulation state
2. Unified configuration: all configuration happens through JSON files,
    making it easy to run simulations/analyses with different options
3. Parquet output: simulation output is in a widely-supported columnar
    file format that enables fast, larger-than-RAM analytics with DuckDB

As in wcEcoli, [raw experimental data](reconstruction/ecoli/flat) is first processed
by the parameter calculator or [ParCa](reconstruction/ecoli/fit_sim_data_1.py) to calculate 
model parameters (e.g. transcription probabilities). These parameters are used to configure [processes](ecoli/processes) that are linked together
into a [complete simulation](ecoli/experiments/ecoli_master_sim.py).

For more details, refer to the [user guide](https://covertlab.github.io/vivarium-ecoli/index.html).

## Installation

> **Note:** The following instructions assume a Linux or MacOS system. Windows users can
> attempt to follow the same instructions after setting up 
> [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).

pyenv lets you install and switch between multiple Python releases and multiple "virtual 
environments", each with its own pip packages. Using pyenv, create a virtual environment 
and install Python 3.11.3. For a tutorial on how to create a virtual environment, follow 
the instructions [here](https://github.com/CovertLab/wcEcoli/blob/master/docs/create-pyenv.md) 
and stop once you reach the "Create the 'wcEcoli3' python virtual environment" step. Then, 
run the following command in your terminal:

    pyenv virtualenv 3.11.3 viv-ecoli && pyenv local viv-ecoli

Update `pip`, `setuptools` and `wheel` to avoid issues with these:

    pip install --upgrade pip setuptools wheel

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

After verifying your Java installation, you can proceed with the Nextflow installation
steps, starting with `curl -s https://get.nextflow.io | bash`. If this command fails, try prepending
`export CAPSULE_LOG=verbose` and re-run it, checking for failed downloads. If found, simply rerun this command until it succeeds.

## Test Installation

To test your installation, from the top-level of the cloned repository, invoke:

    # Must set PYTHONPATH and OMP_NUM_THREADS for every new shell
    export PYTHONPATH=.
    export OMP_NUM_THREADS=1
    python scripts/workflow.py --config ecoli/composites/ecoli_configs/test_installation.json

This will run the following basic simulation workflow:

1. Run the [parameter calculator](scripts/parca.py) to generate simulation data.
2. Run the [simulation](ecoli/experiments/ecoli_master_sim.py) for a single generation, saving output in `out` folder.
3. [Analyze simulation output](scripts/analysis.py) by creating a [mass fraction plot](ecoli/analysis/single/mass_fraction_summary.py).


## Next Steps
For details on configuring simulations or workflows, 
see the [configurations README](readmes/ecoli_configurations.md).
For a walkthrough of a typical model development cycle, see
the [walkthrough notebook](notebooks/workflow.ipynb).
