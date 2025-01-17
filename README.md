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

## Setup

> **Note:** The following instructions assume a local Linux or MacOS system. Windows users can
> attempt to follow the same steps after setting up 
> [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install). Refer to the following pages for non-local setups:
> [Sherlock](https://covertlab.github.io/vEcoli/workflows.html#sherlock),
> [other HPC cluster](https://covertlab.github.io/vEcoli/workflows.html#other-hpc-clusters),
> [Google Cloud](https://covertlab.github.io/vEcoli/gcloud.html).

### Prerequisites

Your system must have git, curl (or wget), and a C compiler.

On Ubuntu/Debian, apt can be used to install all three prerequisites:

    sudo -s eval 'apt update && apt install git curl clang'

On MacOS, curl is preinstalled and git and clang come with the Xcode Command Line Tools:

    xcode-select --install

### Installation

Clone the repository:

    git clone https://github.com/CovertLab/vEcoli.git

[Follow these instructions](https://docs.astral.sh/uv/getting-started/installation/)
to install `uv`, our Python package and project manager of choice.

Navigate into the cloned repository and use `uv` to install the model:

    cd vEcoli
    uv sync --frozen

> **Note:** If your C compiler is not `clang`, run `CC={your compiler} uv sync --frozen`
> instead to work around [this limitation](https://github.com/astral-sh/uv/issues/8429).
> For example, `CC=gcc uv sync --frozen` for `gcc`.

Finally, install `nextflow` [following these instructions](https://www.nextflow.io/docs/latest/install.html).
If you choose to install Java with SDKMAN!, after the Java installation
finishes, close and reopen your terminal before continuing with the
`nextflow` installation steps.

> **Tip:** If any step in the `nextflow` installation fails,
> try rerunning a few times to see if that fixes the issue.

If you are installing the model for active development, we strongly
recommend that you also install the development dependencies using:

    uv sync --frozen --extra dev

After that, you can run ``uv run pre-commit install`` to install
a pre-commit hook that will run the ``ruff`` linter and formatter
before all of your commits.

The development dependencies also include ``pytest``, which lets
you run the test suite, and ``mypy``, which can be invoked to
perform static type checking.

## Test Installation

To test your installation, from the top-level of the cloned repository, invoke:

    uv run runscripts/workflow.py --config ecoli/composites/ecoli_configs/test_installation.json

> **Note:** Start all of your commands to run scripts with `uv run`. Alternatively,
> you can source the Python virtual environment that `uv` created with
> `source .venv/bin/activate` and use `python` as normal, though we recommend
> sticking to `uv run` where possible.

This will run the following basic simulation workflow:

1. Run the [parameter calculator](runscripts/parca.py) to generate simulation data.
2. Run the [simulation](ecoli/experiments/ecoli_master_sim.py)
    for a single generation, saving output in `out` folder.
3. [Analyze simulation output](runscripts/analysis.py) by creating a
    [mass fraction plot](ecoli/analysis/single/mass_fraction_summary.py).


## Next Steps
Check out the [user guide](https://covertlab.github.io/vEcoli/) for a high-level
tutorial of model development, details on key model components, and low-level API documentation.
