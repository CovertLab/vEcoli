set -e

# Load newer Git and Java for nextflow 
module load system git java/21.0.4

# Set PYTHONPATH to root of repo so imports work
export PYTHONPATH=$PWD
# Use one thread for OpenBLAS (better performance and reproducibility)
export OMP_NUM_THREADS=1

# Initialize pyenv
export PYENV_ROOT="${GROUP_HOME}/pyenv"
if [ -d "${PYENV_ROOT}" ]; then
    export PATH="${PYENV_ROOT}/bin:${PATH}"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

### Edit this line to make this branch use another pyenv
pyenv local vEcoli
pyenv activate
