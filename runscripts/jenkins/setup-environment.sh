set -e

export PYTHONPATH=$PWD
module load wcEcoli/python3 java/18.0.2

export PATH="${GROUP_HOME}/pyenv/bin:${PATH}"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

### Edit this line to make this branch use another pyenv
pyenv local viv-ecoli
pyenv activate

make clean compile
