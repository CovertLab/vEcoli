set -e

# Load newer Git, Java (for nextflow), and PyArrow
module load system git java/21.0.4 py-pyarrow

export PATH=$PATH:$GROUP_HOME/vEcoli_env
