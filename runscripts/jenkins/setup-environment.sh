set -e

# Load newer Git, Java (for nextflow), and Python
module load system git java/21.0.4 python/3.12.1

export PATH=$PATH:$GROUP_HOME/vEcoli_env
