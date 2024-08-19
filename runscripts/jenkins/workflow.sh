set -e

source runscripts/jenkins/setup-environment.sh

python runscripts/workflow.py --config $1
