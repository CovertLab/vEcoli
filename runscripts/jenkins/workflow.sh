set -e

# Print stderr of failed Nextflow tasks on exit
function on_exit {
  nextflow log -q | tail -n 1 | awk '{print $1}' | xargs -I {} nextflow log {} -f name,stderr,workdir -F "status == 'FAILED'"
}

# Set the trap for any exit (including error)
trap on_exit EXIT

source runscripts/jenkins/setup-environment.sh

python3 runscripts/workflow.py --config $1
