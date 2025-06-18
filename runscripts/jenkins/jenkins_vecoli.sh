#!/bin/bash
#SBATCH --job-name=jenkins_vecoli
#SBATCH --dependency=singleton
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:SIGUSR1@90
#SBATCH --partition=mcovert
#SBATCH --output=/home/groups/mcovert/jenkins_vecoli/slurm_logs/%j.out
#SBATCH --error=/home/groups/mcovert/jenkins_vecoli/slurm_logs/%j.out

# Set the port Jenkins will use
port=8080

# Get the login node that submitted this job
submit_login_node=${SLURM_SUBMIT_HOST%%\.*}

# Generate a systematic list of login nodes to try
# Format: sh0G-ln0X where G=4,3,2 and X=1,2,...,8
generate_login_nodes() {
    local nodes=()

    # Try the submit node first if it exists
    if [ -n "$submit_login_node" ]; then
        nodes+=("$submit_login_node")
    fi

    # Then try the systematic pattern of login nodes
    for g in {4..2}; do
        for x in {1..8}; do
            node="sh0${g}-ln0${x}"
            # Don't add the submit node twice
            if [ "$node" != "$submit_login_node" ]; then
                nodes+=("$node")
            fi
        done
    done

    echo "${nodes[@]}"
}

# Get array of login nodes to try
login_nodes=($(generate_login_nodes))
echo "Will try these login nodes in order: ${login_nodes[@]}"

# Find first accessible login node
login_node=""
for node in "${login_nodes[@]}"; do
    echo "Testing if $node is accessible..."
    ssh -q -o BatchMode=yes -o ConnectTimeout=5 "$USER@$node" echo accessible &>/dev/null
    if [ $? -eq 0 ]; then
        login_node=$node
        echo "Found accessible login node: $login_node"
        break
    else
        echo "Node $node is not accessible"
    fi
done

# Exit if no accessible login node was found
if [ -z "$login_node" ]; then
    echo "ERROR: Could not find any accessible login node. Cannot proceed."
    exit 1
fi

# Function to handle SIGUSR1 signal by resubmitting job from login node
_resubmit() {
    echo "$(date): job $SLURM_JOBID received SIGUSR1 at $(date), re-submitting from $login_node"

    # Get the current job's mail-user setting
    current_mail_user=$(scontrol show job $SLURM_JOBID | grep -oP 'MailUser=\K[^ ]*')
    echo "Current mail-user: $current_mail_user"

    # Create a temporary script to execute on the login node
    temp_script=$(mktemp)
    cat >$temp_script <<EOF
#!/bin/bash
cd $PWD
echo "Resubmitting Jenkins job from login node $login_node"
sbatch --mail-user=$current_mail_user /tmp/jenkins_script.sh
EOF

    # Copy scripts to login node and execute it there
    scp $temp_script $USER@$login_node:/tmp/resubmit_jenkins.sh
    scp $0 $USER@$login_node:/tmp/jenkins_script.sh
    ssh $USER@$login_node "chmod +x /tmp/resubmit_jenkins.sh && /tmp/resubmit_jenkins.sh && rm /tmp/resubmit_jenkins.sh /tmp/jenkins_script.sh" &>/dev/null

    # Check if the job was successfully submitted
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to resubmit job from login node $login_node"
        echo "Script path: $SCRIPT_PATH"

        # Clean up local temp script before exiting
        rm -f $temp_script
        ssh -o BatchMode=yes $USER@$login_node "rm -f /tmp/resubmit_jenkins.sh /tmp/jenkins_script.sh" &>/dev/null || true

        # Exit with error status
        exit 1
    else
        echo "Job successfully resubmitted from $login_node"
    fi

    # Clean up local temp script
    rm $temp_script

    # Continue running until job actually ends
    echo "Continuing to run until job is terminated"
}

# Register the trap for SIGUSR1
trap _resubmit SIGUSR1

# Update the job comment with the login node info for SSH tunneling
scontrol update jobid=$SLURM_JOBID comment="Jenkins UI accessible via: ssh username@$login_node.sherlock.stanford.edu -L $port:localhost:$port"

# Set up port forwarding from compute node to login node
echo "Setting up SSH tunnel to $login_node..."
ssh -nNT "$USER@$login_node" -R $port:localhost:$port &
SSH_PID=$!

# Verify the SSH tunnel was established
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to establish SSH tunnel to $login_node"
    kill $SSH_PID &>/dev/null

    # Try other nodes if the first choice failed
    for node in "${login_nodes[@]}"; do
        if [ "$node" != "$login_node" ]; then
            echo "Trying $node for forwarding..."
            ssh -nNT "$USER@$node" -R $port:localhost:$port &
            SSH_PID=$!
            if [ $? -eq 0 ]; then
                login_node=$node
                scontrol update jobid=$SLURM_JOBID comment="$login_node"
                echo "Using $login_node for SSH tunnel"
                break
            else
                kill $SSH_PID &>/dev/null
            fi
        fi
    done
fi

# Set trap to clean up SSH tunnel on exit
cleanup() {
    echo "Cleaning up..."
    kill $SSH_PID &>/dev/null
    echo "SSH tunnel terminated"
}
trap cleanup EXIT

# Start Jenkins
echo "Starting Jenkins on port $port, tunneled to $login_node"
module load java/21.0.4 fontconfig
JENKINS_HOME=$GROUP_HOME/jenkins_vecoli java -jar $GROUP_HOME/jenkins_vecoli/jenkins.war --httpPort=$port &
JENKINS_PID=$!

# Wait for Jenkins to finish
wait $JENKINS_PID
