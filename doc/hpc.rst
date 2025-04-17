============
HPC Clusters
============

vEcoli uses Nextflow and Apptainer containers to run on high-performance
computing (HPC) clusters. For users with access to the Covert Lab's partition
on Sherlock, follow the instructions in the :ref:`sherlock` section. For users
looking to run the model on other HPC clusters, follow the instructions in the
:ref:`other-cluster` section.

To speed up HPC workflows, vEcoli supports the HyperQueue executor. See :ref:`hyperqueue`
for more information. 

.. _sherlock:

--------
Sherlock
--------

On Sherlock, once a workflow is started with :mod:`runscripts.workflow`,
``runscripts/container/build-image.sh`` builds an Apptainer image with
a minimal snapshot of your cloned repository. Nextflow starts containers
using this image to run the steps of the workflow. To run or interact
with the model outside of :mod:`runscripts.workflow`, start an
interactive container by following the steps in :ref:`sherlock-interactive`.

.. note::
  Files that match the patterns in ``.dockerignore`` are excluded from the
  Apptainer image.

.. _sherlock-setup:

Setup
=====

.. note::
    The following setup applies to members of the Covert Lab only.

After cloning the model repository to your home directory, add the following
lines to your ``~/.bash_profile``, then close and reopen your SSH connection:

.. code-block:: bash

    # Load newer Git and Java for Nextflow 
    module load system git java/21.0.4
    # Include shared Nextflow, uv, and HyperQueue installations on PATH
    export PATH=$PATH:$GROUP_HOME/vEcoli_env
    # Load virtual environment with PyArrow (only Python dependency
    # required to start a containerized workflow)
    source $GROUP_HOME/vEcoli_env/.venv/bin/activate

.. note::
    To update the versions of the shared dependencies, navigate to
    ``$GROUP_HOME/vEcoli_env`` and run the appropriate command:

    1. Nextflow: ``NXF_EDGE=1 nextflow self-update``
    2. uv: ``uv self update``
    3. HyperQueue: See :ref:`hyperqueue`.
    4. PyArrow: ``uv sync --upgrade``

.. _sherlock-config:

Configuration
=============

To tell vEcoli that you are running on Sherlock, you MUST add the following
options to your configuration JSON (note the top-level ``sherlock`` key)::

  {
    "sherlock": {
      # Boolean, whether to build a fresh Apptainer image. If files that are
      # not excluded by .dockerignore did not change since your last build,
      # you can set this to false to skip building the image.
      "build_image": true,
      # Absolute path (including file name) of Apptainer image to
      # build (or use directly, if build_image is false)
      "container_image": "",
      # Boolean, whether to use HyperQueue executor for simulation jobs
      # (see HyperQueue section below)
      "hyperqueue": true,
      # Boolean, denotes that a workflow is being run as part of Jenkins
      # continuous integration testing. Randomizes the initial seed and
      # ensures that all STDOUT and STDERR is piped to the launching process
      # so they can be reported by GitHub
      "jenkins": false
    }
  }

In addition to these options, you **MUST** set the emitter output directory
(see description of ``emitter_arg`` in :ref:`json_config`) to a path with
enough space to store your workflow outputs. We recommend setting this to
a location in your ``$SCRATCH`` directory (e.g. ``/scratch/users/{username}/out``).

.. warning::
    The output path **MUST** be given in full because environment
    variables in the path (e.g. ``$SCRATCH``) are not automatically resolved.

With these options in the configuration JSON, a workflow can be started by
running ``python runscripts/workflow.py --config {}.json`` on a login node.
This submits a job that will run the Nextflow workflow orchestrator
with a 7-day time limit on the lab's dedicated partition. The workflow orchestrator
will automatically submit jobs for each step in the workflow: one for the ParCa,
one to create variants, one for each cell, and one for each analysis.

If you are trying to run a workflow that takes longer than 7 days, you can
use the resume functionality (see :ref:`fault_tolerance`). Alternatively,
consider running your workflow on Google Cloud, which has no maximum workflow
runtime (see :doc:`gcloud`).

.. note::
  Unlike workflows run locally, Sherlock workflows are run using
  containers with a snapshot of the repository at the time the workflow
  was launched. This means that any changes made to the repository after
  launching a workflow will not be reflected in that workflow.

There is a 4 hour time limit on each job in the workflow, including analyses.
This is a generous limit designed to accomodate very slow-dividing cells.
Generally, we recommend that users exclude analysis scripts which take more
than a few minutes to run from their workflow configuration. Instead, create a
SLURM batch script to run these analyses using :py:mod:`runscripts.analysis`
directly. This also lets you request more CPU cores and RAM for better performance.

.. _sherlock-interactive:

Interactive Container
=====================

.. note::
  The following commands should all be run from the directory where you cloned
  the vEcoli repository.

To debug a failed job in a workflow, you must locate the container image that was
used for that workflow. You can refer to the ``container_image`` key in the
config JSON saved to the workflow output directory (see :ref:`output`). Start
an interactive container with that image name as follows::

  runscripts/container/interactive.sh -i container_image -a

Now, inside the container, navigate to ``/vEcoli`` and add breakpoints to
scripts as you see fit. Finally, navigate to the working directory (see
:ref:`troubleshooting`) for the task that you want to debug. By invoking
``bash .command.sh``, the job will run and pause upon reaching your
breakpoints, allowing you to inspect variables and step through the code.

.. warning::
  Any changes that you make to ``/vEcoli`` inside the container are discarded
  when the container terminates.

The files located in ``/vEcoli`` are a copy of the repository (excluding
files ignored by ``.dockerignore``) at the time the workflow was launched.
To start an interactive container that reflects the current state of your
cloned repository, add the ``-d`` flag to start a "development" container.

In this mode, instead of editing source files in ``/vEcoli``, you can
directly edit the source files in your cloned repository and have those
changes immediately reflected when running those scripts inside the
container. Because you are just modifying your cloned repository, any
code changes you make will persist after the container terminates and
can be tracked using Git version control.

.. note::
  If the image you use to start a development container was built with
  an outdated version of ``uv.lock`` or ``pyproject.toml``, there will
  be a delay on startup while uv updates the packages. To avoid this,
  build a new image with ``runscripts/container/build-image.sh``.

.. _jenkins-setup:

Jenkins Setup
=============

The following describes the steps taken to set up Jenkins on Sherlock to run
long continuous integration tests on the ``master`` branch of vEcoli.

Request an interactive session on Sherlock, taking note of the login node. Once
the interactive session is started, run the following command to forward
the port used by Jenkins to the login node::

    ssh -nNT {username}@{login node} -R 8080:localhost:8080 &

In this same session, download the latest WAR file from the Jenkins website,
load the Java and fontconfig modules, then run Jenkins::

    wget https://get.jenkins.io/war-stable/latest/jenkins.war
    module load java/17.0.4 fontconfig
    JENKINS_HOME=$GROUP_HOME/jenkins_vecoli java -jar jenkins.war --httpPort=8080

In a new terminal, open a new SSH connection to the previously noted login node
with port forwarding::

    ssh {username}@{login node}.sherlock.stanford.edu -L 8080:localhost:8080

On a local machine, open a web browser and navigate to ``localhost:8080``. Proceed
with the post-installation setup wizard (see `Jenkins documentation <https://www.jenkins.io/doc/book/installing/#setup-wizard>`_).

Manually select the following basic plugins to install:
Folders, OWASP Markup Formatter, Build Timeout, Credentials Binding,
Timestamper, Workspace Cleanup, Pipeline, GitHub Branch Source,
Pipeline: GitHub Groovy Libraries, Pipeline Graph View, Git, GitHub,
Matrix Authorization, Email Extension, Mailer, and Dark Theme.

Create an admin user with a username and password of your choice, and keep the
default web URL of ``localhost:8080``. After setup is complete, click on
``Manage Jenkins`` in the left sidebar, then ``Plugins``. Click ``Available Plugins``
in the left sidebar, then search for and install the ``GitHub Checks`` plugin.

Follow the `linked instructions <https://docs.cloudbees.com/docs/cloudbees-ci/latest/cloud-admin-guide/github-app-auth>`_
to create a GitHub App for the Covert Lab organization,
install it on the vEcoli repository, and add it as a credential in Jenkins.

Stop the Jenkins server by pressing ``Ctrl+C`` in the terminal where it is running.
Then, move the ``jenkins.war`` file to the ``$GROUP_HOME/jenkins_vecoli`` directory.
Create a new file called ``jenkins_vecoli.sh`` in the same directory with the following::

    #!/bin/bash
    #SBATCH --job-name=jenkins_vecoli
    #SBATCH --dependency=singleton
    #SBATCH --time=5-00:00:00
    #SBATCH --mem-per-cpu=4GB
    #SBATCH --cpus-per-task=1
    #SBATCH --mail-type=FAIL
    #SBATCH --signal=B:SIGUSR1@90
    #SBATCH --partition=mcovert

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
                    scontrol update jobid=$SLURM_JOBID comment="Jenkins UI accessible via: ssh username@$login_node.sherlock.stanford.edu -L $port:localhost:$port"
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

Finally, create a directory called ``slurm_logs`` in ``$GROUP_HOME/jenkins_vecoli`` and
``cd`` into it. From here, launch Jenkins with ``sbatch --mail-user={your email here} ../jenkins_vecoli.sh``.
This will queue a persistent Jenkins job that should run indefinitely, resubmitting itself
every 5 days. The stdout and stderr from these jobs will be written to the directory in which
you ran the ``sbatch`` command. Remember to run ``sbatch`` in ``slurm_logs`` to keep all logs
in a consistent location accessible to all members of the lab. You will get an email if any of
these jobs fail, in which case you should review the most recent logs and resubmit with ``sbatch``.

.. _new-jenkins-jobs:

Adding New Jenkins Jobs
=======================

First, create a new branch and push a commit to GitHub with your new Jenkinsfile. Refer
to the existing Jenkinsfiles in ``runscripts/jenkins/Jenkinsfile`` for examples.

From the main Jenkins dashboard, click ``New Item`` in the left sidebar and
select ``Multibranch Pipeline``.

Under ``Branch Sources``:

1. Select ``GitHub``.
2. Select the GitHub App credential added in :ref:`jenkins-setup`.
3. Enter the vEcoli repository URL.

Under ``Behaviors``:

1. Add the ``Filter by name (with wildcards)`` behavior and set ``Include`` to ``master``.
   To test the pipeline, you can temporarily add the name of your new branch, then save the
   pipeline. Jenkins should recognize the Jenkinsfile on your branch and trigger the pipeline
   (including setting GitHub commit statuses). Make sure to remove your branch from this
   section, and save the pipeline again when you are done testing.
2. Add the ``Status Checks Properties`` behavior, give it an informative name, and
   tick ``Skip GitHub Branch Source notifications``.

Under ``Build Configuration``:

1. Replace ``Jenkinsfile`` with the path to the Jenkinsfile for the pipeline relative
   to the root of the repository (e.g. ``runscripts/jenkins/Jenkinsfile/anaerobic``).

Click ``Save`` to create the pipeline, scan the repository for branches that match the filter
and contain the Jenkinsfile, and trigger the pipeline as appropriate.


.. _other-cluster:

--------------
Other Clusters
--------------

Nextflow has support for a wide array of HPC schedulers. If your HPC cluster uses
a supported scheduler, you can likely run vEcoli on it with fairly minimal modifications.

Prerequisites
=============

The following are required:

- Nextflow (requires Java)
- PyArrow
- Git clone vEcoli to a location that is accessible from all nodes in your cluster

If your cluster has Apptainer (formerly known as Singularity) installed,
check to see if it is configured to automatically mount all filesystems (see
`Apptainer docs <https://apptainer.org/docs/user/main/bind_paths_and_mounts.html#system-defined-bind-paths>`_).
If not, you may run into errors when running workflows because
Apptainer containers are read-only. You may be able to resolve this by
adding ``--writeable-tmpfs`` to ``containerOptions`` for the ``sherlock``
and ``sherlock-hq`` profiles in ``runscripts/nextflow/config.template``.

If this does not work, Nextflow allows users to define ``beforeScript`` and
``afterScript`` directives for each process that we can potentially use to create
and clean up Apptainer overlay files. Then, the ``containerOptions``
directive can be modified to start containers with these overlays. However,
the simplest solution is likely to set up vEcoli as if Apptainer was not
available (see below). Note that if Apptainer is not configured to automount
filesystems, you will need to manually specify paths to mount when debugging
with interactive containers (see :ref:`sherlock-interactive`). This can be done
using the ``-p`` argument for ``runscripts/container/interactive.sh``.

If your cluster does not have Apptainer, you can try the following steps:

1. Completely follow the local setup instructions in the README (install uv, etc).
2. Delete the following lines from ``runscripts/nextflow/config.template``::

    process.container = 'IMAGE_NAME'
    ...
    apptainer.enabled = true

3. Make sure to always set ``build_runtime_image`` to false in your config JSONs
   (see :ref:`sherlock-config`)


.. _cluster-options:

Cluster Options
===============

If your HPC cluster uses the SLURM scheduler,
you can use vEcoli on that cluster by changing the ``queue`` option in
``runscripts/nextflow/config.template`` and all instances of
``--partition=QUEUE(S)`` in :py:mod:`runscripts.workflow` to the
right queue(s) for your cluster.

If your HPC cluster uses a different scheduler, refer to the Nextflow
`executor documentation <https://www.nextflow.io/docs/latest/executor.html>`_
for more information on configuring the right executor. Beyond changing queue
names as described above, this could be as simple as modifying the ``executor``
directives for the ``sherlock`` and ``sherlock_hq`` profiles in
``runscripts/nextflow/config.template``.


.. _hyperqueue:

----------
HyperQueue
----------

HyperQueue is a job scheduler that is designed to run on top of a traditional HPC
scheduler like SLURM. It consists of a head server that can automatically allocate
worker jobs using the underlying HPC scheduler. These worker jobs can be configured
to persist for long enough to complete multiple tasks, greatly reducing the overhead
of job submission and queuing, especially for shorter jobs.

HyperQueue is distributed as a pre-built binary on GitHub.
Unfortunately, this binary is built with a newer version of GLIBC
than is available on Sherlock, necessitating a rebuild from source. A binary
built in this way is available in ``$GROUP_HOME/vEcoli_env`` (added to ``PATH``
in :ref:`sherlock-setup`) to users with access to the Covert Lab's partition
on Sherlock.

To build from source (e.g. to update to a newer version), follow
`these instructions <https://it4innovations.github.io/hyperqueue/stable/installation/#compilation-from-source-code>`_.
