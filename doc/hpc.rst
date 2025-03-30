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
an Apptainer image is built with all vEcoli Python dependencies using
``runscripts/container/build-runtime.sh``. Nextflow starts containers
using this image that run the steps of the workflow. To run or interact
with the model without using :mod:`runscripts.workflow`, start an
interactive container by following the steps in :ref:`sherlock-interactive`.

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
      # Boolean, whether to build a fresh Apptainer runtime image. If uv.lock
      # did not change since your last build, you can set this to false
      "build_runtime_image": true,
      # Absolute path (including file name) of Apptainer runtime image to
      # build (or use directly, if build_runtime_image is false)
      "runtime_image_name": "",
      # Boolean, whether to use HyperQueue executor for simulation jobs
      # (see section below)
      "use_hyperqueue": true,
      # Boolean, denotes that a workflow is being run as part of Jenkins
      # continuous integration testing. Randomizes the initial seed and
      # ensures that all STDOUT and STDERR is piped to the launching process
      # so they can be reported by GitHub
      "jenkins": false
    }
  }

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


.. warning::
    The emitter output directory (see description of ``emitter_arg``
    in :ref:`json_config`) should be an absolute path to somewhere in your
    ``$SCRATCH`` directory (e.g. ``/scratch/users/{username}/out``). The path must
    be absolute because Nextflow does not resolve environment variables like
    ``$SCRATCH`` in paths.

.. note::
    There is a 4 hour limit on each job in the workflow, including analyses.
    
This is a generous limit designed to accomodate very slow-dividing cells.
Generally, we recommend that analysis scripts which take more than a few minutes
to run be excluded from your workflow configuration. Instead, create a SLURM batch
script to run these analyses using :py:mod:`runscripts.analysis` directly. This
also lets you request more CPU cores and RAM for better performance.

.. _sherlock-interactive:

Interactive Container
=====================

To run and develop the model on Sherlock outside a workflow, you must
have previously run a containerized workflow (default on Sherlock) with
``build_runtime_image`` set to true and the current version of
``uv.lock``. If you are not sure if ``uv.lock`` changed since your last
containerized workflow (or if you have never run a containerized workflow),
run the following to build a new runtime image, picking any ``runtime_image_path``::
  
  runscripts/container/build-runtime.sh -r runtime_image_path -a

Once you have a runtime image, you can start an interactive container with,
substituting in your ``runtime_image_path``::

  runscripts/container/interactive.sh -w runtime_image_path -a

Inside this interactive container, you can use vEcoli as normal. Any code
changes that you make in the cloned repository will be immediately reflected
in commands run inside the container.

If you are trying to debug a failed job in a workflow, add breakpoints to
any Python script in your cloned repository by inserting::

  import ipdb; ipdb.set_trace()
  
Then, inside the interactive container, navigate to the working directory (see
:ref:`troubleshooting`) for the task that you want to debug. By invoking
``bash .command.sh``, the job will run and pause upon reaching your
breakpoints, allowing you to inspect variables and step through the code.


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

    # catch the SIGUSR1 signal
    _resubmit() {
        ## Resubmit the job for the next execution
        echo "$(date): job $SLURM_JOBID received SIGUSR1 at $(date), re-submitting"
        sbatch $0
    }
    trap _resubmit SIGUSR1

    port=8080
    login_node=${SLURM_SUBMIT_HOST%%\.*}

    # Show login node for users to ssh to with `ssh <user>@<login_node> -L <port>:localhost:<port>`
    # in order to access Jenkins web UI in job comment and map Jenkins port to login node port.
    scontrol update jobid=$SLURM_JOBID comment=$login_node
    ssh -nNT "$USER@$login_node" -R $port:localhost:$port &

    module load java/17.0.4 fontconfig
    JENKINS_HOME=$GROUP_HOME/jenkins_vecoli java -jar $GROUP_HOME/jenkins_vecoli/jenkins.war --httpPort=$port &
    wait

Submit this script with ``sbatch $GROUP_HOME/jenkins_vecoli/jenkins_vecoli.sh``. This will
queue a persistent Jenkins job that should run indefinitely, resubmitting itself every 5 days.

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
If not, workflows should still run but you will need to manually specify mount paths
when debugging with interactive containers (see :ref:`sherlock-interactive`).
This can be done using the ``-p`` argument for ``runscripts/container/interactive.sh``.

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
