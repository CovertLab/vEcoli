============
HPC Clusters
============

vEcoli uses Nextflow and Apptainer containers to run on high-performance
computing (HPC) clusters. For users with access to the Covert Lab's partition
on Sherlock, follow the instructions in the :ref:`sherlock` section. For users
looking to run the model on other HPC clusters, follow the instructions in the
:ref:`other-cluster` section.

To speed up HPC workflows, vEcoli supports the HyperQueue executor. See :ref:`hq-info`
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

.. _sherlock-setup:

Setup
=====

.. note::
    The following setup applies to members of the Covert Lab only.

Request a Sherlock Account
--------------------------

If you've never had a Sherlock account: Go to https://www.sherlock.stanford.edu/ and click on ``Request an Account``

.. note::
   Markus will have to approve this.

If you've had a Sherlock account for a previous group: Email srcc-support@stanford.edu and ask them to move your account to mcovert, and CC Markus on the email and in the email body ask for Markus to give approval

Additional Resources: Sherlock Documentation from Stanford
----------------------------------------------------------

* https://srcc.stanford.edu/workshops/sherlock-boarding-session
* https://www.sherlock.stanford.edu/docs/

Login to Sherlock
-----------------

.. code-block:: bash

   ssh <YOUR_SU_NET_ID>@login.sherlock.stanford.edu
   # Type in Stanford Password
   # Do the Duo authentication
   # The following setup steps should be done using the Sherlock terminal 
   # NOTE that this is a LOGIN node, so no major computing should be done here

   # It is best to use a compute node for things like cloning the repo, running code, resetting lpad, etc

   srun -p mcovert --time=4:00:00 --cpus-per-task=2 --pty bash

   # srun is the command for launching a job step under Slurm
   # -p or --partition specifies which partition (queue) to use, choose covert :D
   # --time: sets the job's wall‑clock time limit
   # --cpus-per-task specifies # CPU cores for each task in this job step
   # --pty: allocates a pseudo‑terminal (TTY) to run an interactive session
   # bash: launching a Bash shell
   # When it finished, usually you can see your JOB ID in your shell

   # You can use scancel to abort your job step
   scancel <YOUR_JOB_ID>

You can also refer to the Sherlock Documentation: https://www.sherlock.stanford.edu/docs/getting-started/connecting/

Clone the vEcoli Repository
----------------------------

1. Git clone the vEcoli repo to your Sherlock account:

.. code-block:: bash

   git clone https://github.com/CovertLab/vEcoli.git

If you have already created your branch, you can use:

.. code-block:: bash

   # View all the branches (including remote branch)
   git branch -a

   # Checkout to your own branch
   git checkout <your_branch_name>

   # Validate your current branch
   git branch

After cloning the model repository to your home directory, add the following
lines to your ``~/.bash_profile``, then close and reopen your SSH connection:

.. code-block:: bash

    # Load newer Git, Java (for nextflow), and Python
    module load system git java/21.0.4 python/3.12.1
    # Include shared Nextflow and HyperQueue installations on PATH
    export PATH=$PATH:$GROUP_HOME/vEcoli_env

Then, run the following to test your setup:

.. code-block:: bash

  python3 runscripts/workflow.py --config configs/test_sherlock.json

This will run a small workflow that:

1. Builds an Apptainer image with a snapshot of your cloned repository.
2. Runs the ParCa.
3. Runs one simulation.
4. Runs the mass fraction analysis.

All output files will be saved to a ``test_sherlock`` directory in your
cloned repository. You can modify the workflow output directory by changing
the ``out_dir`` option under ``emitter_arg`` in the config JSON.
See :ref:`sherlock-config` for a description of the Sherlock-specific
configuration options and :ref:`sherlock-running` for details about running
a workflow on Sherlock.

.. note::
   ``test_sherlock.json`` sets ``out_dir`` to ``.``. In relative path syntax,
   this refers to the current directory, meaning the cloned repo. This makes
   the configuration portable as it does not assume the presence of any other
   folders. However, as noted in :ref:`sherlock-config`, we recommend changing
   this in your workflows. 

To run scripts on Sherlock outside a workflow, see :ref:`sherlock-interactive`.
To run scripts on Sherlock through a SLURM batch script, see :ref:`sherlock-noninteractive`.

.. tip::
   * You can use ``nano`` as text editor:
   
   .. code-block:: bash
   
      nano ~/.bash_profile
      # After writing, you can use Ctrl+O to write out, Enter to confirm, and Ctrl+X to exit
      
   * If you choose to use ``vim``, press ``i`` for insert, and press ``Esc``, then type ``:wq`` and Enter for writing out
   * Before running the ``python3`` to set up the env, ensure you are in the vEcoli repo
   * It usually takes time to run first job

.. note::
    The above setup is sufficient to run workflows on Sherlock. However, if you
    have a compelling reason to update the shared Nextflow or HyperQueue binaries,
    navigate to ``$GROUP_HOME/vEcoli_env`` and run:

    1. Nextflow: ``NXF_EDGE=1 nextflow self-update``
    2. HyperQueue: See :ref:`hq-info`.

    Then, reset the permissions of the updated binaries with ``chmod 777 *``.

.. warning::

   Before building your own config file and running an experiment, remember:

   Python scripts (other than runscripts/workflow.py) **WILL NOT** run on Sherlock directly. 
   This includes the standalone ParCa, simulation, and analysis run scripts. 
   Instead, these scripts can be run inside an :ref:`sherlock-interactive` (ideal for script development or debugging) 
   or :ref:`sherlock-noninteractive` (ideal for longer or more resource-intensive scripts that do not require user input).

.. _sherlock-config:

Configuration
=============

To tell vEcoli that you are running on Sherlock, you MUST include the following
keys in your configuration JSON (note the top-level ``sherlock`` key):

.. code-block::

  {
    "sherlock": {
      # Boolean, whether to build a fresh Apptainer image. If files that are
      # not excluded by .dockerignore did not change since your last build,
      # you can set this to false to skip building the image. DO NOT set this
      # to a location in the cloned repo or else the resulting image(s) will be
      # included in future image builds. test_sherlock.json is an exception
      # because the test_sherlock folder is ignored by .dockerignore.
      "build_image": true,
      # Path (relative or absolute, including file name) of Apptainer image to
      # build (or use directly, if build_image is false)
      "container_image": "",
      # Boolean, whether to run using HyperQueue.
      "hyperqueue": false,
      # Boolean, denotes that a workflow is being run as part of Jenkins
      # continuous integration testing. Randomizes the initial seed and
      # ensures that all STDOUT and STDERR is piped to the launching process
      # so they can be reported by GitHub
      "jenkins": false
    }
  }

In addition to these options, you **MUST** set the emitter output directory
(see description of ``emitter_arg`` in :ref:`json_config`) to a path with
enough space to store your workflow outputs. 

.. important::
   We recommend setting ``out_dir`` under ``emitter_arg`` to a location in your
   ``$SCRATCH`` directory to circumvent the ``$HOME`` storage limit
   (run ``sh_quota`` to view). One way to do this is using an absolute path
   (e.g. ``/scratch/users/{username}``). Alternatively, you can create a
   symlink to your scratch directory by running the following command inside
   your cloned repository: ``ln -s /scratch/users/{username} out`` (delete
   ``out`` in your cloned repo first if it already exists). Then, using ``out``
   for ``out_dir`` will cause all simulation output to be redirected to your
   scratch directory.

If using the Parquet emitter and ``threaded`` is not set to false under
``emitter_arg``, a warning will be printed suggesting that you set ``threaded``
to false. This ensures that simulations use only a single CPU core, the default
that is allocated per simulation on Sherlock (regardless of whether HyperQueue
is used). On Sherlock, storage speed is not a bottleneck, so performance with
``threaded`` set to false and 1 core per simulation is comparable to running
with ``threaded`` unset (default: true) and 2 cores per simulation.

If storage speed was a bottleneck, the additional thread would allow
simulation execution to continue while Polars writes Parquet files to disk.
To properly take advantage of this, you would also need to increase the number
of cores per simulation to 2 by modifying the ``cpus`` directive under the
``sherlock`` and ``sherlock_hq`` profiles in ``runscripts/nextflow/config.template``.

.. warning::
  ``~`` and environment variables like ``$SCRATCH`` are not expanded in the
  configuration JSON. See the warning box at :doc:`workflows`.

.. _sherlock-running:

Running Workflows
=================

With these options in the configuration JSON, a workflow can be started by
running ``python3 runscripts/workflow.py --config {}``, substituting
in the path to your config JSON. 

.. warning::
  Remember to use ``python3`` to start workflows instead of ``python``.
  This command is supposed to run on **login node**, which means there is no need to use ``srun`` to request a **compute node**.
  If there is trouble with permission denied for nextflow (you can use ``nextflow -version`` to check out), you can try ``chmod a+rwx``

If ``build_image`` is true in your config JSON, the terminal will report that
a SLURM job was submitted to build the container image. When the image build
job starts, the terminal will report the build progress.

.. note::
  Files that match the patterns in ``.dockerignore`` are excluded from the image.

.. note::
  If the Apptainer build fails, eg:
  ``FATAL:   While performing build: conveyor failed to get: unexpected end of JSON input``,
  try cleaning cache: ``apptainer cache clean``
.. warning::
  Do not make any changes to your cloned repository or close your SSH
  connection until the build has finished.

Once the build has finished, the terminal will report that a **SLURM job**
was submitted for the Nextflow workflow orchestrator before exiting
back to the shell. At this point, you are free to close your connection,
start additional workflows, etc. You can use ``squeue`` to view the status of your SLURM job:

.. code-block:: bash

   # View by job
   squeue -j <Your_Job_ID>
   # View by user
   squeue -u <Your_user_name>

Unlike workflows run locally, Sherlock's
containerized workflows mean any changes made to the repository after the
container image has been built will not affect the running workflow.

Once started, the Nextflow job will stay alive for the duration of the
workflow (up to 7 days) and submit new SLURM jobs as needed.

If you are trying to run a workflow that takes longer than 7 days, you can
use the resume functionality (see :ref:`fault_tolerance`). Alternatively,
consider running your workflow on Google Cloud, which has no maximum workflow
runtime (see :doc:`gcloud`).

You can start additional, concurrent workflows that each build a new image
with different modifications to the cloned repository. However, if possible,
we recommend designing your code to accept options through the config JSON
which modify the behavior of your workflow without modifying core code. This
allows you to save time by reusing a previously built image as follows:
set ``build_image`` to false and ``container_image`` to the path of said image.

There is a 4 hour time limit on each job in the workflow, including analyses.
This is a generous limit designed to accomodate very slow-dividing cells.
Generally, we recommend that users exclude analysis scripts which take more
than a few minutes from their workflow configuration. Instead, either run these
manually following :ref:`sherlock-interactive` or create a
SLURM batch script to run these analyses following :ref:`sherlock-noninteractive`.

.. _sherlock-interactive:

Interactive Container
=====================

.. warning::
  The following steps should be run on a compute node. See the
  `Sherlock documentation <https://www.sherlock.stanford.edu/docs/user-guide/running-jobs/?h=interactive#interactive-jobs>`_
  for details.
  
The maximum resource request for an interactive compute
node is 2 hours, 4 CPU cores, and 8GB RAM/core. Scripts that require more
resources should be submitted as SLURM batch scripts to the ``mcovert``
or ``owners`` partition (see :ref:`sherlock-noninteractive`).

To run scripts on Sherlock, you must have either:

- Previously run a workflow on Sherlock and have access to the built container image
- Built a container image manually using ``runscripts/container/build-image.sh`` with
  the ``-a`` flag

Start an interactive container with your full image path (see the warning box at
:doc:`workflows`) by navigating to your cloned repository and running:

.. code-block:: bash

  runscripts/container/interactive.sh -i container_image -a

.. note::
  Inside the interactive container, you can safely use ``python`` directly
  in addition to the usual ``uv`` commands.

The above command launches a container containing a snapshot of your
cloned repository as it was when the image was built. This snapshot
is located at ``/vEcoli`` inside the container and is mostly intended
to guarantee reproducibility for troubleshooting failed workflow jobs.
More specifically, users who wish to debug a failed workflow job should:

1. Start an interactive container with the image used to run the workflow.
2. Use ``nano`` to add breakpoints (``import ipdb; ipdb.set_trace()``)
   to the relevant scripts in ``/vEcoli``.
3. Navigate to the working directory (see :ref:`troubleshooting`) for the
   job that you want to debug.
4. Invoke ``bash .command.sh`` to run the failing task and pause upon
   reaching your breakpoints, allowing you to inspect variables and step
   through the code.

.. warning::
  ``~`` and environment variables like ``$SCRATCH`` do not work
  inside the container. Follow the instructions in the warning box at
  :doc:`workflows` **outside** the container to get the full path to
  use inside the container.

.. danger::
  Any changes that you make to ``/vEcoli`` inside the container are discarded
  when the container terminates.

Moreover, if you want to exit the interactive image, just type ``exit`` command.

To start an interactive container that reflects the current state of your
cloned repository, navigate to your cloned repository and run the above
command with the ``-d`` flag to start a "development" container:

.. code-block:: bash

  runscripts/container/interactive.sh -i container_image -a -d

In this mode, instead of editing source files in ``/vEcoli``, you can
directly edit the source files in your cloned repository and have those
changes immediately reflected when running those scripts inside the
container. Because you are just modifying your cloned repository, any
code changes you make will persist after the container terminates and
can be tracked using Git version control.

.. _sherlock-noninteractive:

Non-Interactive Container
=========================

To run any script inside a container without starting an interactive session,
use the same command as :ref:`sherlock-interactive` but specify a command
using the ``-c`` flag. For example, to run the ParCa process, navigate to
your cloned repository and run the following command, replacing ``container_image``
with the path to your container image and ``{}`` with the path to your
configuration JSON:

.. code-block:: bash

  runscripts/container/interactive.sh -i container_image -c "python /vEcoli/runscripts/parca.py --config {}"

This feature is intended for use in
`SLURM batch scripts <https://www.sherlock.stanford.edu/docs/getting-started/submitting/#batch-scripts>`_
to manually run analysis scripts with custom resource requests. Make sure
to include one of the following directives at the top of your script:

- ``#SBATCH --partition=owners``: This is the largest partition on Sherlock and
  the most likely to have free resources available for job scheduling. Even so,
  queue times are variable, and other users may preempt your job at any moment,
  though this is anecdotally rare for small jobs under an hour long.
- ``#SBATCH --partition=mcovert``: Best for high priority scripts (short queue time)
  that you cannot risk being preempted. The number of available cores is 32 minus
  whatever is currently being used by other users in the ``mcovert`` partition.
  Importantly, if all 32 cores are in use by ``mcovert`` users, not only will your
  script have to wait for resources to free up, so will any workflows. As such,
  treat this partition as a limited resource reserved for high priority jobs.
- ``#SBATCH --partition=normal``: Potentially longer queue time than either of the
  two options above but no risk of preemption.
- ``#SBATCH --partition=owners,normal``: Uses either the ``owners`` or ``normal``
  partition. This is the recommended option for the vast majority of scripts.

Following is a sample of sbatch scripts for requiring more resources to analysis simulation results:

.. code-block:: bash

   #!/usr/bin/bash
   #SBATCH --job-name=analysis_job
   #SBATCH --output=analysis_job.%j.out
   #SBATCH --error=analysis_job.%j.err
   #SBATCH --time=20:00
   #SBATCH --ntasks=1
   #SBATCH --partition=owners,normal
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=64GB

   srun runscripts/container/interactive.sh -i <Your_image_path>  -a -c "python runscripts/analysis.py --config <Your_config_file_path>"

Then, use ``sbatch`` to submit the job:

.. code-block:: bash

   sbatch <Your_Job_Name>.sh

The ``.err`` and ``.out`` files will be created in the same directory as the sbatch script.

Just as with interactive containers, to run scripts directly from your
cloned repository and not the snapshot, add the ``-d`` flag and drop the
``/vEcoli/`` prefix from script names. Note that changing files in your
cloned repository may affect SLURM batch jobs submitted with this flag.

.. _Download Results to Local from Sherlock:

Download Results to Local from Sherlock
=======================================

It's recommended to turn to 
`Sherlock's Data Transfer documentation <https://www.sherlock.stanford.edu/docs/storage/data-transfer/>`_
for details on transferring files to and from your local machine.

Following are common methods ``scp`` and ``rsync``:

``scp`` is convenient for downloading files from the cluster. You can simply execute the following on your **local terminal**:

.. code-block:: bash

   # -r for recursively duplicate the whole repo:
   scp -r <Your_SU_ID>@login.sherlock.stanford.edu:/path/to/remote/folder  /path/to/local/destination

   # If you only want to download single file:
   scp  <Your_SU_ID>@login.sherlock.stanford.edu:/path/to/remote/file  /path/to/local/destination/

In practice, usually we want to get the analytical results for our simulation. 
Due to the report files being HTML files typically, we can turn to shell wildcard and use ``rsync`` with ``include/exclude`` filters:

.. code-block:: bash

   # Recursively downloads all .html files under the specific directory on Sherlock 
   # to your local machine while preserving the subdirectory structure:

   rsync -av --prune-empty-dirs \
     --include='*/' --include='*.html' --exclude='*' \
     <Your_SU_ID>@login.sherlock.stanford.edu:/path/to/remote/folder  /path/to/local/destination

   # --include='*/': Keeps all directories, allowing rsync to traverse into subdirectories
   # --include='*.html': Includes only .html files
   # --exclude='*': Excludes everything else
   # -a: Archive mode (preserves metadata)
   # -v: Verbose output
   # --prune-empty-dirs: Avoids creating empty directories on the local machine

Both ``scp`` and ``rsync`` will require your password and Duo validation.

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
- Python 3.9+
- Git clone vEcoli to a location that is accessible from all nodes in your cluster
- ``out_dir`` under ``emitter_arg`` set to a location that is accessible from all
  nodes in your cluster

If your cluster has Apptainer (formerly known as Singularity) installed,
check to see if it is configured to automatically mount the filesystem of
``out_dir`` (e.g. ``$SCRATCH``). If not, you will need to add ``-B /full/path/to/out_dir``
to the ``containerOptions`` directives in ``runscripts/nextflow/config.template``,
substituting in the absolute path to ``out_dir``. Additionally, you will need to
manually specify the same paths when running interactive containers
(see :ref:`sherlock-interactive`) using the ``-p`` option.

If your cluster does not have Apptainer, you can try the following steps:

1. Completely follow the local setup instructions in the README (install uv, etc).
2. Delete the following lines from ``runscripts/nextflow/config.template``:

.. code-block:: bash

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
``runscripts/nextflow/config.template`` and ``runscripts/nextflow/template.nf``
and all instances of ``--partition=QUEUE(S)`` in :py:mod:`runscripts.workflow`
to the right queue(s) for your cluster. Also, remove the ``--prefer="CPU_GEN...``
``clusterOptions`` in those same files.

If your HPC cluster uses a different scheduler, refer to the Nextflow
`executor documentation <https://www.nextflow.io/docs/latest/executor.html>`_
for more information on configuring the right executor. Beyond the changes above,
you will at least need to modify the ``executor`` directives for the ``sherlock``
and ``sherlock_hq`` profiles in ``runscripts/nextflow/config.template`` and for the
``hqWorker`` process in ``runscripts/nextflow/template.nf``. Additionally, you will
need to replace the SLURM submission directives in :py:func:`runscripts.workflow.main`
with equivalent directives for your scheduler.


.. _hq-info:

----------
HyperQueue
----------

`HyperQueue <https://it4innovations.github.io/hyperqueue/stable/>`_ consists of a
head server and one or more workers allocated by the underlying HPC scheduler. By
configuring the worker jobs to persist for long enough to complete multiple tasks,
HyperQueue reduces the amount of time spent waiting in the queue, which is especially
important for workflows with numerous shorter tasks like ours. We recommend using
HyperQueue for all workflows that span more than a handful of generations.

Internal Logic
==============

If the ``hyperqueue`` option is set to true under the ``sherlock`` key in the
configuration JSON used to run ``runscripts/workflow.py``, the following steps
will occur in order:

#. If ``build_image`` is True, submit a SLURM job to build the container image.
#. Submit a single long-running SLURM job on the dedicated Covert Lab partition
   to run both Nextflow and the HyperQueue head server.
#. Start the HyperQueue head server (initially no workers).
#. Nextflow submits a SLURM job to run the ParCa then another to create variants.
   Both must finish for Nextflow to calculate the maximum number of concurrent
   simulations ``# seeds * # variants``.
#. Nextflow submits SLURM jobs to start ``(# seeds * # variants) // 4`` HyperQueue
   workers, each worker with 4 cores, 16GB RAM, and a 24 hour limit. A
   proportionally smaller worker is potentially created to handle the remainder
   (e.g. for 2 leftover, 2 cores, 8GB RAM, and same 24 hour limit).
#. Nextflow submits simulation tasks to the HyperQueue head server, which schedules
   them on the available workers.
#. Nextflow submits analysis tasks to SLURM directly as they do not hold up the
   workflow and can wait for a bit in the queue. This increases simulation
   throughput by dedicating all HyperQueue worker resources to running simulations.
#. If any HyperQueue worker job terminates with one of three exit codes
   (see :ref:`fault_tolerance`), it is resubmitted by Nextflow to maintain
   the optimal number of workers for parallelizing the workflow.
#. As lineages fail and/or complete, the number of concurrent simulations decreases
   and HyperQueue workers start to go idle. Idle workers automatically terminate
   after 5 minutes of inactivity.
#. Upon completion of the Nextflow workflow, the HyperQueue head server terminates
   any remaining workers and exits.


Monitoring
==========

As long as ``--server-dir`` is given as described below, the ``hq`` command can be
run on any node to monitor the status of the HyperQueue workers and jobs
for a given workflow
(`cheatsheet <https://it4innovations.github.io/hyperqueue/latest/cheatsheet/>`_).

.. code-block:: bash

  # Replace OUTDIR with the output directory and EXPERIMENT_ID with the
  # experiment ID from your configuration JSON.

  # Get HyperQueue JOB_ID from this list of jobs
  hq --server-dir OUTDIR/EXPERIMENT_ID/nextflow/.hq-server job list

  # Get more detailed information about a specific job by ID, including
  # its work directory, runtime, and node
  hq --server-dir OUTDIR/EXPERIMENT_ID/nextflow/.hq-server job info JOB_ID

Updating
========

HyperQueue is distributed as a pre-built binary on GitHub.
Unfortunately, this binary is built with a newer version of GLIBC than the
one available on Sherlock, necessitating a rebuild from source. A binary built
in this way is available in ``$GROUP_HOME/vEcoli_env`` to users with access to
the Covert Lab's partition on Sherlock. This is added to ``PATH`` in the
Sherlock setup instructions, so no further action is required.

Users who want or need to build from source should follow
`these instructions <https://it4innovations.github.io/hyperqueue/stable/installation/#compilation-from-source-code>`_.
