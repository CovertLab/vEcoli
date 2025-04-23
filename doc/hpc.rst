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

    # Load newer Git, Java (for Nextflow), and PyArrow
    module load system git java/21.0.4 py-pyarrow
    # Include shared Nextflow and HyperQueue installations on PATH
    export PATH=$PATH:$GROUP_HOME/vEcoli_env

.. note::
    To update the versions of the shared dependencies, navigate to
    ``$GROUP_HOME/vEcoli_env`` and run the appropriate command:

    1. Nextflow: ``NXF_EDGE=1 nextflow self-update``
    2. HyperQueue: See :ref:`hyperqueue`.

Then, run the following to test your setup:

.. code-block:: bash

  python3 runscripts/workflow.py --config ecoli/composites/ecoli_configs/test_sherlock.json

This will run a small workflow that:

1. Builds an Apptainer image with a snapshot of your cloned repository and saves it
   to ``$SCRATCH/test_image``.
2. Runs the ParCa.
3. Runs one simulation.
4. Runs the mass fraction analysis.

All output files will be saved to ``$SCRATCH/test_sherlock``.

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
      # you can set this to false to skip building the image.
      "build_image": true,
      # Path (relative or absolute, including file name) of Apptainer image to
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
  containers with a snapshot of the repository.
  
When you run ``python runscripts/workflow.py``, you will get a message
that a SLURM job was submitted to build the image. When that job starts,
you will get terminal output showing the build progress. Avoid making any
changes to your cloned repository until the build has finished.
The built image will contain a snapshot of your cloned repository.
Any changes made to the repository after the container image has been
built will not affect the running workflow.

You can start additional, concurrent workflows that each build a new image
with different modifications to the cloned repository. However, if possible,
we recommend designing your code to accept options through the config JSON
which modify the behavior of your workflow without modifying core code. This
allows you to save time by setting ``build_image`` to false and
``container_image`` to the path of a previously built image.
  

There is a 4 hour time limit on each job in the workflow, including analyses.
This is a generous limit designed to accomodate very slow-dividing cells.
Generally, we recommend that users exclude analysis scripts which take more
than a few minutes from their workflow configuration. Instead, create a
SLURM batch script to run these analyses following :ref:`sherlock-noninteractive`.

.. _sherlock-interactive:

Interactive Container
=====================

To debug a failed job in a workflow, you must locate the container image that was
used for that workflow. You can refer to the ``container_image`` key in the
config JSON saved to the workflow output directory (see :ref:`output`). Start
an interactive container with that image name as follows:

.. code-block:: bash

  runscripts/container/interactive.sh -i container_image -a

.. note::
  Inside the interactive container, you can safely use ``python`` directly
  in addition to the usual ``uv`` commands.

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
cloned repository, navigate to your cloned repository and run the above
command with the ``-d`` flag to start a "development" container.

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

.. _sherlock-noninteractive:

Non-Interactive Container
=========================

To run any script inside a container without starting an interactive session,
use the same command as :ref:`sherlock-interactive` but specify a command
using the ``-c`` flag. For example, to run the ParCa process, navigate to
your cloned repository and run the following command:

.. code-block:: bash

  runscripts/container/interactive.sh -i container_image -c "python /vEcoli/runscripts/parca.py --config {}"

.. note::
  We strongly recommend sticking to running files from the snapshot
  of the repository included in the container image at ``/vEcoli``.
  If you want to run a script from your cloned repository with all
  changes reflected, add the ``-d`` flag and drop the
  ``/vEcoli/`` prefix from the script name.

This is particularly useful for writing
`SLURM batch scripts <https://www.sherlock.stanford.edu/docs/getting-started/submitting/#batch-scripts>`_
to manually run analysis scripts with custom resource requests
(e.g. more than default 4 hours, 1 CPU, 4 GB RAM in workflow).

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
