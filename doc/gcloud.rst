============
Google Cloud
============

Large vEcoli workflows can be run cost-effectively on Google Cloud. This section
covers setup starting from a fresh project, running workflows, and handling outputs.

Members of the Covert Lab should skip to the `Create Your VM`_ section for setup.

-------------------
Fresh Project Setup
-------------------

Create a new project for vEcoli using `this link <https://console.cloud.google.com/projectcreate>`_.
Choose any name that you like and you should be brought to the Google Cloud
console dashboard for your new project. Use the top search bar to find
the following APIs and enable them:

- Compute Engine
- Cloud Build
- Artifact Registry

You will be asked to link a billing account at this time.

.. tip:: 
  If you are new to Google Cloud, we recommend that you take some time to
  familiarize yourself with the Cloud console after enabling the above APIs.

Set a default region and zone for Compute Engine following
`these instructions <https://cloud.google.com/compute/docs/regions-zones/changing-default-zone-region#console>`_.
This avoids unnecessary charges for multi-region data availability and access,
improves latency, and is required for some of vEcoli's code to work.

Create a new repository in Artifact Registry following the steps
on `this page <https://cloud.google.com/artifact-registry/docs/repositories/create-repos>`_.
Make sure to name the repository ``vecoli`` and create it in the same
region as your Compute Engine default. This is where the Docker images
used to run the workflow will be stored (see `Build Docker Images`_).

The Compute Engine VMs that vEcoli spawns to run workflow jobs do not
have external IP addresses (no internet access) but need access to
Google Cloud APIs. Follow the instructions on
`this page <https://cloud.google.com/vpc/docs/configure-private-google-access#enabling-pga>`_
to turn on Private Google Access for these VMs. For a fresh project, you
can click on the ``default`` network, then under the "Subnets"
tab, click on the subnet for your Compute Engine default region.

Compute Engine VMs come with `service accounts <https://cloud.google.com/compute/docs/access/service-accounts>`_
allow users to control access to project resources (compute, storage, etc.).
To run vEcoli workflows, only a small subset of the default
service account permissions are necessary. For that reason, we strongly
recommend that users either modify the default Compute Engine service
account permissions or create a dedicated vEcoli service account.

Using the `Google Cloud console <https://console.cloud.google.com>`_,
navigate to the "IAM & Admin" panel. You can edit the Compute Engine default
service account on this page by clicking the pencil icon in the corresponding row.
To create a new service account, click the "Service Accounts" tab in the side bar
and then "Create Service Account". Once you get to a section where you
can assign roles to the service account, assign the following set of roles:

  - Artifact Registry Writer
  - Batch Agent Reporter
  - Batch Job Editor
  - Cloud Build Editor
  - Compute Instance Admin (v1)
  - Logs Writer
  - Monitoring Metric Writer
  - Service Account User
  - Storage Object Admin
  - Viewer

If you created a dedicated service account, keep its associated email handy
as you will need it to create your VM in the next step.

--------------
Create Your VM
--------------

Click on the terminal shell icon near the top-right corner of the
`Cloud console <https://console.cloud.google.com>`_. Run the command
``gcloud init`` and choose to reinitialize your configuration. Choose
the right account and project, allowing ``gcloud`` to pull in the
project's default Compute Engine zone and region.

.. tip:: 
  We are using the Cloud Shell built into Cloud console for convenience.
  If you would like to do this locally, you can install the ``gcloud``
  CLI on your machine following `these steps <https://cloud.google.com/sdk/docs/install>`_.

Once done, run the following to create a Compute Engine VM to run your workflows,
replacing ``INSTANCE_NAME`` with a unique name of your choosing and ``SERVICE_ACCT``
as described below::

  gcloud compute instances create INSTANCE_NAME \
    --shielded-secure-boot \
    --machine-type=e2-medium \
    --scopes=cloud-platform \
    --service-account=SERVICE_ACCT

If you created a new service account earlier in the setup process, substitute
the email address for that service account. If you are a member of the Covert Lab
or have been granted access to the Covert Lab project, substitute
``fireworker@allen-discovery-center-mcovert.iam.gserviceaccount.com``. Otherwise,
including if you edited the default service account permissions, run
the above command without the ``--service-account`` flag.

.. warning:: 
  Remember to stop your VM when you are done using it. You can either do this
  through the Cloud console or by running ``gcloud compute instances stop INSTANCE_NAME``.
  You can always restart the instance when you need it again and your files will
  persist across sessions.

SSH into your newly created VM (if connection error, wait a moment, then retry)::

  gcloud compute ssh INSTANCE_NAME

Now, on the VM, initialize ``gcloud`` by running ``gcloud init`` and selecting the
right service account and project. Next, install Git and clone the vEcoli repository::

  sudo apt update && sudo apt install git
  git clone https://github.com/CovertLab/vEcoli.git

Now follow the installation instructions from the README starting with
installing ``uv`` and finishing with installing Nextflow.

.. note::
  The only requirements to run :mod:`runscripts.workflow` on Google Cloud
  are Nextflow and PyArrow. The workflow steps will be run inside Docker
  containers (see :ref:`docker-images`). The other Python requirements can be
  omitted for a more minimal installation. You will need to use
  :ref:`interactive containers <interactive-containers>` to run the model using
  any interface other than :mod:`runscripts.workflow`, but this may be a good
  thing for maximum reproducibility.

------------------
Create Your Bucket
------------------

vEcoli workflows persist their final outputs to a Cloud Storage
bucket. To create a bucket, follow the steps on
`this page <https://cloud.google.com/storage/docs/creating-buckets>`_. By default,
buckets are created in the US multi-region. We strongly recommend changing this to
the same single region as your Compute Engine default (``us-west1`` for Covert Lab).
All other settings can be kept as default.

.. danger:: 
  Do NOT use underscores or special characters in your bucket name. Hyphens are OK.

Once you have created your bucket, tell vEcoli to use that bucket by setting the
``out_uri`` key under the ``emitter_arg`` key in your config JSON (see :ref:`json_config`).
The URI should be in the form ``gs://{bucket name}``. Remember to remove the ``out_dir``
key under ``emitter_arg`` if present.

.. _docker-images:

-------------------
Build Docker Images
-------------------

On Google Cloud, each job in a workflow (ParCa, sim 1, sim 2, etc.) is run
on its own temporary VM. To ensure reproducibility, workflows run on Google
Cloud are run using Docker containers. vEcoli contains scripts in the
``runscripts/container`` folder to build the required Docker images from the
current state of your repository, with the built images being automatically
uploaded to the ``vecoli`` Artifact Registry repository of your project.

- ``build-runtime.sh`` builds a base Docker image containing the Python packages
  necessary to run vEcoli as listed in ``uv.lock``
- ``build-wcm.sh`` builds on the base image created by ``build-runtime.sh`` by copying
  the files in the cloned vEcoli repository, honoring ``.gitignore``

.. tip:: 
  If you want to build these Docker images for local testing, you can run
  these scripts locally with ``-l`` as long as you have Docker installed.

These scripts are mostly not meant to be run manually. Instead, users should let
:py:mod:`runscripts.workflow` handle image builds by setting the following
keys in your configuration JSON::

  {
    "gcloud": {
      # Name of image build-runtime.sh built/will build
      "runtime_image_name": ""
      # Boolean, can put false if uv.lock did not change since the last
      # time a workflow was run with this set to true
      "build_runtime_image": true,
      # Name of image build-wcm.sh built/will build
      "wcm_image_image": ""
      # Boolean, can put false if nothing in repository changed since the
      # last time a workflow was run with this set to true
      "build_wcm_image": true
    }
  }

These configuration keys, in addition to the ``out_uri`` key under ``emitter_arg``,
are necessary and sufficient to tell :py:mod:`runscripts.workflow` that you intend to
run the workflow on Google Cloud. After setting these options in your configuration JSON,
you can use ``screen`` to open a virtual console that will persist even after your SSH
connection is closed. In that virtual console, invoke :py:mod:`runscripts.workflow`
as normal to start your workflow::
  
  python runscripts/workflow.py --config {}

Once your workflow has started, you can use press "ctrl+a d" to detach from the
virtual console then close your SSH connection to your VM. The VM must continue
to run until the workflow is complete. You can SSH into your VM and reconnect to
the virtual terminal with ``screen -r`` to monitor progress or inspect the file
``.nextflow.log`` in the root of the cloned repository.

.. warning::
  While there is no strict time limit for workflow jobs on Google Cloud, jobs
  can be preempted at any time due to the use of spot VMs. Analysis scripts that
  take more than a few hours to run should be excluded from workflow configurations
  and manually run using :py:mod:`runscripts.analysis` afterwards. Alternatively, if
  you are willing to pay the significant extra cost for standard VMs, delete
  ``google.batch.spot = true`` from ``runscripts/nextflow/config.template``.

----------------
Handling Outputs
----------------

Once a workflow is complete, all of the outputs should be contained within the Cloud
Storage bucket at the URI in the ``out_uri`` key under ``emitter_arg`` in the
configuration JSON. We strongly discourage users from trying to download this data,
as that will incur significant egress charges. Instead, you should use your VM to run
analyses, avoiding these charges as long as your VM and bucket are in the same region.

Data stored in Cloud Storage is billed for the amount of data and how long it is stored
(prorated). Storing terabytes of simulation data on Cloud Storage can cost upwards of
$1,000/year, dwarfing the cost of the compute needed to generate that data. For that
reason, we recommend that you delete workflow output data from your bucket as soon as
you are done with your analyses. If necessary, it will likely be cheaper to re-run the
workflow to regenerate that data later than to keep it around.

.. _interactive-containers:

----------------------
Interactive Containers
----------------------

.. warning::
  Install
  `Docker <https://docs.docker.com/engine/install/>`_ and
  `Google Cloud Storage FUSE <https://cloud.google.com/storage/docs/cloud-storage-fuse/install>`_
  on your VM before continuing.

Since all steps of the workflow are run inside Docker containers, it can be
helpful to launch an interactive instance of the container for debugging.

To do so, run the following command::
  
  runscripts/container/interactive.sh -w wcm_image_name -b bucket

``wcm_image_name`` should be the same ``wcm_image_name`` from the config JSON
used to run the workflow. A copy of the config JSON should be saved to the Cloud
Storage bucket with the other output (see :ref:`output`). ``bucket`` should be
the Cloud Storage bucket of the output (``out_uri`` in config JSON).

Inside the container, add breakpoints to any Python files located at ``/vEcoli`` by
inserting::
  
  import ipdb; ipdb.set_trace()

Navigate to the working directory (see :ref:`troubleshooting`) of the failing
task at ``/mnt/disks/{bucket}/...``. Evoke ``bash .command.sh`` to run the
task. Execution should pause at your set breakpoints, allowing you to inspect
variables and step through the code.

.. warning::
  Any changes that you make to the code in ``/vEcoli`` inside the container are not
  persistent. For large code changes, we recommend that you navigate to ``/vEcoli``
  inside the container and run ``git init`` then
  ``git remote add origin https://github.com/CovertLab/vEcoli.git``. With the
  git repository initialized, you can make changes locally, push them to a
  development branch on GitHub, and pull/merge them in your container.

---------------
Troubleshooting
---------------

Cloud Storage Permission Issue
==============================

If you are trying to launch a cloud workflow or access cloud
output (e.g. run an analysis script) from a local computer, you
may encounter an error like the following::

  HttpError: Anonymous caller does not have storage.objects.list access to the Google Cloud Storage bucket. Permission 'storage.objects.list' denied on resource (or it may not exist)., 401

We do not recommend using local computers to launch
cloud workflows because that would require the computer to be on and connected
to the internet for the entire duration of the workflow. We STRONGLY discourage
using a local computer to run analyses on workflow output saved in
Cloud Storage as that will incur hefty data egress charges.

Instead, users should stick to launching workflows and running analysis scripts
on Compute Engine VMs. Small VMs are fairly cheap to keep running for the duration
of a workflow, and larger VMs can be created to leverage DuckDB's multithreading
for fast reading of workflow outputs stored in Cloud Storage. Assuming the VMs are
in the same region as the Cloud Storage bucket being accessed, no egress charges
will be applied, resulting in much lower costs.

If you absolutely must interact with cloud resources from a local machine, the above
error may be resolved by running the following command to generate credentials that
will be automatically picked up by PyArrow::

  gcloud auth application-default login

