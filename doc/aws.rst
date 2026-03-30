===================
Amazon Web Services
===================

Large vEcoli workflows can be run cost-effectively on Amazon Web Services (AWS).
This section covers setup starting from a fresh account, running workflows, and
handling outputs.

.. tip::
  In all instructions below, "navigate to X console" means to search for
  X in the AWS Console search bar and click on the corresponding service.

-------------------
Fresh Account Setup
-------------------

.. note::
  This section about creating a fresh AWS account is only necessary if starting
  from scratch. If you have access to the Covert Lab GovCloud account, skip to
  :ref:`aws-launch-ec2`.

Create a new AWS account at `AWS Sign Up <https://aws.amazon.com/>`_. Once you
have created your account, navigate to the AWS Console and familiarize yourself
with the following services that vEcoli uses:

- **Batch**: Manages compute resources for running workflow tasks
- **S3 (Simple Storage Service)**: Object storage for workflow outputs
- **EC2 (Elastic Compute Cloud)**: Virtual machines for running Nextflow and workflow tasks
- **ECR (Elastic Container Registry)**: Stores Docker images for workflow tasks
- **ECS (Elastic Container Service)**: Orchestrates Docker containers
- **CloudWatch**: Monitors and logs AWS resources
- **IAM (Identity and Access Management)**: Manages access to AWS resources

The following sections will guide you through a minimal setup for running vEcoli
workflows on a fresh AWS account. Users requiring more advanced configurations
(e.g., custom VPCs, security groups, etc.) should refer to the AWS documentation
for those services.

Installing and Configuring AWS CLI
==================================

Install the AWS CLI following the
`official documentation <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_.

After installation, configure your AWS credentials:

.. code-block:: bash

  aws configure

You will be prompted for:

- AWS Access Key ID
- AWS Secret Access Key
- Default region name (e.g., ``us-west-2``)
- Default output format (``json`` recommended)

.. warning::
    Authentication to Stanford's GovCloud account requires running
    ``aws configure sso`` with our custom start URL (contact admin).
    Set the profile name to ``default`` when prompted to avoid
    having to specify ``--profile <profile name>`` for every AWS CLI command.

.. note::
    We strongly recommend choosing one specific region for all AWS resources to avoid
    unexpected cross-region data transfer costs.

Setting Up Required Services
============================

Create VPC
----------

Navigate to the VPC console and create a new VPC.

1. Under "Resources to create", select "VPC and more".
2. Under "Name tag auto-generation", choose a name. The generated VPC will be
   called ``<your name>-vpc``.
3. We recommend choosing the maximum number of availability zones in your
   chosen region to maximize resource availability.
4. Change the number of private subnets to 0.
5. Set VPC endpoints to None.
6. Leave other settings as default.

Create IAM Roles
----------------

vEcoli workflows require specific permissions. We recommend creating two IAM roles:

1. **Nextflow**: For head VMs that run the Nextflow workflow manager
2. **Batch**: For AWS Batch jobs that run workflow tasks

Before creating the Nextflow role, first create an IAM policy with the required
permissions. To create an IAM policy, navigate to the IAM console, click "Policies"
in the sidebar, and then click "Create policy". Switch to the JSON tab and paste in
`these permissions <https://www.nextflow.io/docs/latest/aws.html#aws-iam-policies>`_
under the ``Action`` list. In addition to those permissions, paste the following
permissions to let :py:mod:`runscripts.workflow` build and push vEcoli Docker
images to ECR using ``runscripts/container/build-and-push-ecr.sh``:

- ``ecr:CompleteLayerUpload``
- ``ecr:CreateRepository``
- ``ecr:InitiateLayerUpload``
- ``ecr:PutImage``
- ``ecr:UploadLayerPart``

Nextflow also requires access to S3, which must be granted as
`described here <https://www.nextflow.io/docs/latest/aws.html#s3-policies>`_.
The easiest way to do this is to grant full S3 access to the Nextflow policy
by pasting ``s3:*`` to the list of allowed actions.

Then, add ``"*"`` to the ``Resource`` list to make the allowed actions defined
in this permission policy valid for all AWS resources. Click "Next", give the
policy a name and description, and create the policy.

After creating the policy, create a new IAM role for Nextflow by clicking "Roles"
in the sidebar and then "Create role". Select "AWS service" as the trusted entity
type and "EC2" as the use case. Click "Next", then attach the policy you just created.
Click "Next", give the role a name and description, and create the role.

For the Batch role, create a new IAM role as described above but attach the
AWS-managed policies ``AmazonS3FullAccess`` and ``AmazonEC2ContainerServiceforEC2Role``
(`see here <https://www.nextflow.io/docs/latest/aws.html#get-started>`_ for more details).

Setup Batch
-----------

AWS Batch manages the compute resources for running vEcoli workflows. Follow
these steps to set it up:

1. Navigate to the AWS Batch console
2. Click "Environments" in the sidebar, then "Create environment --> Compute environment"
2. Use the following settings to create the new compute environment:

   - Environment type: "Amazon Elastic Compute Cloud (Amazon EC2)"
   - Name: Choose a name
   - Orchestration type: "Managed"
   - Instance role: Select the Batch role you created earlier
   - Use EC2 Spot instances: True (optional, but strongly recommended for cost savings)
   - Maximum vCPUs: Set based on maximum # of lineages you want to run in parallel
     (AWS Batch will create and terminate VM instances as needed to meet demand up
     to this limit)
   - Allowed instance types: See below for recommendations
   - VPC ID: Select the VPC you created earlier
   - Subnets: Select all subnets in that VPC
   - Leave other settings as default

3. Create a Job Queue:

   - Orchestration type: "Amazon Elastic Compute Cloud (Amazon EC2)"
   - Name: Choose a name. This is the name you will use for the
     ``batch_queue`` key in your config JSON.
   - Connected compute environments: Select your compute environment

We strongly recommend using the latest generation of general-purpose Graviton EC2
instances (``M8g`` as of Feb 2026). These instances offer excellent price-performance
for vEcoli workflows, which are CPU-bound and benefit from the lack of hyperthreading
on Graviton processors. They also offer 4 GiB of memory per vCPU, which is the default
memory/CPU allocation for each simulation.

.. warning::
    Do not mix Graviton (ARM) and non-Graviton (x86) instances in the same compute
    environment. If you choose Graviton instances, make sure to use a Graviton
    instance (e.g., ``t4g.medium``) for your head node as well (see below).

.. tip::
    To retrieve the latest price/physical CPU for different instance types,
    run ``uv run runscripts/cloud_pricing/aws.py --region <your region>``.
    Benchmark workflow performance on different instance types to find the best
    price-performance for your specific workflow configuration.
    
.. _aws-launch-ec2:

----------------------
Launch an EC2 Instance
----------------------

Create an Instance
==================

Create a small EC2 instance to run Nextflow. Navigate to the EC2 console and launch
a new instance with:

- An Amazon Linux 2023 Amazon Machine Image (AMI), pick "64-bit (ARM)" architecture
  if using Graviton instances or "64-bit (x86)" if using non-Graviton instances
- Instance type: Min. 4 GiB memory, must match Batch compute environment CPU
  architecture, try ``t4g.medium`` (Graviton) or ``t3.medium`` (non-Graviton)
- Key pair: Create a new key pair or use an existing one to SSH into the instance
- Network: Select the VPC you created earlier
- Security group: Create a new one allowing SSH traffic from your IP only
- Storage: 30 GiB gp3
- Under Advanced details, set IAM instance profile to the Nextflow role you created
  (``ECR`` for the Stanford GovCloud account).

.. note::
  Run ``chmod 400 /path/to/your-key.pem`` on your private key file to ensure
  it has the correct permissions for SSH.

.. warning::
  Remember to stop your EC2 instance when your workflow finishes to avoid
  unnecessary charges.

Connect to Your Instance
========================

SSH into your newly created EC2 instance using the private key from above:

.. code-block:: bash

  ssh -i /path/to/your-key.pem ec2-user@<instance-public-dns>

.. _aws-deps:

Install Dependencies
====================

On the EC2 instance, install Git, Docker, and Java:

.. code-block:: bash

  # Update package manager
  sudo yum update -y

  # Install Git, Java (required for Nextflow), and Docker
  sudo yum install -y git java docker

  # Start Docker service and enable on boot
  sudo systemctl start docker
  sudo systemctl enable docker

  # Add your user to the docker group
  sudo usermod -aG docker $USER

  # Set AWS CLI default region (us-gov-west-1 for Stanford GovCloud)
  aws configure set region <your-region>

  # Log out and back in for group changes to take effect

Clone the vEcoli repository:

.. code-block:: bash

  git clone https://github.com/CovertLab/vEcoli.git --filter=blob:none
  cd vEcoli

`Install uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then
create a new virtual environment and install S3FS:

.. code-block:: bash

  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.bashrc
  uv venv
  uv pip install s3fs boto3

Run the following to automatically activate the virtual environment:

.. code-block:: bash

  echo "source ~/vEcoli/.venv/bin/activate" >> ~/.bashrc
  source ~/.bashrc

Finally, `install Nextflow <https://www.nextflow.io/docs/latest/install.html>`_:

.. code-block:: bash

  curl -s https://get.nextflow.io | bash
  sudo mv nextflow /usr/local/bin/
  chmod +x /usr/local/bin/nextflow

--------------
Setup Workflow
--------------

Create S3 Bucket
================

vEcoli workflows use S3 for output storage. Create a new S3 bucket using
the AWS Console or AWS CLI:

.. code-block:: bash

  aws s3 mb s3://your-vecoli-bucket-name

Replace ``your-vecoli-bucket-name`` with a globally unique bucket name.

.. danger::
  Do NOT use underscores in your bucket name. Use hyphens instead.
  Bucket names must be DNS-compliant.

Configure Your Workflow
=======================

Tell vEcoli to use your S3 bucket by setting the ``out_uri`` key under the
``emitter_arg`` key in your config JSON (see :ref:`json_config`). The URI
should be in the form ``s3://your-vecoli-bucket-name``. Remember to remove
the ``out_dir`` key under ``emitter_arg`` if present.

On AWS, each job in a workflow (ParCa, sim 1, sim 2, etc.) is run using
Docker containers managed by AWS ECS. vEcoli uses a build script to create
and push Docker images to Amazon ECR.

.. note::
  Files that match the patterns in ``.dockerignore`` are excluded from the
  Docker image.

The following configuration keys, in addition to the ``out_uri`` key under
``emitter_arg``, are **REQUIRED** to run :py:mod:`runscripts.workflow` on AWS:

.. code-block::

  {
    "aws": {
      # Boolean, whether to build a fresh Docker image. If files that are
      # not excluded by .dockerignore did not change since your last build,
      # you can set this to false to skip building the image.
      "build_image": true,
      # Name of Docker image to build (or use directly, if build_image is false)
      "container_image": "vecoli-workflow",
      # AWS region (optional, defaults to us-gov-west-1)
      "region": "us-west-2",
      # AWS Batch job queue name (optional, defaults to "vecoli")
      "batch_queue": "vecoli"
    }
  }

.. tip::
  We strongly recommend setting ``progress_bar`` to ``false`` in your config JSON
  when running workflows on AWS to reduce the amount of generated logs,
  which are billed as `described here <https://aws.amazon.com/cloudwatch/pricing/>`_.

Build and Push Image
====================

The build process is handled automatically when you launch a workflow with
``build_image: true``. However, you can also manually build and push images
using the ``runscripts/container/build-and-push-ecr.sh`` script.

-----------------
Running Workflows
-----------------

After setting the required options in your configuration JSON, use ``screen`` or ``tmux``
to open a virtual console that will persist after your SSH connection is closed.
In that console, invoke :py:mod:`runscripts.workflow` as normal to start a workflow::

  python runscripts/workflow.py --config your_config.json

.. note::
  Unlike workflows run locally, AWS workflows are run using containers with a
  snapshot of the repository at the time the workflow was launched. This means
  that any changes made to the repository after launching a workflow will not
  be reflected in that workflow.

Once your workflow has started, you can press "Ctrl+A D" (for screen) or
"Ctrl+B D" (for tmux) to detach from the virtual console and close your SSH
connection. The EC2 instance must continue to run until the workflow is complete.
You can SSH into your instance and reconnect to the virtual terminal with
``screen -r`` or ``tmux attach`` to monitor progress.

.. warning::
  AWS Batch Spot instances can be interrupted at any time. Analysis scripts that
  take more than a few hours to run should be excluded from workflow configurations
  and manually run using :py:mod:`runscripts.analysis` afterwards. If you require
  guaranteed compute, modify your Batch compute environment to not use Spot instances.

----------------
Handling Outputs
----------------

Once a workflow is complete, all outputs should be in your S3 bucket at the
URI specified in the ``out_uri`` key under ``emitter_arg`` in the configuration
JSON.

We strongly discourage users from downloading large amounts of data from S3,
as that will incur significant data transfer charges. Instead, run analyses
on an EC2 instance in the same region as your S3 bucket - this avoids data
transfer fees.

Data stored in S3 incurs charges based on:

1. **Storage amount**: Costs vary by region and storage class
2. **Storage duration**: Charges are prorated
3. **Data transfer**: Transfers out of AWS or between regions incur charges
4. **Request costs**: GET, PUT, etc. have per-request costs

Storing terabytes of simulation data can cost $1000+/year. For cost
management:

- Delete workflow output data from S3 as soon as you finish your analyses
- Consider using S3 Lifecycle policies to automatically move data to cheaper
  storage classes (e.g., S3 Glacier) after a certain period
- Use S3 Intelligent-Tiering for automatic cost optimization
- Run analyses on EC2 instances in the same region as your S3 bucket

If necessary, it is likely cheaper to re-run the workflow to regenerate data
later than to keep it around long-term.

.. _aws-interactive-containers:

--------------------------
AWS Interactive Containers
--------------------------

Since all steps of the workflow are run inside Docker containers, it can be
helpful to launch an interactive instance of the container for debugging. This
is also useful for running standalone analyses on workflow outputs.

For simplicity, we recommend reusing the same EC2 instance that you created to
launch workflows. If you need more compute power (e.g., to run ad-hoc analyses),
you can change the instance type to a more powerful one following
`these instructions <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-resize.html>`_.
Make sure to choose an instance type that matches the CPU architecture of
your Docker image (Graviton/ARM vs. non-Graviton/x86) and remember to change
it back to a smaller instance when done to save costs.

If you need more storage, create and attach a new EBS volume following
`these instructions <https://docs.aws.amazon.com/ebs/latest/userguide/ebs-attaching-volume.html>`_.
Then, follow
`these instructions <https://docs.aws.amazon.com/ebs/latest/userguide/ebs-using-volumes.html>`_
to make the new volume available for use.

From inside your cloned repository on your EC2 instance, run:

.. code-block:: bash

  runscripts/container/interactive.sh -r aws -i container_image

``container_image`` should match the name in your config JSON (e.g.,
``vecoli-workflow``). A copy of the config JSON should be saved to your
S3 bucket with the other output for reference (see :ref:`output`).

.. note::
  Inside the interactive container, you can use ``python`` or ``ipython``
  directly in addition to the usual ``uv`` commands.

Inside the container, navigate to ``/vEcoli`` and add breakpoints as you see fit.
Note the working directory (see :ref:`troubleshooting`) of the Nextflow task you
want to debug. Download the ``.command.run`` file for your task from S3 to a
temporary debug directory and run it:

.. code-block:: bash

    mkdir debug
    cd debug
    aws s3 cp s3://your-vecoli-bucket-name/path/to/workdir/.command.run .
    chmod +x .command.run
    ./command.run

.. warning::
  Any changes that you make to ``/vEcoli`` inside the container are discarded
  when the container terminates.

The files located in ``/vEcoli`` are a copy of your cloned repository (excluding
files ignored by ``.dockerignore``) at the time the workflow was launched.
To start an interactive container that reflects the current state of your
cloned repository, add the ``-d`` flag to start a "development" container.
