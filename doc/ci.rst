======================
Continuous Integration
======================

On the vEcoli GitHub website, there is a badge next to certain commits that is
either a red "X" or a green checkmark. Clicking on this badge reveals a set of
continuous integration (CI) tests that are automatically run to ensure the
model continues to work as expected. Some of these tests are run entirely through
a GitHub service called GitHub Actions. Others are run via a Jenkins instance
that is constantly running on the Sherlock cluster.

--------------
GitHub Actions
--------------

The tests that are run through GitHub Actions are configured by the files in
``.github/workflows``. Some of these "tests" can be more accurately described
as maintenance tasks, like publishing the updated documentation website. These
tasks are:

- ``docs_deploy.yml``: ``deploy-docs`` job updates the documentation
  website every time the ``master`` branch changes (push, PR merge)
- ``docs_test.yml``: ``test-docs`` job ensures that the documentation
  builds properly (Sphinx) on PRs and changes to ``master``
- ``pr_tests.yml``: tests that run on PRs and changes to ``master``
  
  - ``Reproducibility`` ensures that two runs of the ParCa and simulation
    with the same settings produce the same output
  - ``two-gens`` runs a two-generation workflow on glucose minimal media
- ``pytest.yml``: code quality and functionality tests that run on PRs and changes
  to master
  
  - ``Pytest`` runs ``pytest`` (all functions with ``test_`` prefix)
  - ``Mypy`` checks types with ``mypy``
  - ``Lint`` checks code style and formatting with ``ruff``

.. tip::
    If you are still actively working on a pull request, add ``[skip ci]``
    on a new line in your commit message to skip the CI tests.

Logs from these tests can be viewed on the GitHub website and you are
required to get all of these tests passing before merging a PR.

We would appreciate if you added tests as part of your pull requests.
This could be as simple as a Python function with the ``test_`` prefix that ensures
the code added or modified in your PR works as intended using a few test cases.

-------
Jenkins
-------

Longer running CI tests that are not compatible with the free tier of GitHub
Actions are run using Jenkins on Sherlock. For information about this setup
(e.g. for troubleshooting or to replicate it on another cluster), please
refer to :ref:`jenkins-setup`.

Each test has an associated Jenkinsfile in ``runscripts/jenkins/Jenkinsfile``
that, in essence, runs one or more workflows using the JSON configuration
files located in ``runscripts/jenkins/configs``. These tests are designed to
run daily (cron) on the latest commit of the ``master`` branch.

The logs for **failed** tests are publicly viewable on the repository website by
clicking on the relevant commit status badges. Upon doing so, you
will be brought to a page with logs and a link to ``View more details on Jenkins vEcoli``.
This link requires access to the Covert Lab's Sherlock partition.


Connecting to Jenkins
=====================

First, connect to Sherlock and run the following command:

.. code-block:: bash

  squeue -p mcovert -o '%.10i %.9P %.50j %.8u %.2t %.10M %.3C %.6D %.20R %k'

.. tip::
  Create an alias ``sqp`` for the command above by adding
  ``alias sqp="{insert command here}"`` to your ``~/.bashrc``.

Look for a job called ``jenkins_new``. Under the ``COMMENT`` column, there
should be a login node of the format ``shXX-lnXX``. Close your SSH connection
and run the following command, replacing as appropriate:

.. code-block:: bash

  ssh {username}@{login node}.sherlock.stanford.edu -L 8080:localhost:8080

With that SSH connection open, the ``View more details on Jenkins vEcoli`` link
will pull up the details of the failing job in the Jenkins interface.

Additionally, you can open the Jenkins web interface by going to
``http://localhost:8080`` in your web browser. From here, you can
see the status of all jobs, start and stop jobs, add and modify jobs,
and do other administrative tasks.

.. note::
  Jenkins is set-up as a persistent job on Sherlock that automatically resubmits
  itself every ~5 days. If this fails, it can be restarted by running ``sbatch
  --mail-user={your email here} runscripts/jenkins/jenkins_vecoli.sh``.

Modifying Tests
===============

Any modifications to the existing Jenkinsfiles in ``runscripts/jenkins/Jenkinsfile``
will modify the behavior of the corresponding tests. To add a new test, you will
need to create a new Jenkinsfile and (with administrator Jenkins privileges)
add a new multibranch pipeline job (see below).


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


.. _jenkins-setup:

Jenkins Setup
=============

.. note::
  This section is intended for people who want to set up their own Jenkins instance
  on a non-Sherlock cluster or for troubleshooting purposes. Members of the Covert Lab
  should already have a functioning Jenkins instance on Sherlock.

The following describes the steps taken to set up Jenkins on Sherlock to run
long continuous integration tests on the ``master`` branch of vEcoli.

Request an interactive session on Sherlock, taking note of the login node. Once
the interactive session is started, run the following command to forward
the port used by Jenkins to the login node:

.. code-block:: bash

    ssh -nNT {username}@{login node} -R 8080:localhost:8080 &

In this same session, download the latest WAR file from the Jenkins website,
load the Java and fontconfig modules, then run Jenkins:

.. code-block:: bash

    wget https://get.jenkins.io/war-stable/latest/jenkins.war
    module load java/17.0.4 fontconfig
    JENKINS_HOME=$GROUP_HOME/jenkins_vecoli java -jar jenkins.war --httpPort=8080

In a new terminal, open a new SSH connection to the previously noted login node
with port forwarding:

.. code-block:: bash

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
Copy ``runscripts/jenkins/jenkins_vecoli.sh`` to the same directory.

Finally, create a directory called ``slurm_logs`` in ``$GROUP_HOME/jenkins_vecoli`` and
``cd`` into it. From here, launch Jenkins with ``sbatch --mail-user={your email here} ../jenkins_vecoli.sh``.
This will queue a persistent Jenkins job that should run indefinitely, resubmitting itself
every 5 days. The stdout and stderr from these jobs will be written to the directory in which
you ran the ``sbatch`` command. Remember to run ``sbatch`` in ``slurm_logs`` to keep all logs
in a consistent location accessible to all members of the lab. You will get an email if any of
these jobs fail, in which case you should review the most recent logs and resubmit with ``sbatch``.
