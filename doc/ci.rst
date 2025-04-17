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

Logs from these tests can be viewed on the GitHub website and we strongly
recommend that you get all of these tests passing before merging a PR.

When you submit a pull request (PR), a bot will comment with a table showing the current code
coverage of the ``pytest`` suite. As of December 2024, coverage is less than 30%
and even that is misleading because many of the tests are "migration" tests
that compare vEcoli to a snapshot of the original whole cell model. These tests will
be removed in the future as vEcoli is further developed. Additionally, these tests do
not tell us whether the code is working as intended, only that it is working the same
way it worked in the original model. Ideally, we would like to increase test coverage
by adding unit tests which actually test edge cases and ensure the code does what it
is supposed to do.

To that end, we would appreciate if you added tests as part of your pull requests.
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

First, connect to Sherlock and run the following command::

  squeue -p mcovert -o '%.10i %.9P %.50j %.8u %.2t %.10M %.3C %.6D %.20R %k'

.. tip::
  Create an alias ``sqp`` for the command above by adding
  ``alias sqp="{insert command here}"`` to your ``~/.bashrc``.

Look for a job called ``jenkins_new``. Under the ``COMMENT`` column, there
should be a login node of the format ``shXX-lnXX``. Close your SSH connection
and run the following command, replacing as appropriate::

  ssh {username}@{login node}.sherlock.stanford.edu -L 8080:localhost:8080

With that SSH connection open, the ``View more details on Jenkins vEcoli`` link
will pull up the details of the failing job in the Jenkins interface.

Additionally, you can open the Jenkins web interface by going to
``http://localhost:8080`` in your web browser. From here, you can
see the status of all jobs, start and stop jobs, add new jobs
(see :ref:`new-jenkins-jobs`), and do other administrative tasks.

.. note::
  Jenkins is set-up as a persistent job on Sherlock that automatically resubmits
  itself every ~5 days. If this fails, it can be restarted by running ``sbatch
  $GROUP_HOME/jenkins_vecoli/jenkins_vecoli.sh``.

Modifying Tests
===============

Any modifications to the existing Jenkinsfiles in ``runscripts/jenkins/Jenkinsfile``
will modify the behavior of the corresponding tests. To add a new test, you will
need to create a new Jenkinsfile and (with administrator Jenkins privileges)
add a new multibranch pipeline job (see :ref:`new-jenkins-jobs`).
