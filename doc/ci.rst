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

Additionally, we would appreciate if you added tests to improve our test coverage
and improve the likelihood that we catch bugs. This could be as simple as a Python
function with the ``test_`` prefix that ensures some bit of code changed in your
PR works as intended with a few test cases.

-------
Jenkins
-------

Longer running CI tests that are not compatible with the free tier of GitHub
Actions are run using Jenkins on Sherlock. The configuration JSON files for
the workflows that are run during these tests are contained within
``runscripts/jenkins/configs``. Modifying the existing JSON files will modify
the behavior of the tests. However, adding a new JSON file will not add a new
test. That requires additional configuration on the Jenkins side of things
by a member of the Covert Lab with access to Sherlock.

The Jenkins tests are designed to run about once every day (no predictable time)
on the latest commit of ``master``. The tests are:

- ``vEcoli - Anaerobic``: workflow run using ``ecoli-anaerobic.json``
- ``vEcoli - Glucose minimal``: workflow run using ``ecoli-glucose-minimal.json``
- ``vEcoli - Optional features``: workflows run using the remaining JSON files
  except for ``ecoli-with-aa.json`` in succession

Unfortunately, the logs for these tests are not publicly viewable and require
access to Sherlock. If you are part of the Covert Lab Sherlock group, you can
view the logs in two steps. First, SSH into Sherlock and run the following command::

  squeue -p mcovert -o '%.10i %.9P %.50j %.8u %.2t %.10M %.3C %.6D %.20R %k'

.. tip::
  Create an alias ``sqp`` for the command above by adding ``alias sqp="..."``
  to your ``~/.bashrc``.

Look for a job called ``jenkins_new``. Under the ``COMMENT`` column, there
should be a login node of the format ``shXX-lnXX``. Close your SSH connection
and run the following command, replacing as appropriate::

  ssh {username}@{login node}.sherlock.stanford.edu -L 8080:localhost:8080

With that SSH connection open, you should be able to open the logs by clicking
on the commit badges on GitHub.

Administration
==============

.. note::
  This section is intended for members of the Covert Lab responsible for managing
  the Jenkins job on Sherlock.

Jenkins is installed at ``$GROUP_HOME/jenkins_new``. The command to queue up
a 7-day Jenkins instance is ``sbatch $GROUP_HOME/jenkins_new/jenkins_new.sh``.
The submission script is set up to only allow one instance of the job to run at
a time. Therefore, it is safe and strongly recommended that you queue many
Jenkins instances using the command above to minimize downtime.

To open the Jenkins administration portal, follow the steps in the section above
to open an SSH connection to the correct login node, then open ``localhost:8080``
in your web browser. You will need to log in with an account that was granted
administrator privileges by someone else with those privileges.
