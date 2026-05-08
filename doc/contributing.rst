============
Contributing
============

This guide covers the recommended workflow for contributing to Vivarium *E. coli*,
including opening issues for public comment, forking the repository, configuring
commit signing, managing dependencies with ``uv``, and understanding version-pinned
tooling.

.. _open-issue:

-----------------------------------------
Opening an Issue Before Starting Work
-----------------------------------------

Before writing any code, we encourage you to open a GitHub issue to propose
your change or bug fix. This allows lab members and other contributors to
comment on the idea, flag potential conflicts with ongoing work, suggest
alternative approaches, and confirm that the change is within scope before
you invest time implementing it.

To open an issue:

1. Go to the `issue tracker <https://github.com/CovertLab/vEcoli/issues>`_ and
   click **New issue**.
2. Describe the problem or proposed change clearly:

   - For **bug reports**: include steps to reproduce, the observed behaviour,
     and the expected behaviour.
   - For **feature requests or refactors**: describe the motivation, the
     proposed solution, and any alternatives you considered.

3. Submit the issue and allow time for feedback before opening a PR.

.. note::
   For very small, obviously correct fixes (e.g. typos in documentation),
   you may open a PR directly without a preceding issue.

.. _commit-signing:

-----------------------------
Setting Up Commit Signing
-----------------------------

Signed commits prove that a commit was authored by you and have not been
tampered with. GitHub displays a **Verified** badge next to signed commits.
vEcoli requires all commits to be signed for better security and traceability.
We recommend using SSH key signing as the simplest option if you already
use SSH to authenticate with GitHub.

.. tip::
   For a full overview of commit signing options (SSH keys, GPG keys, and
   S/MIME certificates), see the
   `GitHub documentation on signing commits
   <https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification>`_.

Basic Setup with SSH Keys
-------------------------

This approach reuses your existing SSH authentication key for signing, so no
additional key material needs to be generated.

1. **Generate an SSH key** (skip if you already have one)

   See the `GitHub guide on generating an SSH key
   <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`_.
   Ed25519 keys are recommended:

   .. code-block:: console

      ssh-keygen -t ed25519 -C "your_email@example.com"

2. **Add the key to GitHub as a *Signing* key**

   In addition to (or instead of) adding the key as an *Authentication* key,
   go to **Settings → SSH and GPG keys → New SSH key**, paste your public key,
   and set the **Key type** to **Signing Key**.
   See `Adding a new SSH key to your GitHub account
   <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_
   for step-by-step instructions.

3. **Configure Git to use SSH for signing**

   .. code-block:: console

      git config --global gpg.format ssh
      git config --global user.signingkey -/.ssh/id_ed25519.pub

4. **Enable automatic commit signing**

   .. code-block:: console

      git config --global commit.gpgsign true

   With this set, every ``git commit`` will be signed automatically — no
   ``-S`` flag needed.

   If you use Visual Studio Code, you may need to enable commit signing in the
   Git extension settings:

   1. Open VS Code settings (File → Preferences → Settings).
   2. Search for "Git: Enable Commit Signing" and check the box to enable it.
   3. Restart VS Code to apply the change.

5. **Verify a signed commit**

   After making a commit and pushing, visit the commit on GitHub and confirm
   that the **Verified** badge is present.

.. _fork-repo:

------------------------
Forking the Repository
------------------------

.. note::
   If you are a member of the Covert Lab with push access to this repository,
   you can create branches directly on the main repository instead of forking.

External contributors should work from a personal fork rather than pushing
branches directly to ``CovertLab/vEcoli``.

1. Navigate to https://github.com/CovertLab/vEcoli and click **Fork** in the
   top-right corner to create a copy under your own GitHub account.

2. Clone your fork locally:

   .. code-block:: console

      git clone https://github.com/<your-username>/vEcoli.git
      cd vEcoli

3. Add the upstream repository as a remote so you can pull in future changes:

   .. code-block:: console

      git remote add upstream https://github.com/CovertLab/vEcoli.git

4. To keep your fork up to date:

   .. code-block:: console

      git fetch upstream
      git merge upstream/master

5. Create a new branch for your changes:

   .. code-block:: console

      git checkout -b my-feature

6. When your branch is ready, open a Pull Request (PR) from your fork's branch
   against ``CovertLab/vEcoli:master`` on GitHub.

.. _pr-review:

--------------------------
Pull Request Review Process
--------------------------

Once you have opened a Pull Request against ``CovertLab/vEcoli:master``,
a lab member will review it when they are available. Reviews are done on a
best-effort basis alongside research and other lab responsibilities, so please
allow some time before following up.

Before Requesting Review
-------------------------

Make sure your branch and commits meet the following requirements before
marking the PR as ready for review:

1. **All commits must be verified.** Every commit on your branch must show the
   **Verified** badge on GitHub, meaning it is signed with a valid SSH or GPG
   key. See :ref:`commit-signing` for setup instructions. Unverified commits
   will block merging.

2. **Your branch must be up to date with** ``master``. Rebase or merge the
   latest ``master`` into your branch before requesting review to minimise
   conflicts and ensure CI runs against current code:

   .. code-block:: console

      git fetch upstream
      git rebase upstream/master

The ``long ci`` Label
---------------------

The two longest-running GitHub Actions tests — ``Reproducibility`` and
``two-gens`` — only run when a PR carries the ``long ci`` label (see
:ref:`ci` for details). These tests consume a significant number of CI
minutes, so please observe the following policy:

- **Do not** add the ``long ci`` label when you first open the PR.
- **Wait** until you have received at least one round of review and addressed
  any feedback.
- **Then** add the ``long ci`` label to trigger the full test suite before
  the PR is approved and merged.

See the GitHub documentation on
`managing labels <https://docs.github.com/en/issues/using-labels-and-milestones-to-track-work/managing-labels#applying-a-label>`_
for instructions on how to apply a label to a PR.

.. _uv-usage:

------------------------------------
Managing Dependencies with ``uv``
------------------------------------

This project uses uv for Python environment and dependency management.
The authoritative list of dependencies lives in ``pyproject.toml``; the
locked versions are recorded in ``uv.lock``. Refer to the
`uv documentation <https://docs.astral.sh/uv/>`_ 
for more details on how uv works and best practices.

Always use ``uv`` commands to modify dependencies rather than editing
``pyproject.toml`` by hand, so that ``uv.lock`` stays in sync. When
``uv.lock`` is updated, uv commands like ``uv run`` will automatically
sync the environment to match the locked versions.

Running Scripts
---------------

.. note::
    If you are on an HPC cluster or cloud environment, run directly
    with Python instead of ``uv`` after following the relevant
    setup instructions (see :ref:`sherlock`, :doc:`gcloud`, or :doc:`aws`).

Use the ``uvenv`` alias (set up during installation) to run any script inside
the managed environment with the correct environment variables loaded from
``.env``:

.. code-block:: console

   uvenv python runscripts/workflow.py --config configs/my_config.json

Alternatively, ``uv run`` directly:

.. code-block:: console

   uv run --env-file .env python runscripts/workflow.py

Adding a New Package
--------------------

To add a runtime dependency:

.. code-block:: console

   uv add <package>

To add a development-only dependency (tests, linting, etc.):

.. code-block:: console

   uv add --optional dev <package>

Both commands update ``pyproject.toml`` and regenerate ``uv.lock``. Commit
both files together.

Upgrading a Package
-------------------

To upgrade a single package to its latest allowed version:

.. code-block:: console

   uv lock --upgrade-package <package>
   uv sync --extra dev

To upgrade all packages at once (use with caution):

.. code-block:: console

   uv lock --upgrade
   uv sync --extra dev

After upgrading, run the test suite to check for regressions before committing
the updated ``uv.lock``.

.. _nextflow-pinning:

------------------------------
Nextflow Version Pinning
------------------------------

The Nextflow version used by this project is pinned via the ``NXF_VER``
environment variable in the ``.env`` file at the repository root:

.. code-block:: text

   # .env
   NXF_VER=25.10.4

When you run workflow scripts through ``uv run --env-file .env`` (i.e. via
the ``uvenv`` alias), Nextflow automatically reads ``NXF_VER`` and uses the
specified version, downloading it if necessary.

When running scripts **directly with Python** (as is common on HPC clusters
or cloud environments where ``uv`` may not be available),
:py:mod:`runscripts.workflow` loads ``.env`` itself at startup using a small
built-in parser, so ``NXF_VER`` is still injected into the process environment
before Nextflow is invoked.

.. note::
   If you need to upgrade Nextflow, update the ``NXF_VER`` value in ``.env``
   and commit the change.
