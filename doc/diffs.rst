==============================
Model Differences From wcEcoli
==============================

This page documents all the known reasons why a simulation workflow run in
vEcoli yields different results from the same workflow run in wcEcoli.
Beyond the differences below, it is important to note that vEcoli uses
newer versions of dependencies (including Python, NumPy, SciPy) compared to
wcEcoli, which may also contribute to differences in simulation results.

This page was written on 2025-10-22 and compares vEcoli commit 931257c with
wcEcoli commit b5128d0.

-----
ParCa
-----

- Reformulated RNA degradation Km optimization as a minimization problem
  and uses Autograd instead of Aesara (not actively maintained). See
  `PR#266 <https://github.com/CovertLab/vEcoli/pull/266>`_ and
  `PR#260 <https://github.com/CovertLab/vEcoli/pull/260>`_.
- Made new gene position data 1-indexed to match other genome sequences
  and ensure the correct RNA sequence. See
  `commit 3a6da7c from PR#306 <https://github.com/CovertLab/vEcoli/pull/306/commits/3a6da7c6ac438ef08d8f056bcbb7a64b6775f310>`_
  and the source code for
  :py:meth:`~reconstruction.ecoli.dataclasses.process.transcription.Transcription._build_cistron_data`.
- Adjusted select gene transcription and translation efficiencies to better match
  experimental data and fix some mechanistic issues. See
  `commit 7035fb7 from PR#306 <https://github.com/CovertLab/vEcoli/pull/306/commits/7035fb78ae64ddbde745688ef5cbbac4b6eda06b>`_,
  `e94e69a from PR#275 <https://github.com/CovertLab/vEcoli/commit/e94e69a22c8d8793af31d5a340d5e3d6c69cdcaa>`_,
  and `b75278c from PR#243 <https://github.com/CovertLab/vEcoli/commit/b75278ce38718778a15e026c7dc0d0d5170675ec>`_.


----------
Simulation
----------

- Converted ribosome footprint size from nucleotides to amino acids.
  See `commit 8d8fa76 from PR#306 <https://github.com/CovertLab/vEcoli/pull/306/commits/8d8fa76a2b1043df282ab0a45401d7525f564599>`_.
- Molecules at the same coordinates as replisomes are now removed. See
  `PR#268 <https://github.com/CovertLab/vEcoli/pull/268>`_.
- Superhelical density calculations can now handle removed replisomes. See
  `PR#269 <https://github.com/CovertLab/vEcoli/pull/269>`_.
- Stored mass data in a slightly different Numpy format causing floating point
  differences. The wcEcoli behavior can be restored by including
  ``{"process_configs": {"ecoli-mass-listener": {"match_wcecoli": true}}}``
  in the simulation config JSON. See
  `commit e562507 <https://github.com/CovertLab/vEcoli/commit/e562507aa2a69f88cc784cbd4185cb16251b5f52>`_.
- Prevented RNAPs from being initialized outside chromosome domain boundaries.
  See `PR#359 <https://github.com/CovertLab/vEcoli/pull/359>`_.
