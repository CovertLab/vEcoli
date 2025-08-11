Welcome to Vivarium *E. coli*'s documentation!
==============================================

Vivarium *E. coli* is a port of the |text|_ to the `Vivarium framework <https://vivarium-collective.github.io>`_.
For more scientific modeling details, refer to the
`documentation <https://github.com/CovertLab/wcEcoli/tree/master/docs/processes>`_
for the original model as well its corresponding publication
(`10.1126/science.aav3751 <https://www.science.org/doi/10.1126/science.aav3751>`_).
This website covers how the model was implemented using Vivarium and describes the user interface
for developing and running the model. For users unfamiliar with Vivarium and its terminology,
we recommend reading `the topic guides <https://vivarium-core.readthedocs.io/en/latest/guides/index.html>`_
in the Vivarium documentation.

..
   Comment: We need to use text substitution because ReST does not
   support nesting italics and hyperlinking

.. _text: https://doi.org/10.1128/ecosalplus.ESP-0001-2020

.. |text| replace:: Covert Lab's *E. coli* Whole Cell Model

.. image:: ./_static/ecoli_master_topology.png
    :width: 100%
    :alt: A graph with blue, database symbol nodes at the top and
       orange, square nodes on the bottom. The blue nodes are connected
       by solid edges, while the orange nodes are connected to the blue
       nodes by broken edges.

.. WARNING::
   This documentation is very much a work in progress. It likely
   contains errors and poor formatting.

.. tip::
   Any text formatted like :py:mod:`~runscripts.workflow` is a clickable link
   to detailed API documentation.

.. toctree::
   :maxdepth: 2

   stores
   processes
   composites
   experiments
   workflows
   output
   tutorial
   docs
   hpc
   gcloud
   ci
   pycharm
   API Reference <reference/api_ref.rst>
