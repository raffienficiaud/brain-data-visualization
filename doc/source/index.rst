.. mpi.is-brain-connectivity-plugin documentation master file, created by
   sphinx-quickstart on Sun Apr 22 10:51:36 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mpi.is-brain-connectivity-plugin's documentation!
============================================================

.. image:: /_static/paraview_preview.png
    :height: 300

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pages/gallery
   pages/installation
   pages/plugin_details
   pages/other_tools
   pages/conversion_tools

This package contains a set of python tools for assisting in the visualization of brain connectivity information.
The package contains:

* a **Paraview filter**, that is a parametrizable *plugin* running in a Paraview visualization pipeline. This
  plugin shows the connectivity of the brain sites, in a raw or summarized fashion,
* utility functions for converting files and performing off-line computation on the brain connectivity

References
----------
This package was developed by *Lennart Bramlage* and *Raffi Enficiaud* at the `Software Workshop` Central Scientific
Facility of the Max Planck Institute for Intelligent Systems, in Tübingen, Germany. The details of the computations
are given in the following report:

* `Design of a visualization scheme for functional connectivity data of Human Brain <LennartBA_>`_, Lennart Bramlage, Bachelor Thesis,
  Hochschule Osnabrück - University of Applied Sciences, 2017.

.. _LennartBA : https://is.tuebingen.mpg.de/publications/bramlage-2017

Installation
------------
The installation procedures are described on the page :doc:`pages/installation`.

How to use the plugin
---------------------
The usage of the plugin is detailed in the page :doc:`pages/plugin_details`.

Utility function
----------------
The additional tools are described in the pages :doc:`pages/other_tools` and :doc:`pages/conversion_tools`.

License
-------
The code contained in this repository is released under the MIT license and copyright is owned by

* Max Planck Society
* Lennart Bramlage
* and Raffi Enficiaud.

See the `LICENSE.txt` file at the root folder of the repository for more information.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
