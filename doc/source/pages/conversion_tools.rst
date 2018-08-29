File conversion utilities
=========================

.. contents::
  :local:

.. _nifti_edge_file_generating:

Connectivity file from Nifti
----------------------------
The set of connection between two different sites of the brain is contained in a **edge file** file, that makes it
convenient to read in the visualization filter in Paraview.

Generating the edge file is done through the command ``generate_brain_connectivity_edge_file`` and requires:

* the ``nifti`` file that can be read with ``nibabel``,
* the ``csv`` file encoding the connections between sites in the brain,
* the output file.

.. warning::

  This functionality requires the installation of the ``nibabel`` python package. Please make sure you followed
  the :doc:`installation` procedures.


End-points volume image
-----------------------
The 3D volume containing all the end-points of the connexels can be generated from the command line in the same
manner as for the connectivity file. The generated file is in the VTK format.

.. warning::

  Requires ``nibabel`` (see above) and ``VTK`` (see installation instructions).

Volume image
------------
The 3D volumetric image may be generated from the command line with the command ``generate_brain_connectivity_volume_file``.
This call is a wrapper
around the function :py:func:`.generate_volume_file`.

.. _clusters_generating:

Clustering
----------
A script for performing the clustering of all the edges is provided. This scripts generates a file that can
be given to the visualization filter, which makes the exploration of the different edge clusters much faster.

The command to generate the cluster file is ``generate_brain_connectivity_cluster_file``. The command
requires:

* the mesh file,
* the edge file,
* the number of desired clusters,
* the output file

The output file can be given to the visualization filter in Paraview to avoid expensive computations
during the visualization.

.. warning::

  Requires ``nibabel`` (see above) and ``VTK`` (see installation instructions).

.. tip::

  ``scipy`` may be used to speed-up the computation of the k-means if installed.

References
----------

.. automodule:: src.utils.generate_processing
   :members:

.. automodule:: src.utils.generate_plugin_xml
   :members:
