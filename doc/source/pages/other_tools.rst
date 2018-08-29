Computation utilities
=====================

.. contents::
  :local:

Along the Paraview plugin, several functions have been developed. Those utilities have different purposes:

* make off-line computations in order to speed-up the visualization
* support the writing of new visualization functions

Visualization preparation
-------------------------
The visualization requires a little preparation step that aims at creating necessary or optional files.
Those files are either required for the visualization, or assist the visualization with
additional elements.

The only necessary file is the **edge file** that indicates the plugin the end-points of the different
connectivity edges in the 3d volume.

Additionally, it is possible to create:

* a volume file from a nifti volumetric data. An example of use of such a file in Paraview
  is given in the image below

  .. image:: /_static/volume_view.png
      :width: 300

* a grid file of all edges end-points location.

  .. image:: /_static/end_points_location_view.png
      :width: 300

Off-line computations
---------------------

Clustering with kMeans
^^^^^^^^^^^^^^^^^^^^^^
The clustering of edges is performed with the kmeans algorithm.

The version of this algorithm currently being in use within the plugin is implemented in a naive way in Python, using a
distance function that is specific to the needs of the visualization. Accelerated algorithms exist such as the
[Elkan]_ and that involves a lower number of computations and distance evaluations.

The hierarchical version of the kMeans runs the kMeans at each level of the hierarchy, and has also a complexity
similar to the kMeans multiplied by the number of hierarchy levels.

Since the clustering may take some time to converge, a script accessible after the installation of the
package in a virtual environment is provided.


Bibliographic references
------------------------

.. [Elkan] Elkan: Using the triangle inequality to accelerate k-means, ICML'03


References
----------

.. automodule:: src.naive_k_means
   :members:
