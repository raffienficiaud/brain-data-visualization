Computation utilities
=====================

.. contents::
  :local:

Along the paraview plugin, several functions have been developed. Those utilities have different purposes:

* make off-line computations in order to speed-up the visualization
* support the writing of new visualization functions

Off-line computations
---------------------

kMeans and hierarchical kMeans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The kMeans algorithm currently being in use within the plugin is a naive implementation using a distance function
that is specific to the needs of the visualization. Accelerated algorithms exist such as the [Elkan]_ and that involves
a lower number of computations and distance estimations. The complexity of the kMeans is at best polynomial in the
number of points.

The hierarchical version of the kMeans runs the kMeans at each level of the hierarchy, and has also a complexity


Alignment
^^^^^^^^^
The data to visualize should be in the correct format in order to use the plugin.

Bibliographic references
------------------------

.. [Elkan] Elkan: Using the triangle inequality to accelerate k-means, ICML'03


References
----------

.. automodule:: src.naive_k_means
   :members:
