Paraview plugin
===============

.. contents::
  :local:


We describe in this page how to use the plugin inside Paraview:

# the plugin acts on a mesh of point representing the surface of the brain
# several visualization algorithm and parameters are shown in the user interface.

Input data
----------
In order to function, the visualization needs:

* the mesh on which computation will be performed. This is usually a mesh file that Paraview
  can understand,
* the connectivity file, that contains the topology of connection network in the brain. This
  connectivity file is a mapping between pair of mesh locations and a weight.

The way the connectivity file is created is detailed in another section.

Parameters
----------

The parameters exposed to the user by the plugin are shown in the picture below:

.. image:: /_static/plugin_panel.png
    :height: 300

The meaning of the parameters are:

* ``Edge res``: indicates the edges resolution, ie. the number of intermediate steps a trajectory
  connecting two sites is divided into. The higher, the smoother the trajectory but also the
  heavier the on-screen drawing,
* ``Edge selection``: a range of edges on which the processing will be performed. Any edge falling
  outside of this range is ignored in the visualization and the computation,
* ``Edge path``: the **absolute** path of the file containing the definition of the topology,
* ``show edges``: if checked,

