Installation
============

The repository contains two parts:

* a python plugin that runs inside Paraview. For running this plugin, you need to
  install, in a first step, the plugin dependencies, and then to install the plugin
  into Paraview
* a python package that contains several utilities for aligning data and computing
  clusters of connectivity.

Package Installation
--------------------

Some background
^^^^^^^^^^^^^^^
There is no streamlined way to install a python plugin into Paraview, and/or to use
an external python library. Paraview has different installation flavour that differ
according to the platform or the way you installed the tool. We tried to make the
intallation procedure as easy as possible.

OSX
^^^
Paraview on OSX relies on the system Python (2.7.X at the current day). This means that
it is possible to install the package and the plugins from a unique location using a
python virtual environment, and this will be the preferred method on this platform.

The installation of Paraview is often located in `/Application` folder, and it is not
possible to write into the Paraview application folder. Therefor the installation
on OSX requires a virtual environment or a global installation. We focus here on the
virtual environment installation.

First create a virtual environment:
```
mkdir ~/.venvs
virtualenv --system-site-packages ~/.venvs/venv_paraview_brain
. ~/.venvs/venv_paraview_brain/bin/activate
```




Linux
^^^^^


Plugin Installation
-------------------
Once the package has been installed, the library uses a tool for converting
the python files into a Paraview installable plugin.

