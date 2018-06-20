import sys
from mpi_is_sw.brain_connectivity.utils.generate_plugin_xml import generate_plugin_after_install

# getting the folder of the main entry point
import mpi_is_sw.brain_connectivity.plugin as plugin_main

import inspect

file_to_process = inspect.getsourcefile(plugin_main)

print
print '#' * 20
print "Generating the Brain fMRI plugin/Paraview filter for installation to file", sys.argv[1]
print "Once generated, add this XML file in the plugin manager. Remove any prior existing plugin"
print '#' * 20
print
print file_to_process
print
print '#' * 20
print

#generatePythonFilterFromFiles(file_to_process, sys.argv[1])
generate_plugin_after_install(sys.argv[1])
