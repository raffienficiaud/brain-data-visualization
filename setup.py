try:
    from setuptools import setup, convert_path
    has_setup_tools = True

except ImportError:
    # Installing from Paraview directly does not include setuptools
    from distutils.core import setup
    from distutils.util import convert_path
    has_setup_tools = False


def _get_version():
    """ Convenient function to get the version of this package."""

    import os
    ns = {}
    version_path = convert_path('src/version.py')
    if not os.path.exists(version_path):
        return None
    with open(version_path) as version_file:
        exec(version_file.read(), ns)

    return ns['__version__']


# this package will go to the following namespace
namespace_package = 'mpi_is_sw'

# not installing the test files
packages = [namespace_package,
            '%s.brain_connectivity' % namespace_package,
            '%s.brain_connectivity.utils' % namespace_package,
            ]  # actual subpackage described here

package_dir = {namespace_package: '%s-namespace' % namespace_package,
               '%s.brain_connectivity' % namespace_package: 'src',
               '%s.brain_connectivity.utils' % namespace_package: 'src/utils',
               }

additional_setup_args = {}

if has_setup_tools:
    additional_setup_args["entry_points"] = {
        'console_scripts': ['generate_brain_connectivity_plugin=%s.brain_connectivity.utils.generate_plugin_xml:main_virtualenv' % namespace_package,
                            'generate_brain_connectivity_edge_file=%s.brain_connectivity.utils.generate_processing:main_generate_edge_file' % namespace_package,
                            'generate_brain_connectivity_edge_volume_file=%s.brain_connectivity.utils.generate_processing:main_generate_edge_volume_file' % namespace_package,
                            'generate_brain_connectivity_volume_file=%s.brain_connectivity.utils.generate_processing:main_generate_volume_file' % namespace_package,
                            'generate_brain_connectivity_cluster_file=%s.brain_connectivity.utils.generate_processing:main_generate_cluster_file' % namespace_package,
                            ],

    }
    additional_setup_args["install_requires"] = ['numpy']
    additional_setup_args["zip_safe"] = False

else:
    # we create two helper scripts for generating the plugin XML file
    # if we are here, it means we are using distutils and setuptools is not available, and we cannot be in
    # a virtual environment.
    additional_setup_args["scripts"] = ['scripts/generate_brain_connectivity_plugin',
                                        'scripts/brain_connectivity_plugin_generator_wrapper.py']

setup(name="brain-connectivity-visualization",
      version=_get_version(),
      packages=packages,
      package_dir=package_dir,
      author='Lennart Bramlage, Raffi Enficiaud',
      author_email='raffi.enficiaud@tuebingen.mpg.de',
      maintainer='Raffi Enficiaud',
      maintainer_email='raffi.enficiaud@tuebingen.mpg.de',
      url='https://is.tuebingen.mpg.de/software-workshop',
      description='Brain fMRI connectivity visualization plugin for Paraview',
      long_description=open(convert_path('README.md')).read(),
      license='MIT',
      **additional_setup_args
      )
