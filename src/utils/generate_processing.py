"""
utils.generate_from_nifty
-------------------------

Provides tools for constructing intermediate files needed for the visualization.

.. note::

    This file is meant to be installed during the installation of the package as a
    callable script with several commands.

The functions provided here are:

* the transformation of a Nifti indicator file to a set of edge coordinates. Those
  edges will be used by the visualization algorithm
* the transformation of the Nifti volumetric data to a VTK image that can be rendered
  inside Paraview.

The edge file requires:

* a Nifti volume that indicates the end points on a grid, and
* a CSV file that contains the topology that connects the previous end points. This CSV file
  should have two columns and a number of rows identical to the number of edges. Each column
  entry is an index to a coordinate that is extracted from the previous Nifti file. The other
  columns are ignored.

"""

from __future__ import print_function

import sys
import numpy as np


def generate_edge_file(csv_file_path,
                       nifti_edge_file_path,
                       output_file):
    """Convenience script for creating an edge file

    :param nifti_edge_file_path: the path containing the nifti volume indicating
      the locations of the edge end points
    :param csv_file_path: the file containing the topology
    :param output_file: the output file in a numpy array format

    .. seealso:: :py:func:`..file_ops.assemble_edges`
    """

    from mpi_is_sw.brain_connectivity.file_ops import assemble_edges

    edges = assemble_edges(nifti_path=nifti_edge_file_path,
                           csv_path=csv_file_path)

    np.save(output_file, edges)
    print("Successfully created {} edges from file {} to file {}".format(edges.shape[0],
                                                                         csv_file_path,
                                                                         output_file))


def generate_edge_volume_file(nifti_edge_file_path,
                              output_file):
    """Convenience script for creating a volume locating all edges end-points

    :param nifti_edge_file_path: the path containing the nifti volume indicating
      the locations of the edge end points
    :param csv_file_path: the file containing the topology
    :param output_file: the output file in a VTK file format

    .. seealso:: :py:func:`..file_ops.assemble_edges`
    .. seealso:: :py:func:`..file_ops.save_points_to_vtk_array`
    """

    from mpi_is_sw.brain_connectivity.file_ops import (
        pointset_from_indicator_nifti_file,
        save_points_to_vtk_array)

    nifti = pointset_from_indicator_nifti_file(nifti_edge_file_path)

    save_points_to_vtk_array(numpy_points=nifti, filename=output_file)
    print("Successfully saved the edges to the VTK file {}".format(output_file))


def generate_volume_file(nifti_volumetric_file_path,
                         output_file):
    """Convenience script for creating a VTK compatible volume image file.
    """

    from mpi_is_sw.brain_connectivity.file_ops import nifti_volumetric_data_to_vtkimage, save_vtkimage

    vtkimage = nifti_volumetric_data_to_vtkimage(nifti_volumetric_path=nifti_volumetric_file_path)

    save_vtkimage(vtkimage, output_file)


def generate_cluster_file(mesh_file,
                          edge_file,
                          number_of_clusters,
                          output_file,
                          max_iterations=100):
    """Convenience function for running the clustering on a set of edges

    :param mesh_file: the mesh that will be used for centering the data and making it isotropic
    :param edge_file: the file containing the edges as computed from :py:func:`.generate_edge_file`
    :param number_of_cluster: the number of desired clusters
    :param output_file: the file in which the results of the clutering will be done
    :param max_steps: maximum number of iterations for the kmeans

    .. note::

        The plugin will not know if the files and parameters are consistent. If you provide a file
        name that contains the computation of the centroids for a specific edge file, and the edge
        file is incorrectly given to the plugin, the plugin will be unable to check.
    """

    from mpi_is_sw.brain_connectivity.naive_k_means import kmeans
    from mpi_is_sw.brain_connectivity.plot import TransformSpherical, get_coords
    import vtk

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(mesh_file)
    reader.Update()

    poly_data_input = reader.GetOutput()
    mesh_points = get_coords(poly_data_input)
    spherical_transformation = TransformSpherical(coordinates=mesh_points)

    edges = np.load(edge_file)

    # same operations as for the plugin/filter
    edges -= np.vstack((spherical_transformation.mean, spherical_transformation.mean)).T
    edges[:, 0:3] = spherical_transformation.transform_to_spherical(edges[:, 0:3].T).T
    edges[:, 3:6] = spherical_transformation.transform_to_spherical(edges[:, 3:6].T).T

    print("Generating %d clusters out of %d edges. Max iteration is set to %d" % (number_of_clusters, edges.shape[0], max_iterations))

    centroids, labels = kmeans(k=number_of_clusters,
                               edges=edges,
                               max_iterations=max_iterations,
                               verbose=True)

    # applying the reverse transformation
    # this is to stay consistent in case we perform something different with the data already on disk
    # this also removes the "mesh" variable from the saved data (for instance: the mesh center is
    # not relevant any more).
    centroids[:, 0:3] = spherical_transformation.transform_to_origin(centroids[:, 0:3].T).T
    centroids[:, 3:6] = spherical_transformation.transform_to_origin(centroids[:, 3:6].T).T
    centroids += np.vstack((spherical_transformation.mean, spherical_transformation.mean)).T

    # saving the file back to disk
    np.save(output_file, (centroids, labels))

    print("Successfully saved the clusters to the numpy file {}".format(output_file))


def main_generate_edge_file():
    if len(sys.argv) != 4:
        print("""Generates the edge file needed for the brain data rendering.""")
        print('Usage: %s <csv_edge_file> <nifti_file> <output_file>' % sys.argv[0])
        sys.exit(1)

    generate_edge_file(
        csv_file_path=sys.argv[1],
        nifti_edge_file_path=sys.argv[2],
        output_file=sys.argv[3])


def main_generate_edge_volume_file():
    if len(sys.argv) != 4:
        print("""Generates the edge file as a VTK file acting as an indicator function.

        This is mainly for debugging the content of the nifti file.""")

        print('Usage: %s <nifti_file> <output_file>' % sys.argv[0])
        sys.exit(1)

    generate_edge_volume_file(
        nifti_edge_file_path=sys.argv[1],
        output_file=sys.argv[2])


def main_generate_volume_file():
    if len(sys.argv) != 3:
        print('Usage: %s <nifti_volume_file> <output_file>' % sys.argv[0])
        sys.exit(1)

    generate_volume_file(
        nifti_volumetric_file_path=sys.argv[1],
        output_file=sys.argv[2])


def main_generate_cluster_file():
    if len(sys.argv) != 5:
        print('Usage: %s <mesh file> <edge_file> <numbre_of_clusters> <output_file>' % sys.argv[0])
        sys.exit(1)

    generate_cluster_file(
        mesh_file=sys.argv[1],
        edge_file=sys.argv[2],
        number_of_clusters=int(sys.argv[3]),
        output_file=sys.argv[4])
