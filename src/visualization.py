
from __future__ import print_function

import numpy as np
from .plot import get_coords, get_center, TransformSpherical
import naive_k_means
import hierarchical_k_means
import plot as pl
import os

_verbose = True


def draw_edges(edges_files_path,
               poly_data_input,
               poly_data_output,
               centroids_file_path=None,
               edge_selection=None,
               resolution=None,
               nb_clusters=None,
               hierarchy_index=None,
               show_edges=True,
               cluster_index=None,
               center_edges=False):

    """
    Function for drawing connectivity edges in a Paraview filter pipeline.

    :param poly_data_input: Paraview upstream data (coming from the previous step in the pipeline)
    :param poly_data_output: Paraview result structure
    :param edges_files_path: numpy compatible file containing the set of all connectivity edges. The format
      is a 6-dimensional array indicating the source and destination of each edge as a unique entry in the
      array. The orientation of the edge is not taken into account.
    :param resolution: the number of segments each trajectory is divided into. The higher this number, the
      smoother the trajectory, but also the heavier the drawing. Defaults to `16`.
    :param centroids_file_path: if set to an existing file name, loads the computation of the k-means (
      including hierarchical k-means) from the specified file.
    :param nb_clusters: number of clusters. If hierarchical, this is the number of clusters by level of hierarchy.
    :param cluster_index: if not ``None``, indicates the index of a specific cluster to display (discards the visualization
      of all other clusters)
    :param center_edges: if ``True``, centers the input mesh
    """

    import vtk

    if centroids_file_path is None:
        centroids_file_path = ''

    if edge_selection is None:
        edge_selection = [120000, 120010]

    if resolution is None:
        resolution = 16

    if nb_clusters is None:
        nb_clusters = 7

    if cluster_index is None:
        cluster_index = -1

    if hierarchy_index is None:
        hierarchy_index = -1

    if not edges_files_path or not os.path.exists(edges_files_path):
        print("Indicated edge file is incorrect or does not exist")
        return

    # initialize input object of type vtkPolyData
    pts = get_coords(poly_data_input)

    # geometric transformation object
    spherical_transformation = TransformSpherical(coordinates=pts)

    ###################################
    # Loading/Generation of mean data #
    ###################################

    # load prepared edges from file
    try:
        projected_edges = np.load(edges_files_path)
    except IOError:
        print('Please select a valid file path to load edge information from.')
        return

    # load kmeans data from file, if exist and given
    hierarchical = False
    centroids = None
    labels = None
    if centroids_file_path and os.path.exists(centroids_file_path):
        try:
            centroids, labels = np.load(centroids_file_path)
            hierarchical = len(centroids.shape) == 1
            if _verbose:
                print('{}K-Means data loaded successfully.'.format(((hierarchical) * 'Hierarchical ')))
        except IOError as e:
            print('An error occurred while loading the specified centroids file:', e)
    elif centroids_file_path:
        print('Cluster file does not exist')

    # selecting the edges, only if the centroids file does not exist
    # otherwise we will have a mismatch between the user input and the file content
    if _verbose:
        print("Edges mean is before projection {}".format(get_center(np.vstack((projected_edges[:, 0:3], projected_edges[:, 3:6])).T)))
    if centroids is None:
        projected_edges = projected_edges[edge_selection[0]:edge_selection[1], :]

    if _verbose:
        print("Edges mean is before projection {}".format(get_center(np.vstack((projected_edges[:, 0:3], projected_edges[:, 3:6])).T)))

    # centering the edges
    projected_edges -= np.vstack((spherical_transformation.mean, spherical_transformation.mean)).T

    # projecting the edges to spherical representation
    projected_edges[:, 0:3] = spherical_transformation.transform_to_spherical(projected_edges[:, 0:3].T).T
    projected_edges[:, 3:6] = spherical_transformation.transform_to_spherical(projected_edges[:, 3:6].T).T

    # compute the kmeans/clustering if no data from the clustering loaded
    if centroids is None or labels is None:
        if _verbose:
            print('Calculating the kmeans on {} edges... this might take some time'.format(projected_edges.shape[0]))
        centroids, labels = naive_k_means.kmeans(nb_clusters, projected_edges)
    else:
        # applying the same transformation as for the edges
        centroids -= np.vstack((spherical_transformation.mean, spherical_transformation.mean)).T

        # projecting the edges to spherical representation
        centroids[:, 0:3] = spherical_transformation.transform_to_spherical(centroids[:, 0:3].T).T
        centroids[:, 3:6] = spherical_transformation.transform_to_spherical(centroids[:, 3:6].T).T

    # choose centroids and edges to display
    labels = hierarchical_k_means.assemble_labels((centroids, labels), hierarchy_index) if hierarchical else labels

    # we do not want the colors of the clusters to change if we select another cluster
    # so we take the max of the labels
    index_max = np.amax(labels) * 1.0 + 1

    if cluster_index > -1:
        centroid_indices = hierarchical_k_means.fetch_mean_labels(cluster_index,
                                                                  (centroids, labels),
                                                                  hierarchy_index) if hierarchical else np.array([cluster_index])

        mean_edges = np.array(centroids[hierarchy_index][centroid_indices]) if hierarchical else np.array([centroids[cluster_index]])

        assorted_labels = np.hstack(np.nonzero(labels == m)[0] for m in centroid_indices) if hierarchical else (labels == cluster_index)

        projected_edges = projected_edges[assorted_labels]
        labels = labels[assorted_labels]
    else:
        mean_edges = centroids[hierarchy_index] if hierarchical else centroids
        centroid_indices = np.arange(centroids[hierarchy_index].shape[0]) if hierarchical else np.arange(centroids.shape[0])
        # to replace with np.unique, in case a label is not used

    #######################
    # Generation of Lines #
    #######################

    # sorting by mean edge length.
    # edge_lengths = np.sqrt(((projected_edges[:, :3] - projected_edges[:, 3:]) ** 2).sum(axis=1))
    edge_lengths = ((projected_edges[:, :3] - projected_edges[:, 3:]) ** 2).sum(axis=1)
    radii = np.array([edge_lengths[labels == _].mean() for _ in centroid_indices])
    radii *= 120 / radii.max()  # [(r / radii.max()) * 60. for r in radii]

    # calculate the paths for the means
    mean_sets = pl.create_poly_coords(mean_edges[:, :3].T,
                                      mean_edges[:, 3:].T,
                                      r=radii,
                                      steps=resolution)

    # back to original space
    for _ in range(mean_sets.shape[2]):
        mean_sets[:, :, _] = spherical_transformation.transform_to_origin(mean_sets[:, :, _])
        mean_sets[:, :, _] += spherical_transformation.mean

    if show_edges:
        # reorder edges to match related mean edges - minimizes crossing lines and
        # visual loops when displaying edges
        projected_edges = naive_k_means.reorder_edges(projected_edges,
                                                      naive_k_means.distances(projected_edges, mean_edges)[1])

        # getting back the edge associated radiuses
        edge_radii = np.zeros(projected_edges.shape[0])
        for current_label, current_radius in zip(centroid_indices, radii):
            edge_radii[labels == current_label] = current_radius

        edge_sets = pl.create_poly_coords(projected_edges[:, :3].T,
                                          projected_edges[:, 3:].T,
                                          r=edge_radii,
                                          steps=resolution)

        # transform back to original space
        for _ in range(edge_sets.shape[2]):
            edge_sets[:, :, _] = spherical_transformation.transform_to_origin(edge_sets[:, :, _])
            edge_sets[:, :, _] += spherical_transformation.mean

        # interpolates with the centroid
        interpolation_array = pl.get_interpolation_array(edge_sets.shape[2])
        assert(mean_sets.shape[1] == centroid_indices.shape[0])
        for current_mean_index, current_label in zip(range(mean_sets.shape[1]), centroid_indices):
            current_mean = mean_sets[:, current_mean_index, :].reshape((3, 1, edge_sets.shape[2]))
            edge_sets[:, labels == current_label, :] = (1 - interpolation_array) * edge_sets[:, labels == current_label, :] + interpolation_array * current_mean

    #
    # Drawing things for output in vtk
    #

    # initialize output polydata object
    poly_data_output.Allocate(1, 1)

    # adding colors
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(4)
    colors.SetName('Colored-edges')

    assert(mean_sets.shape[1] == centroid_indices.shape[0])

    for current_mean_index, current_centroid_label in zip(range(mean_sets.shape[1]), centroid_indices):

        current_centroid_segment = mean_sets[:, current_mean_index, :]
        pl.draw_poly_line(poly_data_output, current_centroid_segment)

        color_mean = [(1 - current_centroid_label / index_max) * 69.,
                      (current_centroid_label / index_max) * 255.,
                      200.,
                      200]

        colors.InsertNextTuple4(*color_mean)
        corresponding_edges = edge_sets[:, labels == current_centroid_label, :]

        color_edges = [_ for _ in color_mean]
        color_edges[-1] = 100

        for index_current_edge in range(corresponding_edges.shape[1]):
            current_edge_segment = corresponding_edges[:, index_current_edge, :]
            pl.draw_poly_line(poly_data_output, current_edge_segment)

            colors.InsertNextTuple4(*color_edges)

    poly_data_output.GetCellData().SetScalars(colors)

    return
