
import numpy as np
from .plot import get_coords, get_center, center_matrix, origin_projection_matrix, get_cov
import naive_k_means
import hierarchical_k_means
import plot as pl
import os


def draw_edges(edges_files_path,
               poly_data_input,
               poly_data_output,
               means_path=None,
               edge_selection=None,
               edge_res=None,
               k_means=None,
               hierarchy_index=None,
               show_edges=True,
               mean_index=None):

    """
    Function for drawing edges in a Paraview plugin pipeline.

    :param edge_res: the number of segments each trajectory is divided into. The higher, the smoother the
      trajectory, the heavier the drawing.
    :param kmeans: number of clusters. If hierarchical, this is the number of clusters by level of hierarchy.
    """

    import vtk

    if means_path is None:
        means_path = ''

    if edge_selection is None:
        edge_selection = [120000, 120200],

    if edge_res is None:
        edge_res = 16

    if k_means is None:
        k_means = 7

    if mean_index is None:
        mean_index = -1

    if hierarchy_index is None:
        hierarchy_index = -1

    pts_offset = edge_res + 2

    # initialize input object of type vtkPolyData
    pts = get_coords(poly_data_input)
    pts_center = get_center(pts)
    pts = center_matrix(pts)

    print "edge_res", edge_res
    print pts

    # store covariance of coordinates
    cov = get_cov(pts)
    # get the backward projection matrix
    b_p = origin_projection_matrix(cov)

    ###################################
    # Loading/Generation of mean data #
    ###################################
    hierarchical = False

    # load prepared edges from file
    try:
        projected_edges = np.load(edges_files_path)
        edges = projected_edges[edge_selection[0]:edge_selection[1]]
    except IOError:
        print 'Please select a valid file path to load edge information from.'
        return

    # load kmeans data from file, alternatively calculate new kmeans
    means = None
    if means_path and os.path.exists(means_path):
        try:
            means = np.load(means_path)
            hierarchical = len(means[0].shape) == 1
            print '{}K-Means data loaded successfully.'.format(((hierarchical) * 'Hierarchical '))
        except IOError:
            print 'Mean path does not exist'

    if means is None:
        print 'Calculating kmeans...'
        means = naive_k_means.kmeans(k_means, edges)

    # choose means and edges to display
    labels = hierarchical_k_means.assemble_labels(means, hierarchy_index) if hierarchical else means[1]

    index_max = np.amax(labels) * 1.0
    mean_indices = np.arange(means[0][hierarchy_index].shape[0]) if hierarchical else np.arange(means[0].shape[0])

    if mean_index > -1:
        mean_indices = hierarchical_k_means.fetch_mean_labels(mean_index,
                                                              means,
                                                              hierarchy_index) if hierarchical else np.array([mean_index])

        mean_edges = np.array(means[0][hierarchy_index][
            mean_indices]) if hierarchical else np.array([means[0][mean_index]])

        assorted_labels = np.hstack(np.nonzero(labels == m)[
                                    0] for m in mean_indices) if hierarchical else [labels == mean_index]

        edges = edges[assorted_labels]
        labels = labels[assorted_labels]
    else:
        mean_edges = means[0][hierarchy_index] if hierarchical else means[0]

    #######################
    # Generation of Lines #
    #######################
    # initialize line_sets to add to output
    line_sets = np.array([[], [], []])
    mean_sets = np.array([[], [], []])
    edge_sets = np.array([[], [], []])

    radii = np.array([((edges[:, :3] - edges[:, 3:]) ** 2).sum(axis=1)[labels == _].mean() for _ in mean_indices])
    radii = [(r / radii.max()) * 60. for r in radii]

    for i, edge in enumerate(mean_edges):
        radius = radii[i]
        edge_pts = pl.create_poly_coords(
            edge[:3], edge[3:], r=radius, steps=edge_res, interpolate=True)
        mean_sets = np.append(mean_sets, edge_pts, axis=1)

    if show_edges:
        # reorder edges to match related mean edges - minimizes crossing lines and
        # visual loops when displaying edges
        edges = naive_k_means.reorder_edges(edges, naive_k_means.distances(edges, mean_edges)[1])
        for i in xrange(edges.shape[0]):
            line_pts = pl.create_poly_coords(
                edges[:, :3][i], edges[:, 3:][i],
                r=1, steps=edge_res)
            edge_sets = np.append(edge_sets, line_pts, axis=1)

        edge_sets = b_p.dot(edge_sets)
        edge_sets = np.array(np.split(edge_sets.T, edge_sets.shape[1] / pts_offset, axis=0))

    mean_sets = b_p.dot(mean_sets)
    line_sets = np.append(line_sets, mean_sets, axis=1)

    interpolation_array = pl.get_interpolation_array(pts_offset)

    if show_edges:
        for i, mean_id in enumerate(mean_indices):
            edge_ids = np.where(labels == mean_id)[0]
            pl.interpolate_arrays(
                edge_sets, mean_sets[:, i * pts_offset:(i + 1) * pts_offset].T, interpolation_array, edge_ids)

        for edge in edge_sets:
            line_sets = np.append(line_sets, edge.T, axis=1)

    # project back line coordinates
    line_sets += pts_center

    #
    # Drawing things for output in vtk
    #

    lines = vtk.vtkCellArray()

    # initialize output polydata object
    poly_data_output.Allocate(1, 1)
    poly_data_output.SetPoints(pl.create_points(pts))
    poly_data_output.SetLines(lines)

    # after initializing everything, we can append the lines to the output
    for line in np.split(line_sets, line_sets.shape[1] / pts_offset, axis=1):
        pl.draw_poly_line(poly_data_output, line)

    # adding colors
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(4)
    colors.SetName('Colors!')

    for i in xrange(mean_edges.shape[0]):
        index = i if mean_index == -1 else mean_indices[i]
        colors.InsertNextTuple4((1 - index / index_max) * 69., (index / index_max) * 255., 200., 200)

    for i in xrange(edges.shape[0]):
        index = labels[i]
        colors.InsertNextTuple4((1 - index / index_max) * 69., (index / index_max) * 255., 200., 200)

    poly_data_output.GetCellData().SetScalars(colors)
    del colors  # Raffi: needed ?
