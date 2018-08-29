#!usr/bin/env python2
"""Connectivity visualisation filter, to be used as xml.
"""
Name = 'BrainConnectivityVis'
Label = 'FuNet'
Help = 'Help for the Test Filter'

NumberOfInputs = 1
InputDataType = 'vtkPolyData'
OutputDataType = 'vtkPolyData'
ExtraXml = ''

Properties = dict(
    bdv_path='/Users/Lennart/Desktop/brain-data-viz',
    edges_path='/Users/Lennart/Desktop/brain-data-viz/data/projected_edges.npy',
    means_path='',
    edge_selection=[120000, 120200],
    edge_res=16,
    k_means=7,
    show_edges=True,
    mean_index=-1,
    hierarchy_index=-1
)


def RequestData():
    import sys
    import numpy as np

    if not bdv_path in sys.path:
        sys.path.append(bdv_path)

    from brain_data_viz import plot as pl
    from brain_data_viz import naive_k_means as nm
    from brain_data_viz import hierarchical_k_means as hm

    reload(pl)
    reload(nm)
    reload(hm)

    pts_offset = edge_res + 2

    # initialize input object of type vtkPolyData
    poly_data_input = self.GetPolyDataInput()
    pts = pl.get_coords(poly_data_input)
    pts_center = pl.get_center(pts)
    pts = pl.center_matrix(pts)

    # store covariance of coordinates
    cov = pl.get_cov(pts)
    # get the backward projection matrix
    b_p = pl.origin_projection_matrix(cov)

    print '#####'

    ###################################
    # Loading/Generation of mean data #
    ###################################
    hierarchical = False

    # load prepared edges from file
    try:
        projected_edges = np.load(edges_path)
        edges = projected_edges[edge_selection[0]:edge_selection[1]]
    except IOError:
        return 'Please select a valid file path to load edge information from.'

    # load kmeans data from file, alternatively calculate new kmeans
    try:
        means = np.load(means_path)
        hierarchical = len(means[0].shape) == 1
        print '{}K-Means data loaded succesfully.'.format(((hierarchical) * 'Hierarchical '))
    except IOError:
        means = nm.kmeans(k_means, edges)
        print 'Calculating kmeans...'

    # choose means and edges to display
    labels = hm.assemble_labels(
            means, hierarchy_index) if hierarchical else means[1]

    index_max = np.amax(labels) * 1.0
    mean_indices = np.arange(means[0][hierarchy_index].shape[0]) if hierarchical else np.arange(means[0].shape[0])

    if mean_index > -1:
        mean_indices = hm.fetch_mean_labels(
                    mean_index, means, hierarchy_index) if hierarchical else np.array([mean_index])

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

    radii = np.array([((edges[:, :3] - edges[:, 3:]) ** 2).sum(axis=1)
                        [labels == id].mean() for id in mean_indices])
    radii = [(r / radii.max()) * 60. for r in radii]

    for i, edge in enumerate(mean_edges):
        radius = radii[i]
        edge_pts = pl.create_poly_coords(
            edge[:3], edge[3:], r=radius, steps=edge_res, interpolate=True)
        mean_sets = np.append(mean_sets, edge_pts, axis=1)

    if show_edges:
        # reorder edges to match related mean edges - minimizes crossing lines and
        # visual loops when displaying edges
        edges = nm.reorder_edges(edges, nm.distances(edges, mean_edges)[1])
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

    #########################################
    lines = vtk.vtkCellArray()

    # initialize output polydata object
    output = self.GetPolyDataOutput()
    output.Allocate(1, 1)
    output.SetPoints(pl.create_points(pts))
    output.SetLines(lines)

    # after initializing everything, we can append the lines to the output
    for line in np.split(line_sets, line_sets.shape[1] / pts_offset, axis=1):
        pl.draw_poly_line(output, line)

    # adding colors
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(4)
    colors.SetName('Colors!')

    for i in xrange(mean_edges.shape[0]):
        index = i if mean_index == -1 else mean_indices[i]
        #colors.InsertNextTuple4(255,69,69,255)
        colors.InsertNextTuple4((1 - index / index_max)
                                * 69., (index / index_max) * 255., 200., 200)

    for i in xrange(edges.shape[0]):
        index = labels[i]
        #colors.InsertNextTuple4(255,255,255,255)
        colors.InsertNextTuple4((1 - index / index_max)
                                * 69., (index / index_max) * 255., 200., 200)

    output.GetCellData().SetScalars(colors)
    del colors
