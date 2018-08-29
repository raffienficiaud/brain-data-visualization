# -*- coding: utf-8 -*-

"""Edge and image volume transformation functions.
"""

import csv
import numpy as np


def assemble_edges(nifti_path, csv_path):
    """Returns edges as coordinates based on a set of points loaded from
    ``nifti_path`` and a set of index pairs (and weights) loaded through
    ``csv_path``.

    The returned matrix (numpy array) has a number of rows corresponding to the number of
    edges, and 6 columns. The first 3 columns are the coordinates of the edge
    sources, and the last 3 the location of the edge destinations.

    :param nifti_path: path of the nifti file containing the location of the edges
    :param csv_path: path of the CSV file containing the edges end-point indices
      from the nifti file
    :returns: a numpy array of shape ``N x 6``, where ``N`` is the number of edges.

    .. note::

        The returned points coordinates is scaled according to the nifti
        voxel size and orientation wrt. VTK.

    """

    # the returned set of points is already scaled
    nifti = pointset_from_indicator_nifti_file(nifti_path)
    indices = read_csv(csv_path)[:, 0:2].astype(np.int32)

    edges = np.vstack((np.take(nifti, indices[:, 0], axis=1),
                       np.take(nifti, indices[:, 1], axis=1))).T
    return edges


def subsample_edges(n, origin, edges, r=5.):
    """Sample ``n`` edges around a point ``origin``.
    """
    closest_edges = np.empty([1, 6])
    for i in xrange(edges.shape[0]):
        if len(closest_edges) == n:
            closest_edges = np.delete(closest_edges, 0, axis=0)
            return closest_edges
        if np.sqrt(((origin - edges[:, 3:][i]) ** 2).sum(axis=0)) < r:
            closest_edges = np.append(closest_edges, [edges[i]], axis=0)

    closest_edges = np.delete(closest_edges, 0, axis=0)
    return closest_edges


def adjust_array_to_vtk(np_array):
    """Flips the dimensions of an array to be VTK conformant

    All dimensions except for the X axis are flipped, to account for the
    way images and coordinate system is organized in VTK.

    .. note:

        Requires numpy >= 1.14
    """

    if hasattr(np, 'flip'):
        np_array = np.flip(np_array, axis=2)
        np_array = np.flip(np_array, axis=1)
    else:
        # old way, needed for some version of numpy embedded in Paraview
        np_array = np_array[:, :, ::-1][:, ::-1, :]
    return np_array


def apply_nii_scaling(nii_data, np_array):
    """Applies a scaling to a set of coordinates from the nifti data

    The nifti files contain the voxel size/resolution. This voxel size is used here
    in order to scale a set of coordinates.

    :param niidata: the nifti object returned by the nifti reader
    :param np_array: the set of coordinates organized in a matrix. Each
      column is a coordinate, each row is a dimension of the coordinates.
    :return: a numpy array of the same dimension as the ``np_array``.
    """
    scaling = nii_data.header['pixdim'][1:4]

    return np.diag(scaling).dot(np_array)


def pointset_from_indicator_nifti_file(file_path):
    """Returns a set of points with non-zero values from a nifti file

    Given a nifti file that contains values on a regular grid, this function
    retrieves the points where there is actually a value (non-zero) and return
    the position of those points in the grid.

    This function is used for instance to get the list of points from
    an indicator grid, for instance for assembling the edges end-points in the
    context of connexel visualisation.

    :param file_path: full path to the nifti file
    :returns: a matrix of floats of shape (3, n), where N is the number of
        voxels present in the volume

    .. note:: requires the ``nibabel`` python package
    """
    import nibabel as nib

    nii = nib.load(file_path)
    data_3d = adjust_array_to_vtk(nii.get_data())

    data_1d = np.array(np.nonzero(data_3d), dtype=np.float32)

    return apply_nii_scaling(nii, data_1d)


def nifti_volumetric_data_to_vtkimage(nifti_volumetric_path):
    """Transforms a grid/array to a VTK image

    A scaling is performed as a VTK transformation, to account for
    the voxel size.
    """

    import vtk
    from vtk.util import numpy_support
    import nibabel as nib

    nii_vol = nib.load(nifti_volumetric_path)

    data_vol_3d = adjust_array_to_vtk(nii_vol.get_data())
    pixdims = nii_vol.header['pixdim'][1:4]

    # order F seems to give better results

    # we make a deep copy such that there is no reference to the numpy array here
    # order F is important
    VTK_data = numpy_support.numpy_to_vtk(num_array=data_vol_3d.ravel(order='F'),
                                          deep=True,
                                          array_type=vtk.VTK_SHORT)  # to FIX: depends on the numpy array

    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(data_vol_3d.shape)
    vtk_image.SetSpacing(pixdims)  # this accounts for the voxel sizes
    vtk_image.SetOrigin(np.zeros(len(data_vol_3d.shape)))
    vtk_image.GetPointData().SetScalars(VTK_data)

    # ... the example below works except for the voxel size transformation
    # ... but is more complicated than the current implementation

    # dataImporter = vtk.vtkImageImport()
    # data_string = data_vol_3d.flatten(order='F').tostring()
    # dataImporter.CopyImportVoidPointer(data_string, len(data_string))

    # dataImporter.SetDataScalarTypeToShort()
    # dataImporter.SetNumberOfScalarComponents(1)

    # s = data_vol_3d.shape
    # dataImporter.SetDataExtent(0, s[0] - 1, 0, s[1] - 1, 0, s[2] - 1)
    # dataImporter.SetWholeExtent(0, s[0] - 1, 0, s[1] - 1, 0, s[2] - 1)

    # ... apply the transformation for the scaling
    # ... not working

    # trans = vtk.vtkTransform()
    # trans.Scale(pixdims[0], pixdims[1], pixdims[2])

    # dataImporter.SetTransform(trans) # this is the line not working

    # ... we get the image from the output of the importer `dataImporter.GetOutput()`

    return vtk_image


def get_vtkcellarray_from_vtkpoints(vtkpoints):
    """Creates the cells associated to each point taken individually

    :param vtkpoints: an object ``vtkPoints`` holding the set of points.
    :returns: the associated ``vtkCellArray``

    The cells do not have any connectivity and are associated to unique points
    """
    import vtk
    from vtk.util import numpy_support

    verts = vtk.vtkCellArray()

    if 0:
        # safe but slow way
        for index in range(vtkpoints.GetNumberOfPoints()):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(index)

    else:
        # much faster
        # we have to create a list (1, id0, 1, id1, ....) where 1 indicates the number
        # of points in the cell, and idX is the id of the point
        array_cells = np.ones(vtkpoints.GetNumberOfPoints())
        array_cells = np.vstack((array_cells, np.arange(vtkpoints.GetNumberOfPoints())))
        array_cells = np.ascontiguousarray(array_cells.ravel(order='F'),
                                           dtype=np.int64)

        verts.SetCells(vtkpoints.GetNumberOfPoints(),
                       numpy_support.numpy_to_vtkIdTypeArray(array_cells, deep=True))

    return verts


def get_vtkcellarray_from_numpy(numpy_points):
    """Creates the cells associated to each point taken individually

    :param numpy_point: numpy array of points. The format follows the same
      as the input of :py:func:`.create_points`
    :returns: a tuple containing the ``vtkPoints`` and the associated ``vtkCellArray``

    The cells do not have any connectivity and are associated to unique points

    """

    from .plot import create_points
    vtkpoints = create_points(numpy_points)
    return vtkpoints, get_vtkcellarray_from_vtkpoints(vtkpoints)


def save_points_to_vtk_array(numpy_points, filename):
    """Utility function that writes the numpy points to a VTK file

    This is a convenience function for displaying the set of points, mostly
    for checking the proper alignment of the different sources of information
    (nifti, mesh, etc).

    """
    import vtk

    vtkpoints, verts = get_vtkcellarray_from_numpy(numpy_points)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtkpoints)
    polydata.SetVerts(verts)
    writer.SetInputData(polydata)
    writer.Update()
    writer.Write()


def save_vtkimage(vtk_image, filename):
    """Utility that saves a VTK image to disk

    The vtk image may have been constructed with :py:func:`nifti_volumetric_data_to_vtkimage`.

    :param vtk_image: the VTK image to save
    :param filename: the output file

    .. note::

        it is currently not possible to select the format
    """

    import vtk
    image_saver = vtk.vtkMetaImageWriter()
    image_saver.SetInputData(vtk_image)
    image_saver.SetFileName(filename)
    image_saver.Write()


def read_csv(csv_file_path):
    """Reads a csv file and returns its contents as a numpy array.

    :param csv_file_path: the CVS file containing the connections. Only the columns
      containing the edge indices are relevant (first 2)
    :returns: a matrix of floats of shape (n, 3)
    """
    with open(csv_file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # data = np.vstack((np.array(row, dtype=np.float32) for row in reader))
        data = np.array([np.array(row, dtype=np.float32) for row in reader])
    return data
