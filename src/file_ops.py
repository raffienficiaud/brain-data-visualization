#-*- coding: utf-8 -*-

"""These methods will help to set up edge information for the visualization.
"""

import csv
import numpy as np


def assemble_edges(nifti_path, csv_path):
    """Returns a number of edges based on a set of points loaded from
    ``nifti_path`` and a set of index pairs (and weights) loaded through
    ``csv_path``.
    """
    nifti = read_nifti(nifti_path)
    indices = read_csv(csv_path)
    edges = np.array([nifti[:, [int(index[0]), int(
        index[1])]].T.flatten() for index in indices])
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


def read_nifti(file_path):
    """Reads a nifti format file and returns a set of indices corresponding to
    voxel coordinates of a brain volume, where the voxel index is not zero.

    :returns: a matrix of floats of shape (3, n), where N is the number of
        voxels present in the volume

    .. note:: requires a successful installation of nibabel
    """
    import nibabel as nib

    nii = nib.load(file_path)
    data_3d = nii.get_data()
    data_1d = np.array(np.nonzero(data_3d), dtype=np.float32)
    return data_1d


def read_csv(file_path):
    """Reads a csv file and returns its contents as a numpy array.

    :returns: a matrix of floats of shape (n, 3)
    """
    with open(file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = np.array([np.array(row, dtype=np.float32) for row in reader])
    return data
