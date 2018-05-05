#!usr/bin/env python2
# -*- coding: utf-8 -*-
"""The all enclosing brain data viz module.
"""
import vtk
import numpy as np

# matrix operations


def get_coords(vtk_object, step=1):
    """Extracts the coordinates of a ``vtkPoints`` or ``vtkPolyData`` object.

    :param vtk_object: the only requirement for the ``vtk_object`` parameter is, that it is
    an instance of a vtk class, that implements vtkPoints
    :type vtk_object: the ``vtk_object`` parameter is of type vtkPolydata or vtkPoints
    :type step: the parameter ``step`` is of type integer
    :returns: this returns a coordinate matrix where each column is a single observation
    :rtype: numpy array
    """
    len_pts = vtk_object.GetNumberOfPoints()
    coords = np.array([vtk_object.GetPoint(i)
                       for i in range(0, len_pts, step)])
    return coords.T


def get_center(data_matrix):
    """Returns the centroid of a matrix. If data_matrix is normalized this equals [0, 0, 0].

    :param data_matrix: a coordinate matrix where each column is a single observation
    :type data_matrix: numpy array
    :returns: this returns a vector, representing the center position of the input matrix
    :rtype: numpy array
    """
    return np.mean(data_matrix, axis=1, keepdims=1)


def center_matrix(data_matrix):
    """Centers a matrix, so that its centroid equals the zero vector.

    :param data_matrix: a coordinate matrix where each column is a single observation
    :type data_matrix: numpy array
    :returns: this returns a version of the input matrix where the current centroid has
    been subtracted from each observation
    :rtype: numpy array
    """
    return data_matrix - get_center(data_matrix)


def get_cov(data_matrix):
    """Returns a covariance matrix for a given coordinate matrix.

    :param data_matrix: a coordinate matrix where each column is a single observation, shape (3, n)
    :type data_matrix: numpy array
    """
    assert np.allclose(get_center(data_matrix), [0, 0, 0])
    return np.cov(data_matrix, rowvar=True)


def transform_to_spherical(coord_mat, cov_mat):
    """Expects a matrix like array object of three-dimensional points and its related
    covariance matrix and projects it to a spherical representation.
    """
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    eigen_matrix = np.sqrt(np.diag(max(eigen_values) * eigen_values ** -1))

    m = eigen_vectors.T.dot(coord_mat)
    m = eigen_matrix.dot(m)
    return m


def transform_to_origin(coord_mat, cov_mat):
    """Expects a projected, matrix like array object and the covariance matrix
    derived from its origin and projects it back to its original state.
    """
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    eigen_matrix = np.sqrt(np.diag(max(eigen_values) * eigen_values ** -1))

    m = np.linalg.inv(eigen_matrix).dot(coord_mat)
    m = eigen_vectors.dot(m)
    return m


def spherical_projection_matrix(cov_mat):
    """Returns a projection matrix for a nigh spherical representation.

    :param cov_mat: a covariance matrix of shape (3, 3)
    :returns: a projection matrix of shape (3, 3)
    """
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    eigen_matrix = np.sqrt(np.diag(max(eigen_values) * eigen_values ** -1))

    # multiplication is not commutative...
    m = eigen_vectors.dot(eigen_matrix).T
    return m


def origin_projection_matrix(cov_mat, flip=True):
    """Returns a projection matrix that converts a spherisized matrix to its original state.

    :param cov_mat: a covariance matrix of shape (3, 3)
    :param flip: when working with none-projected edges, set this to false. Coordinates will
    be projected to original space, then be flipped to their original position.
    :returns: a projection matrix of shape (3, 3)
    """
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    eigen_matrix = np.sqrt(np.diag(max(eigen_values) * eigen_values ** -1))

    # multiplication is not commutative...
    if flip:
        m = np.linalg.inv(eigen_matrix).dot(eigen_vectors.T).T
    else:
        m = eigen_vectors.dot(np.linalg.inv(
            eigen_matrix).dot(np.linalg.inv(eigen_vectors))).T
    return m

# coordinate conversion


def polar_to_cartesian(vector):
    """Converts a vector in polar coordinate space to cartesian coordinate space.
    spherical coordinates = (r, theta, phi), where
    theta = azimuthal angle
    phi = polar angle

    :param vector: a vector containing three elements
    :type vector: list
    :rtype: numpy array
    """
    x = vector[0] * np.sin(vector[2]) * np.cos(vector[1])
    y = vector[0] * np.sin(vector[2]) * np.sin(vector[1])
    z = vector[0] * np.cos(vector[2])
    return np.array([x, y, z])


def cartesian_to_polar(vector, r=None):
    """Converts a vector in cartesian coordinate space to polar coordinate space.
    spherical coordinates = (r, theta, phi), where
    theta = azimuthal angle
    phi = polar angle

    :param vector: a vector containing three elements
    :type vector: list
    :rtype: numpy array
    """
    if r is None:
        r = np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

    if r == 0:
        return np.array([0., 0., 0.])

    theta = np.arctan2(vector[1], vector[0])
    phi = np.arctan2(np.sqrt(vector[0] ** 2 + vector[1] ** 2), vector[2])
    return np.array([r, theta, phi])

# coordinate creation


def find_perpendicular(v1, v2):
    """Returns the normalized cross product of two vectors. In case vectors are colinear, one of
    them will be offset in order to find a perpendicular vector.

    :param v1, v2: two vectors, represented by numpy arrays, in cartesian coordinate space
    :returns: a vector in cartesian coordinate space, perpendicular to both input vectors
    """
    v_p = np.cross(v1, v2)
    if np.linalg.norm(v_p) == 0:
        v_p = find_perpendicular(v1, np.random.random([3]))
    return v_p / np.linalg.norm(v_p)


def find_angle_delta(v1, v2):
    """Calculates the angle delta between two points, provided their origin lies at [0, 0, 0].

    :param v1, v2: two vectors in cartesian coordinate space
    :returns: the difference in rotation around a perpendicular axis as a floating point number
    """
    v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    return np.arccos(np.dot(v1, v2))


def rodr(vector, axis, theta):
    """Returns a vector ``vector``, rotated around an axis ``axis`` by an angle
    ``theta``. This is used to rotate an origin vector towards a destination
    vector using the methods find_perpendicular and find_angle_delta with
    vector = origin = v1 and destination = v2.

    :param vector: the origin vector, that is to be rotated
    :param axis: a vector perpendicular to vector and a previously provided destination vector
    :param theta: an angle by which vector is to be rotated
    """
    v_r = vector * np.cos(theta) + np.cross(axis, vector) * np.sin(theta) + \
        np.dot(axis, np.dot(axis, vector)) * (1 - np.cos(theta))
    return np.array(v_r)


def create_poly_coords(origin, destination, steps=100, r=1, interpolate=False):
    """Returns a matrix of coordinates of shape (3, steps). Coordinates include
    origin [;,0] and destination [;,-1].
    Inbetween coordinate vectors are interpolated and projected at constant
    radius.

    :params origin, destination: vectors in cartesian coordinate space
    :param steps: the number of inbetween points
    :param r: the constant radius at which to project the inbetween vectors
    """

    # we need to retrieve the point coordinates first, they should already be in
    # the points array of the vtk_poly_data object, later we will probably not have
    # vectors as input arguments but indices
    # we don't need to normalize if we can set the radius manually!
    coords = np.array([origin, destination])

    k = find_perpendicular(origin, destination)
    org_dist = np.sqrt((origin ** 2).sum(axis=0))
    dst_dist = np.sqrt((destination ** 2).sum(axis=0))
    delta = find_angle_delta(origin, destination)

    lerped_coordinates = np.array([
        rodr(polar_to_cartesian(cartesian_to_polar(origin, r=radius(org_dist, dst_dist, i, steps, r))), k, (float(i) / (steps - 1)) * delta) for i in xrange(steps)])
    return np.insert(coords, 1, lerped_coordinates, axis=0).T


def radius(org_dist, dst_dist, i, steps, r=100):
    """Interpolates over a sine function, atop of the interpolation between the
    radii of two projected endpoints.
    """

    r = (org_dist + ((dst_dist - org_dist) * (i / (steps - 1.0)))
         ) + np.sin(i * (np.pi / (steps - 1))) * r
    return r

# vtk operations


def create_points(data_matrix):
    """This creates a vtkPoints object, so that coordinates can be handled in the
    vtk pipeline.

    :param data_matrix: a coordinate matrix where each column is a single observation
    :type data_matrix: numpy array
    :returns: this returns a vtkPoints object, that contains all coordinates extracted
    from the input matrix
    :rtype: vtkPoints
    """
    data_matrix = data_matrix.T
    pts = vtk.vtkPoints()
    for i, point in enumerate(data_matrix):
        pts.InsertPoint(i, point[0], point[1], point[2])
    return pts


def draw_line(vtk_poly_data, pt_0, pt_1):
    """This draws a line in the vtk pipeline.

    :param vtk_poly_data: a vtk object with a points and a cells attribute
    :type vtk_poly_data: vtkPolyData
    :params pt_0, pt_1: two vectors in three-dimensional space
    :type pt_0, pt_1: list
    """
    points = vtk_poly_data.GetPoints()
    lines = vtk_poly_data.GetLines()
    line = vtk.vtkLine()

    points.InsertNextPoint(pt_0)
    points.InsertNextPoint(pt_1)

    line.GetPointIds().SetId(0, points.GetNumberOfPoints() - 2)
    line.GetPointIds().SetId(1, points.GetNumberOfPoints() - 1)

    lines.InsertNextCell(line)
    vtk_poly_data.SetPoints(points)
    vtk_poly_data.SetLines(lines)


def draw_poly_line(vtk_poly_data, coords):
    """Adds a vtkPolyLine with specified coordinates to a poly data object. For this
    the lines property of the poly data object needs to be initialized.

    :param vtk_poly_data: a vtk object with a points and a cells attribute
    :type vtk_poly_data: vtkPolyData
    """
    points = vtk_poly_data.GetPoints()
    lines = vtk_poly_data.GetLines()

    points_offset = vtk_poly_data.GetNumberOfPoints()
    pts = create_points(coords)
    pts_num = pts.GetNumberOfPoints()

    if vtk.VTK_MAJOR_VERSION <= 5:
        for i in xrange(pts_num):
            points.InsertPoint(pts_num + i, points.GetPoint(i))
    else:
        points.InsertPoints(points_offset, pts_num, 0, pts)

    p_line = vtk.vtkPolyLine()
    p_line.GetPointIds().SetNumberOfIds(pts_num)

    for i in range(pts_num):
        p_line.GetPointIds().SetId(i, points_offset + i)

    lines.InsertNextCell(p_line)
    vtk_poly_data.SetPoints(points)
    vtk_poly_data.SetLines(lines)


def get_interpolation_array(n):
    """Generate a balanced array of length ``n``.

    :param n: Number of steps to interpolate in.
    :returns: An array of floats in the interval [0, 1]
    """

    _range = np.arange(0., n)
    _function = lambda x: 1 - ((x - ((len(x) - 1) / 2.)) ** 4 / len(x)
                               ** (np.log((-((len(x) - 1) / 2.)) ** 4) / np.log(len(x))))
    # _function = lambda x: np.sin(x*(np.pi/len(x)))

    interpolation_array = _function(_range)
    interpolation_array *= 0.95

    return interpolation_array


def interpolate_array(a1, a2, interpolation_array):
    """Interpolates two arrays value by value.

    :params a1, a2: two arrays of equal lengths, containing float values.
    :returns: a single array of interpolated values.
    """
    if len(a1) != len(a2):
        raise ValueError('Array lengths do not match!', len(a1), '!=', len(a2))
    # np.array([(a2[i] - a1[i]) * interpolation_array[i] + a1[i] for i in xrange(len(a1))])
    final_array = (a2 - a1).T * interpolation_array + a1.T
    return final_array.T


def interpolate_arrays(a1, a2, interpolation_array, mask):
    # a1 shape should be (n, pts, dimension)
    # interpolation_array shape should be (pts, 1), could use [:, np.newaxis]
    a1[mask] += (a2 - a1[mask]) * interpolation_array[:, np.newaxis]
