# -*- coding: utf-8 -*-

"""Line and points construction functions for the brain visualization


"""
import numpy as np


def get_center(data_matrix):
    """Returns the centroid of a matrix.

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
    """Returns a projection matrix to spherical representation.

    :param cov_mat: a covariance matrix of shape (3, 3)
    :returns: a projection matrix of shape (3, 3)
    """
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    eigen_matrix = np.sqrt(np.diag(max(eigen_values) * eigen_values ** -1))

    # multiplication is not commutative...
    m = eigen_vectors.dot(eigen_matrix).T
    return m


class TransformSpherical(object):

    def __init__(self,
                 coordinates=None,
                 mean=None,
                 covariance=None):
        """
        :param coordinates: each column indicates a data point. This parameter should be passed
          if the covariance or the mean is not provided. If the mean is provided and covariance
          not, the coordinates should have already been centered
        """

        self.mean = mean
        if self.mean is None:
            assert(coordinates is not None)
            self.mean = get_center(coordinates)
            coordinates = coordinates - self.mean
        self.covariance = covariance
        if self.covariance is None:
            assert(coordinates is not None)
            self.covariance = get_cov(coordinates)

        self.eigen_values, self.eigen_vectors = np.linalg.eigh(self.covariance)
        self.eigen_matrix = np.sqrt(np.diag(max(self.eigen_values) * (self.eigen_values ** -1)))
        self.inv_eigen_matrix = np.sqrt(np.diag(self.eigen_values / max(self.eigen_values)))

    def transform_to_spherical(self, coordinates):
        """Applies a linear transformation for making the data isotropic

        This transformation is based on the previous covariance matrix. The data
        should have been previously centered
        """

        m = self.eigen_vectors.T.dot(coordinates)
        m = self.eigen_matrix.dot(m)

        return m

    def transform_to_origin(self, coordinates):
        """Applies the inverse linear transformation as for :py:meth:`.transform_to_spherical`
        """
        m = self.inv_eigen_matrix.dot(coordinates)
        m = self.eigen_vectors.dot(m)
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
    """Returns the normalized cross product of two vectors. In case vectors are colinear, a random
    orthogonal vector is returned instead.

    :param v1, v2: two arrays of 3D vectors, represented by numpy arrays, in cartesian coordinate space.
      Each column represents a vector.
    :returns: an array of 3D vector in cartesian coordinate space, where each column is orthogonal
      to the corresponding columns in ``v1`` and ``v2``. Each returned vector has a norm of ``1``.

    .. note::

       In case a pair of vectors is colinear, a random perpendicular vector is returned instead.
    """
    assert v1.shape[0] == 3
    assert v2.shape[0] == 3
    v_p = np.cross(v1.T, v2.T).T
    norm_vp = np.linalg.norm(v_p, axis=0)
    elements_colinear = norm_vp == 0
    elements_colinear_subset = v_p[:, elements_colinear]

    if elements_colinear_subset.shape[1] != 0:
        random_vectors = np.random.random((3, elements_colinear_subset.shape[1]))
        v1_subset_directions = v1[:, elements_colinear]
        v1_subset_directions /= np.linalg.norm(v1_subset_directions, axis=0)
        # the second term below contains the dot product of the pairs of vectors in v1 and random_vectors
        orthogonal_directions_quantities = (v1_subset_directions * random_vectors).sum(axis=0)
        v_p[:, elements_colinear] = random_vectors - v1_subset_directions * orthogonal_directions_quantities
        norm_vp[elements_colinear] = np.linalg.norm(v_p[:, elements_colinear], axis=0)
        pass

    return v_p / norm_vp


def find_angle_delta(v1, v2):
    """Calculates the angle between two vectors in $\R^3$.

    :param v1, v2: two vectors in cartesian coordinate space
    :returns: the difference in rotation around a perpendicular axis as a floating point number
    """

    assert len(v1.shape) == 2
    assert v1.shape[0] == 3
    assert len(v2.shape) == 2
    assert v2.shape[0] == 3

    # here we may have computed the norms already, to optimize a bit
    v1, v2 = v1 / np.linalg.norm(v1, axis=0), v2 / np.linalg.norm(v2, axis=0)
    # return np.arccos(np.dot(v1, v2))
    # better as it handles some weird normalization cases
    # return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    dotproduct = (v1 * v2).sum(axis=0)
    return np.arccos(np.clip(dotproduct, -1.0, 1.0))


def rodrigues(vector, axis, theta):
    """Returns a vector ``vector``, rotated around an axis ``axis`` by an angle
    ``theta``. This is used to rotate an origin vector towards a destination
    vector using the methods find_perpendicular and find_angle_delta with
    vector = origin = v1 and destination = v2.

    :param vector: the origin vector, that is to be rotated
    :param axis: a vector perpendicular to vector and a previously provided destination vector
    :param theta: an angle by which vector is to be rotated
    """

    assert len(vector.shape) == 2
    assert vector.shape[0] == 3
    assert len(axis.shape) == 2
    assert axis.shape[0] == 3

    # can be cached
    all_cross_products = np.cross(axis.T, vector.T).T
    dotproducts = (axis * vector).sum(axis=0)

    #
    costheta = np.cos(theta)

    v_r = vector * costheta + \
        all_cross_products * np.sin(theta) + \
        axis * dotproducts * (1 - costheta)
    return v_r


def create_poly_coords(origin,
                       destination,
                       steps=100,
                       r=1,
                       rshift=None):
    """Creates a polygonal approximation of a 3D curve joining two points at constant radius
    from a center.


    Returns a matrix of coordinates of shape (3, steps). Coordinates include
    origin [;,0] and destination [;,-1].
    In between coordinate vectors are interpolated and projected at constant
    radius.

    :params origin, destination: vectors in cartesian coordinate space
    :param steps: the number of intermediate points for each curve
    :param r: the constant radius at which to project the inbetween vectors
    :param rshift: radius shift to be applied to each radius between first and last coordinates.
    """

    # we need to retrieve the point coordinates first, they should already be in
    # the points array of the vtk_poly_data object, later we will probably not have
    # vectors as input arguments but indices
    # we don't need to normalize if we can set the radius manually!

    # coords = np.array([origin, destination])

    assert len(origin.shape) == 2
    assert origin.shape[0] == 3
    assert len(destination.shape) == 2
    assert destination.shape[0] == 3

    k = find_perpendicular(origin, destination)

    org_dist = np.linalg.norm(origin, axis=0)
    dst_dist = np.linalg.norm(destination, axis=0)

    delta = find_angle_delta(origin, destination)

    if rshift:
        out = np.ndarray((origin.shape[0], origin.shape[1], steps + 2))
        out[:, :, 0] = origin
        out[:, :, steps + 1] = destination
    else:
        out = np.ndarray((origin.shape[0], origin.shape[1], steps))

    # we have to normalize origin otherwise we multiply by a radius twice (in the loop)
    normalized_origin = origin / org_dist

    for current_step in xrange(steps):
        radius_step = radius(org_dist, dst_dist, current_step, steps, r)
        if rshift:
            radius_step += rshift
        current_rotated = radius_step * rodrigues(vector=normalized_origin,
                                                  axis=k,
                                                  theta=(delta * float(current_step) / (steps - 1)))
        if rshift:
            out[:, :, current_step + 1] = current_rotated
        else:
            out[:, :, current_step] = current_rotated

    return out


def radius(org_dist, dst_dist, current_step, nb_steps, r=100):
    """Interpolates over a sine function, atop of the interpolation between the
    radii of two projected endpoints.
    """

    alpha = current_step / (nb_steps - 1.0)
    r = (1 - alpha) * org_dist + alpha * dst_dist + np.sin(np.pi * alpha) * r

    # combining the sine with a sigmoide might also give interesting results
    # points(xx, ((1+exp(-beta))/(1+exp(-beta*sin(xx))) - 0.5)*2, col='purple')
    # or even better (which can be further simplified)
    # points(xx, (1/(1+exp(-beta*sin(xx))) - 0.5)*2*(1+exp(-beta))/(1-exp(-beta)), col='red')
    return r

# vtk operations


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


def create_points(data_matrix):
    """This creates a vtkPoints object, so that coordinates can be handled in the
    vtk pipeline.

    :param data_matrix: a coordinate matrix where each column is a single observation
    :type data_matrix: numpy array
    :returns: vtkPoints object containing all coordinates extracted from the input matrix
    :rtype: vtkPoints
    """
    import vtk
    from vtk.util import numpy_support

    if 0:
        data_matrix = data_matrix.T
        pts = vtk.vtkPoints()
        for i, point in enumerate(data_matrix):
            pts.InsertPoint(i, point[0], point[1], point[2])
        return pts

    assert data_matrix.shape[0] == 3, "The number of components/lines of the data matrix should be equal to 3 (3d coordinates)"

    vtkfloatarray = numpy_support.numpy_to_vtk(num_array=data_matrix.astype(np.float).ravel(order='F'),  # warning on the order
                                               deep=True,
                                               array_type=vtk.VTK_FLOAT)

    pts = vtk.vtkPoints()
    vtkfloatarray.SetNumberOfComponents(3)
    pts.SetData(vtkfloatarray)
    return pts


def draw_line(vtk_poly_data, pt_0, pt_1):
    """This draws a line in the vtk pipeline.

    :param vtkPolyData vtk_poly_data: a vtk object with a points and a cells attribute
    :params pt_0, pt_1: two vectors in three-dimensional space
    :type pt_0, pt_1: list
    """
    import vtk
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

    .. note::

        Requires VTK
    """
    import vtk

    points = vtk_poly_data.GetPoints()
    if points is None:
        points = vtk.vtkPoints()
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
    vtk_poly_data.SetPoints(points)  # needed in the case created locally
    vtk_poly_data.SetLines(lines)


def get_interpolation_array(n):
    """Generate a balanced array of length ``n``.

    :param n: Number of steps to interpolate in.
    :returns: An array of floats in the interval [0, 1]
    """

    _range = np.arange(0., n)

    def _function(x):
        return 1 - ((x - ((len(x) - 1) / 2.)) ** 4 / len(x) ** (np.log((-((len(x) - 1) / 2.)) ** 4) / np.log(len(x))))
    # _function = lambda x: np.sin(x*(np.pi/len(x)))

    interpolation_array = _function(_range)
    interpolation_array *= 0.95

    return interpolation_array
