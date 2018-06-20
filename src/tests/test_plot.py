import unittest
import vtk
import numpy as np
import math

from .. import plot as pl


class TestPlotFunctions(unittest.TestCase):

    def setUp(self):
        self.points = vtk.vtkPoints()
        self.vtk_poly_data = vtk.vtkPolyData()
        self.coords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        self.lines = vtk.vtkCellArray()

        for i in range(len(self.coords)):
            self.points.InsertPoint(i,
                                    self.coords[i][0],
                                    self.coords[i][1],
                                    self.coords[i][2])
        self.vtk_poly_data.SetPoints(self.points)
        self.vtk_poly_data.SetLines(self.lines)

    # matrix operations

    def test_center_matrix(self):
        centered_points = pl.create_points(pl.center_matrix(pl.get_coords(self.points)))
        self.vtk_poly_data.SetPoints(centered_points)
        self.assertEquals(list(self.vtk_poly_data.GetCenter()), [0, 0, 0])

    def test_get_center(self):
        self.assertEquals(list(pl.get_center(pl.get_coords(self.points))), list(
            self.vtk_poly_data.GetCenter()))

    def test_get_covariance_matrix(self):
        M = pl.center_matrix(pl.get_coords(self.vtk_poly_data))
        cov_mat = pl.get_cov(M)
        self.assertEquals(cov_mat.shape, (3, 3))

    def test_project_with_covariance(self):
        M = pl.get_coords(self.vtk_poly_data)
        cov = np.cov(M)
        M_spherical = pl.transform_to_spherical(M, cov)  # Raffi: looks like a bug that is not testing properly stuff
        M_origin = pl.transform_to_origin(M, cov)
        self.assertTrue(np.array_equal(M, M_origin))

    def test_apply_projection_matrix(self):
        M = pl.get_coords(self.vtk_poly_data)
        cov = np.cov(M)
        f_p = pl.spherical_projection_matrix(cov)
        b_p = pl.origin_projection_matrix(cov)
        self.assertTrue(np.array_equal(b_p.dot(f_p.dot(M)), M))

    # coordinate conversion

    def test_convert_cartesian_to_polar(self):
        v = self.coords[3]
        expected = np.array([1.73205, 0.785398, 0.955317])
        self.assertTrue(np.allclose(pl.cartesian_to_polar(v), expected))

    def test_convert_polar_to_cartesian(self):
        p = [13.5, -37, 31]
        v = pl.cartesian_to_polar(p)
        self.assertTrue(np.allclose(pl.polar_to_cartesian(v), p))

    def test_convert_with_assigned_radius(self):
        v = np.array([-51., -531., -81.])
        self.assertTrue(np.allclose(pl.polar_to_cartesian(
            pl.cartesian_to_polar(v, r=10)), v / np.linalg.norm(v) * 10))

    def test_convert_zero_vector(self):
        self.assertTrue(np.allclose(
            pl.cartesian_to_polar([0, 0, 0]), [0, 0, 0]))

    def test_convert_z_equals_r(self):
        v = [0., 0., 42.]
        self.assertTrue(np.allclose(
            pl.polar_to_cartesian(pl.cartesian_to_polar(v)), v))

    def test_convert_negative_coordinates(self):
        v = np.array([-1., -1., -1.])
        self.assertTrue(np.allclose(
            pl.polar_to_cartesian(pl.cartesian_to_polar(v)), v))

    # coordinate creation

    def test_create_poly_coords(self):
        """Basic checks for the polyline drawing function"""
        pt_0 = np.array([[1., 1., 0.]]).T
        pt_1 = np.array([[1., -1., 0.]]).T

        poly_pts = pl.create_poly_coords(pt_0, pt_1, steps=6, r=10, rshift=1)

        self.assertEqual(len(poly_pts.shape), 3)
        self.assertEqual(poly_pts.shape[2], 6 + 2)

        # basic checks for beginning and end
        self.assertTrue(np.allclose(poly_pts[:, :, 0], pt_0))
        self.assertTrue(np.allclose(poly_pts[:, :, -1], pt_1))

        # all vectors are in the xy plane:
        # iteration over the last dimension
        for vector in poly_pts.T:
            self.assertEqual(vector.T.shape[0], 3)
            self.assertEqual(vector.T[2, 0], 0)

        # check the angles
        # first 2 and last 2 are colinear (only the radius changes)
        self.assertFalse(np.allclose(poly_pts[:, :, 0], poly_pts[:, :, 1]))
        self.assertTrue(np.allclose(poly_pts[:, :, 0] / np.linalg.norm(poly_pts[:, :, 0]),
                                    poly_pts[:, :, 1] / np.linalg.norm(poly_pts[:, :, 1])))

        self.assertFalse(np.allclose(poly_pts[:, :, -1], poly_pts[:, :, -2]))
        self.assertTrue(np.allclose(poly_pts[:, :, -1] / np.linalg.norm(poly_pts[:, :, -1]),
                                    poly_pts[:, :, -2] / np.linalg.norm(poly_pts[:, :, -2])))

        # we have an angle of pi/2 between the 2 end points, and 6 steps, which means
        # 5 segments. There is hence an angle of pi / (2*5) between two consecutive vectors
        for i in range(6 - 1):
            v1 = poly_pts[:, :, 1 + i]
            v2 = poly_pts[:, :, 2 + i]

            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)

            self.assertTrue(np.allclose(np.dot(v1.T, v2), np.cos([math.pi / 10])))

    def test_create_poly_coords_without_shift(self):
        """Basic checks for the polyline drawing function, without additional shift"""
        pt_0 = np.array([[1., 1., 0.]]).T
        pt_1 = np.array([[1., -1., 0.]]).T

        # 0 should be an acceptable rshift
        poly_pts_0 = pl.create_poly_coords(pt_0, pt_1, steps=6, r=10, rshift=0)

        self.assertEqual(len(poly_pts_0.shape), 3)
        self.assertEqual(poly_pts_0.shape[2], 6)

        # rshift defaults to None, same result as 0
        poly_pts = pl.create_poly_coords(pt_0, pt_1, steps=6, r=10)
        self.assertTrue(np.allclose(poly_pts_0, poly_pts))

        self.assertEqual(len(poly_pts.shape), 3)
        self.assertEqual(poly_pts.shape[2], 6)

        # basic checks for beginning and end
        self.assertTrue(np.allclose(poly_pts[:, :, 0], pt_0))
        self.assertTrue(np.allclose(poly_pts[:, :, -1], pt_1))

        # all vectors are in the xy plane:
        # iteration over the last dimension
        for vector in poly_pts.T:
            self.assertEqual(vector.T.shape[0], 3)
            self.assertEqual(vector.T[2, 0], 0)

        # check the angles, no similar consecutive vectors
        self.assertFalse(np.allclose(poly_pts[:, :, 0], poly_pts[:, :, 1]))
        self.assertFalse(np.allclose(poly_pts[:, :, 0] / np.linalg.norm(poly_pts[:, :, 0]),
                                     poly_pts[:, :, 1] / np.linalg.norm(poly_pts[:, :, 1])))

        self.assertFalse(np.allclose(poly_pts[:, :, -1], poly_pts[:, :, -2]))
        self.assertFalse(np.allclose(poly_pts[:, :, -1] / np.linalg.norm(poly_pts[:, :, -1]),
                                     poly_pts[:, :, -2] / np.linalg.norm(poly_pts[:, :, -2])))

        # we have an angle of pi/2 between the 2 end points, and 6 steps, which means
        # 5 segments. There is hence an angle of pi / (2*5) between two consecutive vectors
        for i in range(6 - 1):
            v1 = poly_pts[:, :, i]
            v2 = poly_pts[:, :, 1 + i]

            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)

            self.assertTrue(np.allclose(np.dot(v1.T, v2), np.cos([math.pi / 10])))

    def test_create_poly_coords_radius(self):
        """Checks for the polyline radius function"""
        pt_0 = np.array([[1., 1., 0.]]).T * 10
        pt_1 = np.array([[1., -1., 0.]]).T * 20

        # interpolate the radius to 30 between the 2 end points
        poly_pts = pl.create_poly_coords(pt_0, pt_1, steps=11, r=30)

        self.assertEqual(len(poly_pts.shape), 3)
        self.assertEqual(poly_pts.shape[2], 11)

        # basic checks for beginning and end
        self.assertTrue(np.allclose(poly_pts[:, :, 0], pt_0))
        self.assertTrue(np.allclose(poly_pts[:, :, -1], pt_1))

        # all vectors are in the xy plane:
        # iteration over the last dimension
        for vector in poly_pts.T:
            self.assertEqual(vector.T.shape[0], 3)
            self.assertEqual(vector.T[2, 0], 0)

        # we have an angle of pi/2 between the 2 end points, and 6 steps, which means
        # 5 segments. There is hence an angle of pi / (2*5) between two consecutive vectors
        for i in range(11 - 1):
            v1 = poly_pts[:, :, i]
            v2 = poly_pts[:, :, 1 + i]

            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            self.assertGreaterEqual(v1_norm, 10 * math.sqrt(2))
            self.assertGreaterEqual(v2_norm, 10 * math.sqrt(2))

            self.assertLessEqual(v1_norm, 30 + 20 * math.sqrt(2))
            self.assertLessEqual(v2_norm, 30 + 20 * math.sqrt(2))

            # check the monotonicity
            if i < ((11 - 1) // 2):
                self.assertLessEqual(v1_norm, v2_norm)
            else:
                self.assertGreaterEqual(v1_norm, v2_norm)

    def test_create_poly_coords_colinear(self):
        """Checks of the polyline in case of colinear vector"""

        pt_0 = np.array([[0., 1., 0.]]).T
        pt_1 = np.array([[0., -1., 0.]]).T

        # check that it works
        poly_pts = pl.create_poly_coords(pt_0, pt_1, steps=8, r=1, rshift=1)

        self.assertEqual(len(poly_pts.shape), 3)
        self.assertEqual(poly_pts.shape[2], 8 + 2)

    # vtk operations

    def test_get_coords_shape(self):
        M = pl.get_coords(self.vtk_poly_data)
        self.assertEquals(M.shape, (3, 4))

    def test_get_coords(self):
        M = pl.get_coords(self.vtk_poly_data)
        self.assertTrue(np.array_equal(M, np.array(self.coords).T))

    def test_create_points(self):
        M = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        M = M.T

        self.vtk_poly_data = vtk.vtkPolyData()
        self.vtk_poly_data.SetPoints(pl.create_points(M))
        for i in range(M.shape[1]):
            self.assertTrue(np.array_equal(
                pl.get_coords(self.vtk_poly_data)[:, i], M[:, i]))

    def test_create_points_asserts_when_wrong_shape(self):
        """checks that an assert is fired when the wrong shape of data is given to create_points"""
        M = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        # M = M.T # we check the wrong shape

        with self.assertRaises(AssertionError):
            pl.create_points(M)

    def test_assign_transposed_points(self):
        M_p = pl.create_points(pl.center_matrix(
            pl.get_coords(self.vtk_poly_data)))
        self.vtk_poly_data.SetPoints(M_p)
        self.assertEquals(self.vtk_poly_data.GetNumberOfPoints(), 4)

    def test_append_poly_line(self):
        pl.draw_poly_line(self.vtk_poly_data, self.coords.T)
        self.assertEquals(self.vtk_poly_data.GetNumberOfLines(), 1)

    # test filter internal
    def test_interpolation_array_length(self):
        n = np.random.randint(1, 100)
        array = pl.get_interpolation_array(n)
        self.assertEquals(len(array), n)

    def test_interpolation_array_balance(self):
        n = 8
        array = pl.get_interpolation_array(n)
        self.assertTrue(np.allclose(array[4:], np.flipud(array[:4])))

    def test_interpolation_array_application(self):
        pass


class TestCoordinateTransform(unittest.TestCase):
    """Test various coordinate transformations"""

    def test_spherical_transform(self):

        # generate many points, each being a column
        nb_samples = 1000
        points = (100 * np.random.random((3, nb_samples)))

        transform_object = pl.TransformSpherical(coordinates=points)

        self.assertTrue(np.allclose(transform_object.mean, np.mean(points, axis=1, keepdims=1)))

        points_centered = points - transform_object.mean
        variance_each_axis = np.var(points_centered, axis=1) * nb_samples / (nb_samples - 1)  # var uses another normalization
        self.assertTrue(np.allclose(np.diag(transform_object.covariance), variance_each_axis))

    def test_to_and_from_spherical_idempotency(self):

        # generate many points, each being a column
        points = (100 * np.random.random((3, 1000)))

        transform_object = pl.TransformSpherical(coordinates=points)

        self.assertTrue(np.allclose(transform_object.mean, np.mean(points, axis=1, keepdims=1)))

        points_centered = points - transform_object.mean
        self.assertTrue(np.allclose(np.mean(points_centered, axis=1), np.zeros(3)))

        transformed_points = transform_object.transform_to_spherical(points_centered)
        transformed_points_back = transform_object.transform_to_origin(transformed_points)

        self.assertTrue(np.allclose(transformed_points_back, points_centered))


class TestGeometricUtilities(unittest.TestCase):

    def test_multiple_find_perpendicular(self):
        """Smoke test for the find_perpendicular function"""

        nb_samples = 1000
        point_set1 = (100 * np.random.random((3, nb_samples)))
        point_set2 = (100 * np.random.random((3, nb_samples)))

        perpendicular_vectors = pl.find_perpendicular(point_set1, point_set2)

        self.assertEqual(perpendicular_vectors.shape[0], 3)
        self.assertEqual(perpendicular_vectors.shape[1], nb_samples)

    def test_multiple_find_perpendicular_all_normalized(self):

        nb_samples = 100
        point_set1 = (100 * np.random.random((3, nb_samples)))
        point_set2 = (100 * np.random.random((3, nb_samples)))

        perpendicular_vectors = pl.find_perpendicular(point_set1, point_set2)
        for vector in perpendicular_vectors.T:
            self.assertTrue(np.allclose(np.sqrt((vector ** 2).sum(axis=0)), 1))

        # this holds true for the cross product yielding zeros
        perpendicular_vectors = pl.find_perpendicular(point_set1, point_set1)
        for vector in perpendicular_vectors.T:
            self.assertTrue(np.allclose(np.sqrt((vector ** 2).sum(axis=0)), 1))

    def test_multiple_find_perpendicular_returns_perpendicular(self):
        """Checks the perpendicularity of the returned vectors"""

        nb_samples = 1000
        point_set1 = (100 * np.random.random((3, nb_samples)))
        point_set2 = (100 * np.random.random((3, nb_samples)))

        perpendicular_vectors = pl.find_perpendicular(point_set1, point_set2)

        for vector1, vector2, vector3 in zip(perpendicular_vectors.T, point_set1.T, point_set2.T):
            self.assertTrue(np.allclose(np.dot(vector1, vector2), 0))
            self.assertTrue(np.allclose(np.dot(vector1, vector3), 0))

        # same for the perpendicularity yielding zeros
        perpendicular_vectors = pl.find_perpendicular(point_set1, point_set1)

        for vector1, vector2 in zip(perpendicular_vectors.T, point_set1.T):
            self.assertTrue(np.allclose(np.dot(vector1, vector2), 0))
