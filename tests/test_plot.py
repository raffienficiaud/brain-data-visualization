import unittest, csv, os, vtk
from .context import plot as pl
from .context import matrix_ops as mat_ops
from .context import vtk_ops
from .context import file_ops as fo
import numpy as np
import nibabel as nib


class TestPlotFunctions(unittest.TestCase):

    def setUp(self):
        self.points = vtk.vtkPoints()
        self.vtk_poly_data = vtk.vtkPolyData()
        self.coords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        self.lines = vtk.vtkCellArray()

        self.csv_path = os.path.join(os.getcwd(), 'test.csv')
        self.nii_path = os.path.join(os.getcwd(), 'test.nii')

        for i in range(len(self.coords)):
            self.points.InsertPoint(i, self.coords[i][0], self.coords[
                                    i][1], self.coords[i][2])
        self.vtk_poly_data.SetPoints(self.points)
        self.vtk_poly_data.SetLines(self.lines)

# matrix operations

    def test_center_matrix(self):
        centered_points = vtk_ops.create_points(
            mat_ops.center_matrix(mat_ops.get_coords(self.points)))
        self.vtk_poly_data.SetPoints(centered_points)
        self.assertEquals(list(self.vtk_poly_data.GetCenter()), [0, 0, 0])

    def test_get_center(self):
        self.assertEquals(list(mat_ops.get_center(mat_ops.get_coords(self.points))), list(
            self.vtk_poly_data.GetCenter()))

    def test_get_covariance_matrix(self):
        M = mat_ops.center_matrix(mat_ops.get_coords(self.vtk_poly_data))
        cov_mat = mat_ops.get_cov(M)
        self.assertEquals(cov_mat.shape, (3, 3))

    def test_project_with_covariance(self):
        M = mat_ops.get_coords(self.vtk_poly_data)
        cov = np.cov(M)
        M_spherical = mat_ops.transform_to_spherical(M, cov)
        M_origin = mat_ops.transform_to_origin(M, cov)
        self.assertTrue(np.array_equal(M, M_origin))

    def test_apply_projection_matrix(self):
        M = mat_ops.get_coords(self.vtk_poly_data)
        cov = np.cov(M)
        f_p = mat_ops.spherical_projection_matrix(cov)
        b_p = mat_ops.origin_projection_matrix(cov)
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
        # exactly in this case we want something else to happen!!!
        pt_0 = np.array([1., 1., 0.])
        pt_1 = np.array([1., -1., 0.])
        poly_pts = pl.create_poly_coords(pt_0, pt_1, steps=6, r=10)
        self.assertTrue(np.allclose(np.array(poly_pts[:, 4], dtype=np.int32), np.array([10., -1., 0.])))

    def test_create_poly_coords_dst_org(self):
        pt_0 = np.array([0., 1., 0.])
        pt_1 = np.array([0., -1., 0.])
        poly_pts = pl.create_poly_coords(pt_0, pt_1, steps=8, r=1)
        self.assertTrue(np.allclose(poly_pts[:, 0], np.array([0., 1., 0.])))
        self.assertTrue(np.allclose(
            poly_pts[:, len(poly_pts.T) - 1], np.array([0., -1., 0.])))

    def test_create_poly_coords_colinear(self):
        # how to test this?
        pt_0 = np.array([0., 1., 0.])
        pt_1 = np.array([0., -1., 0.])
        poly_pts = pl.create_poly_coords(pt_0, pt_1, steps=8, r=1)

# vtk operations

    def test_get_coords_shape(self):
        M = pl.get_coords(self.vtk_poly_data)
        self.assertEquals(M.shape, (3, 4))

    def test_get_coords(self):
        M = pl.get_coords(self.vtk_poly_data)
        self.assertTrue(np.array_equal(M, np.array(self.coords).T))

    def test_create_points(self):
        M = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.vtk_poly_data.SetPoints(pl.create_points(M))
        self.assertTrue(np.array_equal(
            pl.get_coords(self.vtk_poly_data)[0], M[0]))

    def test_assign_transposed_points(self):
        M_p = pl.create_points(pl.center_matrix(
            pl.get_coords(self.vtk_poly_data)))
        self.vtk_poly_data.SetPoints(M_p)
        self.assertEquals(self.vtk_poly_data.GetNumberOfPoints(), 4)

    def test_append_poly_line(self):
        pl.draw_poly_line(self.vtk_poly_data, self.coords.T)
        self.assertEquals(self.vtk_poly_data.GetNumberOfLines(), 1)

# nifti operations

    def test_nifti_reader(self):
        mat = np.array([[[123.]]])
        affine = np.diag([1, 1, 1, 1])
        img = nib.Nifti1Image(mat, affine)
        nii_path = os.path.join(os.getcwd(), 'test.nii')
        img.to_filename(nii_path)

        self.assertTrue(np.allclose([[ 0.], [ 0.], [ 0.]], fo.nifti_fast(nii_path)))

        os.remove(nii_path)

    def test_csv_reader(self):
        with open('test.csv', 'wb') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            for row in self.coords:
                writer.writerow(row)

        csv_file = fo.csv_reader(self.csv_path)
        self.assertTrue(np.array_equal(self.coords, csv_file))
        os.remove(self.csv_path)

# test filter interna
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


if __name__ == '__main__':
    unittest.main()
