import numpy as np
from vtk import *
from brain_data_viz import plot as pl
from brain_data_viz import file_ops as f

nifti_path = "/Users/raffi/OwncloudSync/Shared/brain-data-visualisation/data/map.nii"
csv_path =  "/Users/raffi/OwncloudSync/Shared/brain-data-visualisation/data/mat.csv"

# the same brain mesh as opened in paraview
file_name = '/Users/raffi/OwncloudSync/Shared/brain-data-visualisation/data/levelset_reinit_inf.vtk'
# alternatively load the mesh...
# make sure pts holds 3D coordinates in the shape (d, n)
# where d is the number of dimensions and n the number of points
reader = vtk.vtkPolyDataReader()
reader.SetFileName(file_name)
reader.Update()

poly_data_input = reader.GetOutput()
pts = pl.get_coords(poly_data_input)

# get voxel coordinates in nifti
nifti = f.read_nifti(nifti_path)
# nifti_cov = pl.get_cov(nifti)
nifti_center = pl.get_center(nifti)

# aligns the matrix with the mesh in this specific case
alignment_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
nifti -= nifti_center

# find min and max extents of nifti
n_min = [np.amin(nifti[0]), np.amin(nifti[1]), np.amin(nifti[2])]
n_max = [np.amax(nifti[0]), np.amax(nifti[1]), np.amax(nifti[2])]

# find min and max extents of mesh
m_min = [np.amin(pts[0]), np.amin(pts[1]), np.amin(pts[2])]
m_max = [np.amax(pts[0]), np.amax(pts[1]), np.amax(pts[2])]

# compute factor by which to multiply nifti coordinates
x_ex = (abs(m_min[0]) + m_max[0]) / (abs(n_min[0]) + n_max[0])
y_ex = (abs(m_min[1]) + m_max[1]) / (abs(n_min[1]) + n_max[1])
z_ex = (abs(m_min[2]) + m_max[2]) / (abs(n_min[2]) + n_max[2])

# flips over the y and z axis
alignment_matrix = np.array([[x_ex, 0, 0], [0, -y_ex, 0], [0, 0, -z_ex]])

nifti = alignment_matrix.dot(nifti)
nifti += nifti_center

# get indices of connections
indices = f.read_csv(csv_path)

edges = np.array([nifti[:, [int(index[0]), int(
        index[1])]].T.flatten() for index in indices])

np.save('edges', edges)

# output.SetPoints(pl.create_points(np.load('Users/Lennart/Desktop/brain-data-viz/data/map.npy')))
# vert_subset = vtk.vtkCellArray()
# vert_subset.InsertNextCell(415197, np.arange(415197))
# output.SetVerts(vert_subset)
