import sys, os
sys.path = ['', '/Users/Lennart/Desktop/brain-data-viz', '/Users/Lennart/Library/Python/2.7/bin', '/usr/local/Cellar/vtk/7.1.1/lib', '/usr/local/Cellar/vtk/7.1.1/bin', '/Users/Lennart', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python27.zip', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-darwin', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac',
            '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac/lib-scriptpackages', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-tk', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-old', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload', '/Users/Lennart/Library/Python/2.7/lib/python/site-packages', '/usr/local/lib/python2.7/site-packages', '/usr/local/Cellar/numpy/1.12.1/libexec/nose/lib/python2.7/site-packages']

import numpy as np
from brain_data_viz import plot as pl
reload(pl)

# initialize input and output objects of type vtkPolyData
input = self.GetPolyDataInput()
output = self.GetPolyDataOutput()

pts = pl.get_coords(input)
pts_center = pl.get_center(pts)
pts = pl.center_matrix(pts)
point_data_length = input.GetNumberOfPoints()

#########################################
nifti_path = 'Users/Lennart/Desktop/brain-data-viz/data/map.nii'

# reading files
# get voxel coordinates in nifti
nifti = pl.nifti_fast(nifti_path)
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
alignment_matrix = np.array([[x_ex, 0, 0], [0, -y_ex, 0], [0, 0, -z_ex]])

nifti = alignment_matrix.dot(nifti)
#nifti += nifti_center
nifti += pts_center
#########################################

point_subset = pl.create_points(nifti)
vert_subset = vtk.vtkCellArray()

# other than in the above example we can also add all vertices in a single cell like so
# one method for older vtk versions
pids = [i for i in xrange(point_subset.GetNumberOfPoints())]
if vtk.VTK_MAJOR_VERSION <= 6:
    vert_subset.InsertNextCell(len(pids))
    for i in range(0, len(pids)):
        vert_subset.InsertCellPoint(i)
else:
    vert_subset.InsertNextCell(len(pids), pids)

# allocate memory for output object, we only have one cell to allocate for
output.Allocate(1, 1)
output.SetPoints(point_subset)
output.SetVerts(vert_subset)
