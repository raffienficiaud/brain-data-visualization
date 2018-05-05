import numpy as np
import sys
import os

sys.path = ['', '/Users/Lennart/Desktop/brain-data-viz', '/Users/Lennart/Library/Python/2.7/bin', '/usr/local/Cellar/vtk/7.1.1/lib', '/usr/local/Cellar/vtk/7.1.1/bin', '/Users/Lennart', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python27.zip', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-darwin', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac',
            '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac/lib-scriptpackages', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-tk', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-old', '/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload', '/Users/Lennart/Library/Python/2.7/lib/python/site-packages', '/usr/local/lib/python2.7/site-packages', '/usr/local/Cellar/numpy/1.12.1/libexec/nose/lib/python2.7/site-packages']

from brain_data_viz import plot as pl
reload(pl)

from brain_data_viz import naive_k_means as nm
reload(nm)

# initialize input object of type vtkPolyData
poly_data_input = self.GetPolyDataInput()
pts = pl.get_coords(poly_data_input)
pts_center = pl.get_center(pts)
pts = pl.center_matrix(pts)

# Calculation of kmeans and polyline coordinates
#########################################
# store covariance of coordinates
cov = np.array([[  976.11039718,   -84.83226021,    -3.22440051],[  -84.83226021,  1404.36336796,    38.85441103],[   -3.22440051,    38.85441103,   294.69293392]])
# get the backward projection matrix
b_p = pl.origin_projection_matrix(cov)

pts = b_p.dot(pts)
pids = range(len(pts.T))
print pids
vert_subset = vtk.vtkCellArray()

vert_subset.InsertNextCell(len(pids), pids)

# initialize output polydata object
output = self.GetPolyDataOutput()
output.Allocate(1, 1)
output.SetPoints(pl.create_points(pts))
output.SetVerts(vert_subset)
