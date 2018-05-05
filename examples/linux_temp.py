import numpy as np
import sys
import os

sys.path = ['/is/sg/lbramlage/env_py/lib/python2.7/site-packages', '/is/sg/lbramlage/env_py/include', '/is/sg/lbramlage/env_py/lib', '/is/sg/lbramlage/env_py', '/is/sg/lbramlage/env_py/bin', '/is/sg/lbramlage/paraview/lib/paraview-5.4', '/is/sg/lbramlage/paraview/lib/python27.zip', '/is/sg/lbramlage/paraview/lib/python2.7',
            '/is/sg/lbramlage/paraview/lib/python2.7/plat-linux2', '/is/sg/lbramlage/paraview/lib/python2.7/lib-tk', '/is/sg/lbramlage/paraview/lib/python2.7/lib-old', '/is/sg/lbramlage/paraview/lib/python2.7/lib-dynload', '/is/sg/lbramlage/paraview/lib/python2.7/site-packages', '/is/sg/lbramlage/brain-data-viz']

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
cov = pl.get_cov(pts)
# get the backward projection matrix
b_p = pl.origin_projection_matrix(cov)

# load prepared edges from file
projected_edges = np.load('../../brain-data-viz/data/projected_edges.npy')

# create a bunch of coordinate sets in spherical space
line_sets = np.array([[], [], []])
print "### ###"

edges = projected_edges[50000:50200]

for i in xrange(edges.shape[0]):
    line_pts = pl.create_poly_coords(
        edges[:, :3][i], edges[:, 3:][i],
        r=200, steps=16)
    line_sets = np.append(line_sets, line_pts, axis=1)

means = np.load('../../brain-data-viz/data/8m-200e.npy')
mean_lines = means[0]

for line in mean_lines:
    line_pts = pl.create_poly_coords(line[3:], line[:3], r=240, steps=16)
    line_sets = np.append(line_sets, line_pts, axis=1)

# project back line coordinates
line_sets = b_p.dot(line_sets)
line_sets += pts_center
#########################################

lines = vtk.vtkCellArray()

# initialize output polydata object
output = self.GetPolyDataOutput()
output.Allocate(1, 1)
output.SetPoints(pl.create_points(pts))
output.SetLines(lines)

# after initializing everything we need, we can append the lines
for line in np.split(line_sets, edges.shape[0] + mean_lines.shape[0], axis=1):
    pl.draw_poly_line(output, line)

# adding colors
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(4)
colors.SetName("Colors!")
labels = means[1]
index_max = np.amax(labels) * 1.0
for i in xrange(edges.shape[0]):
    index = labels[i]
    colors.InsertNextTuple4((1 - index / index_max) *
                            169., (index / index_max) * 255., 200, 120)

for i in xrange(mean_lines.shape[0]):
    # colors.InsertNextTuple3(255, 69, 69)
    colors.InsertNextTuple4((1 - i / index_max) * 169.,
                             (i / index_max) * 255., 200, 255)

output.GetCellData().SetScalars(colors)
del colors
