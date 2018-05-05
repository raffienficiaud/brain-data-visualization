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
cov = pl.get_cov(pts)
# get the backward projection matrix
b_p = pl.origin_projection_matrix(cov)

# load prepared edges from file
projected_edges = np.load('Users/Lennart/Desktop/brain-data-viz/data/projected_edges.npy')

# create a bunch of coordinate sets in spherical space
line_sets = np.array([[], [], []])
print "######"

edges = projected_edges[50000:50128]
#edges = projected_edges[0:128]

### LINES
for i in xrange(edges.shape[0]):
    line_pts = pl.create_poly_coords(
        edges[:, :3][i], edges[:, 3:][i],
        r=200, steps=16)
    line_sets = np.append(line_sets, line_pts, axis=1)
###

#means = np.load('Users/Lennart/Desktop/brain-data-viz/data/means/8m-200e.npy')
#means = np.load('Users/Lennart/Desktop/brain-data-viz/data/means/6m-0;128.npy')
means = nm.kmeans(6, edges)
mean_lines = means[0]

for line in mean_lines:
    line_pts = pl.create_poly_coords(line[3:], line[:3], r=240, steps=16)
    line_sets = np.append(line_sets, line_pts, axis=1)

print line_sets.shape
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
for line in np.split(line_sets, line_sets.shape[1] / 19, axis=1):
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
