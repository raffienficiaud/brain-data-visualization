import numpy as np
import sys
import os

if not bdv_path in sys.path:
    sys.path.append(bdv_path)

from brain_data_viz import plot as pl
from brain_data_viz import naive_k_means as nm

reload(pl)
reload(nm)

line_res = 16
pts_offset = line_res + 3

# initialize input object of type vtkPolyData
poly_data_input = self.GetPolyDataInput()
pts = pl.get_coords(poly_data_input)
pts_center = pl.get_center(pts)
pts = pl.center_matrix(pts)

# store covariance of coordinates
cov = pl.get_cov(pts)
# get the backward projection matrix
b_p = pl.origin_projection_matrix(cov)

print '#####'

###################################
# Loading/Generation of mean data #
###################################

# load prepared edges from file
try:
    projected_edges = np.load(edges_path)
    edges = projected_edges[edge_selection[0]:edge_selection[1]]
except IOError:
    return 'Please select a valid file path to load edge information from.'

# load kmeans data from file, alternatively calculate new kmeans
try:
    means = np.load(means_path)
    print 'K-Means data loaded succesfully.'
except IOError:
    means = nm.kmeans(k_means, edges)
    print 'Calculating kmeans...'

# choose means and edges to display
if mean_index > -1:
    mean_edges = np.array([means[0][mean_index]])
    edges = nm.fetch_edges(mean_index, edges, means[0])
else:
    mean_edges = means[0]

# labels assign each edge to a centroid
labels = means[1]

#######################
# Generation of Lines #
#######################

# initialize line_sets to add to output
line_sets = np.array([[], [], []])
mean_sets = np.array([[], [], []])
edge_sets = np.array([[], [], []])

for line in mean_edges:
    line_pts = pl.create_poly_coords(line[3:], line[:3], r=240, steps=line_res)
    mean_sets = np.append(mean_sets, line_pts, axis=1)

if show_edges:
    for i in xrange(edges.shape[0]):
        line_pts = pl.create_poly_coords(
            edges[:, :3][i], edges[:, 3:][i],
            r=100, steps=line_res)
        edge_sets = np.append(edge_sets, line_pts, axis=1)

mean_sets = b_p.dot(mean_sets)
edge_sets = b_p.dot(edge_sets)

line_sets = np.append(line_sets, mean_sets, axis=1)

if show_edges:
    for i in xrange(edges.shape[0]):
        mean_id = labels[i] if mean_index == -1 else 0
        line_pts = np.flipud(edge_sets[:,i*pts_offset:(i+1)*pts_offset].T)
        line_pts = pl.interpolate_array(line_pts, mean_sets[:,mean_id*pts_offset:(mean_id+1)*pts_offset].T)
        line_sets = np.append(line_sets, line_pts.T, axis=1)

# project back line coordinates
#line_sets = b_p.dot(line_sets)
line_sets += pts_center
#########################################

lines = vtk.vtkCellArray()

# initialize output polydata object
output = self.GetPolyDataOutput()
output.Allocate(1, 1)
output.SetPoints(pl.create_points(pts))
output.SetLines(lines)

# after initializing everything, we can append the lines to the output
for line in np.split(line_sets, line_sets.shape[1] / pts_offset, axis=1):
    pl.draw_poly_line(output, line)

# adding colors
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(4)
colors.SetName('Colors!')
# labels = means[1]
index_max = np.amax(labels) * 1.0
for i in xrange(mean_edges.shape[0]):
    # the following line will initialize the mean edges with a red color
    colors.InsertNextTuple4(255, 69, 69, 255)
    index = i if mean_index == -1 else mean_index
    #colors.InsertNextTuple4((1 - index / index_max) * 169.,
    #                        (index / index_max) * 255., 200, 255)

for i in xrange(edges.shape[0]):
    index = labels[i] if mean_index == -1 else mean_index
    colors.InsertNextTuple4((1 - index / index_max) *
                            169., (index / index_max) * 255., 200, 255)

output.GetCellData().SetScalars(colors)
del colors
