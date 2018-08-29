# Running VTK
from examples.nifti_alignment import nifti

#export DYLD_LIBRARY_PATH=/Applications/ParaView-5.4.1.app/Contents/Libraries/:$DYLD_LIBRARY_PATH

#>> python
#import sys
#sys.path.append("/Applications/ParaView-5.4.1.app/Contents/Python")

# easier
# start pvpython
# > /Applications/ParaView-5.4.1.app/Contents/bin/pvpython
# import sys
# sys.path.append("/Users/raffi/code/sandbox/venv_paraview/lib/python2.7/site-packages")

import sys
sys.path.insert(0, '/Users/raffi/usr/local/lib/python2.7/site-packages/')

import numpy as np
from vtk import *
from mpi_is_sw.brain_connectivity import plot as PP
from mpi_is_sw.brain_connectivity import file_ops as FF
import os

origin_path = "/Users/raffi/Documents/owncloud/Projects/BrainVisualization/"
#origin_path = "/home/renficiaud/Documents/Owncloud/Projects/BrainVisualization/"

nifti_path = os.path.join(origin_path, "new-data/3dvis/map.nii")
csv_path = os.path.join(origin_path, "new-data/3dvis/mat.csv")

# the same brain mesh as opened in paraview
mesh_file_name = os.path.join(origin_path, "new-data/3dvis/fmri_lvls_csf_gm_bf_FIX_reinit.vtk")

if 0:
    # alternatively load the mesh...
    # make sure mesh_points holds 3D coordinates in the shape (d, n)
    # where d is the number of dimensions and n the number of points
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(mesh_file_name)
    reader.Update()

    poly_data_input = reader.GetOutput()
    mesh_points = PP.get_coords(poly_data_input)

# volumetric data
import nibabel as nib

# saving the 3D points to a VTK file
# trying to save the nifty

# get voxel coordinates in nifti
nifti = FF.pointset_from_indicator_nifti_file(nifti_path)

vtkpoints = PP.create_points(nifti)
verts = vtk.vtkCellArray()
verts = FF.get_vtkcellarray_from_vtkpoints(vtkpoints)

writer = vtk.vtkPolyDataWriter()
writer.SetFileName('test6.vtk')

polydata = vtk.vtkPolyData()
polydata.SetPoints(vtkpoints)
polydata.SetVerts(verts)
writer.SetInputData(polydata)
writer.Update()
writer.Write()

##
##

edges = FF.assemble_edges(nifti_path, csv_path)
np.save('edges2', edges.T)

##
##
##

# testing with VTK tools
from vtk.numpy_interface import dataset_adapter as dsa
test_array = dsa.numpyTovtkDataArray(nifti.T)
points = vtk.vtkPoints()
points.SetData(test_array)

# saving the volumetric data to a VTK file
nifti_volumetric_path = os.path.join(origin_path, "new-data/id1090_timestep.nii")

nii_vol = nib.load(nifti_volumetric_path)

data_vol_3d = nii_vol.get_data()
pixdims = nii_vol.header['pixdim'][1:4]

data_vol_3d = np.flip(data_vol_3d, axis=2)
data_vol_3d = np.flip(data_vol_3d, axis=1)
#data_vol_3d = np.flip(data_vol_3d, axis=0)

#data_vol_3d_test = dsa.numpyTovtkDataArray(data_vol_3d)
#data_vol_1d = np.array(np.nonzero(data_vol_3d), dtype=np.float32)

# order F seems to give better results
dataImporter = vtk.vtkImageImport()
data_string = data_vol_3d.flatten(order='F').tostring()
dataImporter.CopyImportVoidPointer(data_string, len(data_string))

dataImporter.SetDataScalarTypeToShort()
dataImporter.SetNumberOfScalarComponents(1)

s = data_vol_3d.shape
#dataImporter.SetDataExtent(0, s[2] - 1, 0, s[1] - 1, 0, s[0] - 1)
#dataImporter.SetWholeExtent(0, s[2] - 1, 0, s[1] - 1, 0, s[0] - 1)

dataImporter.SetDataExtent(0, s[0] - 1, 0, s[1] - 1, 0, s[2] - 1)
dataImporter.SetWholeExtent(0, s[0] - 1, 0, s[1] - 1, 0, s[2] - 1)

test_image = vtk.vtkMetaImageWriter()
dataImporter.Update()
test_image.SetInputData(dataImporter.GetOutput())
test_image.SetFileName(os.path.join(origin_path, 'test_image_XX5.vtk'))
test_image.Write()

np.save('edges', edges)
np.save('/Users/raffi/Documents/owncloud/Projects/BrainVisualization/new-data/3dvis/generated_edge_file_with_alignment_raffi.npy', edges)
np.save('/Users/raffi/Documents/owncloud/Projects/BrainVisualization/new-data/3dvis/generated_edge_file_with_alignment_raffi2.npy', edges)

##
## trying to render something without the installation burden

from mpi_is_sw.brain_connectivity import visualization as VIS

# creating the mesh source
reader = vtk.vtkPolyDataReader()
reader.SetFileName(mesh_file_name)
reader.Update()

#create_index = vtk.vtkProgrammableFilter()
#create_index.AddInputConnection(reader.GetOutputPort())


def create_index_callback():
    input = pf.GetInput()
    output = pf.GetOutput()

    output.ShallowCopy(input)

    edges_files_path = os.path.join(origin_path, "new-data/3dvis/edges.npy")
    edges_files_path = "edges2.npy"

    VIS.draw_edges(edges_files_path,
                   input,
                   output,
                   means_path=None,
                   edge_selection=None,
                   edge_res=16,
                   k_means=None,
                   hierarchy_index=None,
                   show_edges=True,
                   mean_index=None)


def create_scene(flt):
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(flt.GetOutputPort())

    a = vtk.vtkActor()
    a.SetMapper(m)

    ren = vtk.vtkRenderer()
    ren.AddActor(a)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600, 600)

    return renWin


pf = vtk.vtkProgrammableFilter()
pf.SetInputConnection(reader.GetOutputPort())
pf.SetExecuteMethod(create_index_callback)

renWin = create_scene(pf)
renWin.Render()
time.sleep(3)

create_index.SetExecuteMethod(create_index_callback)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(create_index.GetPolyDataOutput())

#cylinderMapper = vtk.vtkPolyDataMapper()
#cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetPointSize(2)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0, 0, 1)  # Set background to blue

# Create the RendererWindow
renderer_window = vtk.vtkRenderWindow()
renderer_window.AddRenderer(renderer)

renderer.ResetCamera()
renderer_window.Render()

# Create the RendererWindowInteractor and display the vtk_file
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderer_window)
interactor.Initialize()
interactor.Start()

view = vtkGraphLayoutView()
view.AddRepresentationFromInputConnection(create_index.GetOutputPort())
view.SetVertexLabelArrayName("vertex_id")
view.SetVertexLabelVisibility(True)
view.SetEdgeLabelArrayName("edge_target")
view.SetEdgeLabelVisibility(True)

theme = vtkViewTheme.CreateMellowTheme()
view.ApplyViewTheme(theme)
theme.FastDelete()

view.GetRenderWindow().SetSize(600, 600)
view.ResetCamera()
view.Render()

view.GetInteractor().Start()

# output.SetPoints(PP.create_points(np.load('Users/Lennart/Desktop/brain-data-viz/data/map.npy')))
# vert_subset = vtk.vtkCellArray()
# vert_subset.InsertNextCell(415197, np.arange(415197))
# output.SetVerts(vert_subset)

if False:
    # nifti_cov = PP.get_cov(nifti)
    nifti_center = PP.get_center(nifti)

    mesh_center = PP.get_center(mesh_points)

    # aligns the matrix with the mesh in this specific case
    alignment_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    nifti -= nifti_center
    mesh_points -= mesh_center

    # find min and max extents of nifti
    n_min = [np.amin(nifti[0]), np.amin(nifti[1]), np.amin(nifti[2])]
    n_max = [np.amax(nifti[0]), np.amax(nifti[1]), np.amax(nifti[2])]

    # find min and max extents of mesh
    m_min = [np.amin(mesh_points[0]), np.amin(mesh_points[1]), np.amin(mesh_points[2])]
    m_max = [np.amax(mesh_points[0]), np.amax(mesh_points[1]), np.amax(mesh_points[2])]

    # compute factor by which to multiply nifti coordinates
    x_ex = (abs(m_min[0]) + m_max[0]) / (abs(n_min[0]) + n_max[0])
    y_ex = (abs(m_min[1]) + m_max[1]) / (abs(n_min[1]) + n_max[1])
    z_ex = (abs(m_min[2]) + m_max[2]) / (abs(n_min[2]) + n_max[2])

    x_ex = (m_max[0] - m_min[0]) / (n_max[0] - n_min[0])
    y_ex = (m_max[1] - m_min[1]) / (n_max[1] - n_min[1])
    z_ex = (m_max[2] - m_min[2]) / (n_max[2] - n_min[2])

    # flips over the y and z axis
    alignment_matrix = np.array([[x_ex, 0, 0], [0, -y_ex, 0], [0, 0, -z_ex]])

    #alignment_matrix = np.array([[x_ex, 0, 0], [0, y_ex, 0], [0, 0, z_ex]])

    #nifti = alignment_matrix.dot(nifti)
    # nifti += nifti_center

    # centering now on the mesh
    nifti += mesh_center

if 0:
    # get indices of connections
    indices = FF.read_csv(csv_path).astype(np.int32)

    edges = np.array([nifti[:,
                            [int(index[0]), int(index[1])]
                     ].T.flatten() for index in indices])

