
# to call with
# python -m src.tests.manual_test

import numpy as np
import vtk
from .. import plot as PP
from .. import file_ops as FF
from .. import visualization as VIS

import os
import time

#sys.path.insert(0, '/Users/raffi/usr/local/lib/python2.7/site-packages/')

origin_path = "/Users/raffi/Documents/owncloud/Projects/BrainVisualization/"
origin_path = "/home/raffi/Documents/owncloud/Projects/BrainVisualization/"

nifti_path = os.path.join(origin_path, "new-data/3dvis/map.nii")
csv_path = os.path.join(origin_path, "new-data/3dvis/mat.csv")

# the same brain mesh as opened in paraview
mesh_file_name = os.path.join(origin_path, "new-data/3dvis/fmri_lvls_csf_gm_bf_FIX_reinit.vtk")

if 0:
    # generate the stuff
    nifti = FF.pointset_from_indicator_nifti_file(nifti_path)
    vtkpoints = PP.create_points(nifti)
    verts = vtk.vtkCellArray()
    verts = FF.get_vtkcellarray_from_vtkpoints(vtkpoints)

    # writes the edge file
    edges = FF.assemble_edges(nifti_path, csv_path)
    np.save('edges2', edges)


def get_mesh_source(mesh_file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(mesh_file_name)
    reader.Update()

    return reader


class PluginAlgorithm(object):

    def __init__(self, programmable_filter):
        import weakref
        self.programmable_filter = weakref.ref(programmable_filter)

    def __call__(self):
        filter_input = self.programmable_filter().GetInput()
        output = self.programmable_filter().GetOutput()

        edges_files_path = "edges_raffi.npy"
        centroids_file_path = "clusters2.npy"

        VIS.draw_edges(edges_files_path,
                       filter_input,
                       output,
                       centroids_file_path=centroids_file_path,
                       edge_selection=None,
                       resolution=16,
                       nb_clusters=None,
                       hierarchy_index=None,
                       show_edges=True,
                       cluster_index=1)


def create_scene(flt, mesh_reader):
    ren = vtk.vtkRenderer()

    # mesh
    mesh_mapper = vtk.vtkPolyDataMapper()
    mesh_mapper.SetInputConnection(mesh_reader.GetOutputPort())
    mesh_actor = vtk.vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    ren.AddActor(mesh_actor)

    # plugin
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(flt.GetOutputPort())

    a = vtk.vtkActor()
    a.SetMapper(m)
    ren.AddActor(a)

    # axis
    # The axes are positioned with a user transform
    transform = vtk.vtkTransform()
    transform.Translate(1.0, 0.0, 0.0)

    axes = vtk.vtkAxesActor()
    axes.SetUserTransform(transform)
    ren.AddActor(axes)

    # the actual text/color of the axis label can be changed:
    # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(colors.GetColor3d("Red"));
    # axes->SetXAxisLabelText("test");

    # bounding box
    outline_filter = vtk.vtkOutlineFilter()
    outline_filter.SetInputData(flt.GetOutput())
    outline_mapper = vtk.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline_filter.GetOutputPort())
    outline_actor = vtk.vtkActor()
    outline_actor.SetMapper(outline_mapper)
    ren.AddActor(outline_actor)

    # window and interaction
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600, 600)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    iren.Initialize()

    return renWin, iren


def create_pipeline():
    reader = get_mesh_source(mesh_file_name)

    pf = vtk.vtkProgrammableFilter()
    pf.SetInputConnection(reader.GetOutputPort())
    algo = PluginAlgorithm(pf)
    pf.SetExecuteMethod(algo)

    renWin, iren = create_scene(pf, reader)

    # some additional customization
    renWin.SetWindowName("Demo plugin")

    renWin.Render()
    time.sleep(3)

    iren.Start()

    #raw_input()


create_pipeline()

