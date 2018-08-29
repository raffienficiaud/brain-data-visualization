# -*- coding: utf-8 -*-

"""Connectivity visualization filter

Paraview plugin wrapper.
"""

Name = 'mpi-is-brain-connectivity'
Label = 'mpi-is-brain-connectivity'
Help = 'Brain fMRI connectivity visualization plugin'
LongHelp = 'Paraview filter for visualizing brain connectivity data.'

NumberOfInputs = 1
InputDataType = 'vtkPolyData'
OutputDataType = 'vtkPolyData'
ExtraXml = ''

Properties = dict(
    edges_file='',
    clusters_file='',
    edge_selection=[120000, 120200],
    resolution=16,
    nb_clusters=7,
    show_edges=True,
    cluster_index=-1,
    hierarchy_index=-1
)


def RequestData():
    from mpi_is_sw.brain_connectivity.visualization import draw_edges

    poly_data_input = self.GetPolyDataInput()
    poly_data_output = self.GetPolyDataOutput()

    draw_edges(poly_data_input=poly_data_input,
               poly_data_output=poly_data_output,
               edges_files_path=edges_file,
               edge_selection=edge_selection,
               resolution=resolution,
               nb_clusters=nb_clusters,
               centroids_file_path=clusters_file,
               hierarchy_index=hierarchy_index if hierarchy_index >= 0 else None,
               show_edges=show_edges,
               cluster_index=cluster_index if cluster_index >= 0 else None,
               center_edges=True)


def RequestDataVirtualEnv():
    activate_this = '$VIRTUAL_ENV_SUBSTITUTION'
    execfile(activate_this, dict(__file__=activate_this))

    # from paraview.simple import *
    # import numpy as np

    from mpi_is_sw.brain_connectivity.visualization import draw_edges

    poly_data_input = self.GetPolyDataInput()
    poly_data_output = self.GetPolyDataOutput()

    draw_edges(poly_data_input=poly_data_input,
               poly_data_output=poly_data_output,
               edges_files_path=edges_file,
               edge_selection=edge_selection,
               resolution=resolution,
               nb_clusters=nb_clusters,
               centroids_file_path=clusters_file,
               hierarchy_index=hierarchy_index if hierarchy_index >= 0 else None,
               show_edges=show_edges,
               cluster_index=cluster_index if cluster_index >= 0 else None,
               center_edges=True)
