# -*- coding: utf-8 -*-

"""Connectivity visualisation filter

Paraview plugin wrapper.
"""

Name = 'BrainConnectivityVis'
Label = 'mpi-is-brain-connectivity'
Help = 'Brain fMRI connectivity visualization plugin'

NumberOfInputs = 1
InputDataType = 'vtkPolyData'
OutputDataType = 'vtkPolyData'
ExtraXml = ''

Properties = dict(
    edges_path='',
    means_path='',
    edge_selection=[120000, 120200],
    edge_res=16,
    k_means=7,
    show_edges=True,
    mean_index=-1,
    hierarchy_index=-1
)


def RequestData():
    from mpi_is_sw.brain_connectivity.visualization import draw_edges

    poly_data_input = self.GetPolyDataInput()
    poly_data_output = self.GetPolyDataOutput()

    draw_edges(poly_data_input=poly_data_input,
               poly_data_output=poly_data_output,
               edges_files_path=edges_path,
               edge_selection=edge_selection,
               edge_res=edge_res,
               k_means=k_means,
               means_path=means_path,
               hierarchy_index=hierarchy_index if hierarchy_index >= 0 else None,
               show_edges=show_edges,
               mean_index=mean_index if mean_index >= 0 else None)


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
               edges_files_path=edges_path,
               edge_selection=edge_selection,
               edge_res=edge_res,
               k_means=k_means,
               means_path=means_path,
               hierarchy_index=hierarchy_index if hierarchy_index >= 0 else None,
               show_edges=show_edges,
               mean_index=mean_index if mean_index >= 0 else None)
