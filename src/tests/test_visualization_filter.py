"""Some basic tests of the visualization file"""

import unittest
import os
import numpy as np
import vtk

from .. import visualization


class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.path_files = os.path.join(os.path.dirname(__file__), "test_data")
        self.nii_file = os.path.join(self.path_files, "map.nii")
        self.csv_file = os.path.join(self.path_files, "mat.csv")
        self.mesh_file = os.path.join(self.path_files, "fmri_lvls_csf_gm_bf_FIX_reinit.vtk")

    def get_input_mesh(self, input_file=None):
        if input_file is None:
            input_file = self.mesh_file

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(input_file)
        reader.Update()

        vtk_poly_data_input = reader.GetOutput()

        return vtk_poly_data_input

    def test_visalization_requires_edge_file(self):
        """Checks that visualization does not work without prior creation of edge files"""

        # a mesh is needed as input
        vtk_poly_data_input = self.get_input_mesh()

        # output can be empty
        vtk_poly_data_output = vtk.vtkPolyData()

        visualization.draw_edges(edges_files_path=None,
                                 poly_data_input=vtk_poly_data_input,
                                 poly_data_output=vtk_poly_data_output,
                                 centroids_file_path=None,
                                 edge_selection=None,
                                 resolution=None)

        self.assertEqual(vtk_poly_data_output.GetNumberOfCells(), 0)

    def test_visualization_basic_run(self):
        """Checks a basic run of the filter"""
        import tempfile
        from ..file_ops import assemble_edges

        vtk_poly_data_input = self.get_input_mesh()
        vtk_poly_data_output = vtk.vtkPolyData()

        # creates the edge file

        edges = assemble_edges(nifti_path=self.nii_file,
                               csv_path=self.csv_file)

        with tempfile.NamedTemporaryFile(suffix='.npy') as edge_file:
            np.save(edge_file.name, edges)

            visualization.draw_edges(edges_files_path=edge_file.name,
                                     poly_data_input=vtk_poly_data_input,
                                     poly_data_output=vtk_poly_data_output,
                                     centroids_file_path=None,
                                     edge_selection=[0, 100],
                                     resolution=None)

        self.assertGreater(vtk_poly_data_output.GetNumberOfCells(), 0)

    def test_visualization_cluster_file_disables_edges_selection(self):
        """Checks that providing the cluster file disables the edge selection"""
        import tempfile
        from ..file_ops import assemble_edges
        from ..utils import generate_processing

        vtk_poly_data_input = self.get_input_mesh()
        vtk_poly_data_output = vtk.vtkPolyData()

        # cluster_file = os.path.join(self.path_files, "clusters.npy")
        # creates the edge file

        edges = assemble_edges(nifti_path=self.nii_file,
                               csv_path=self.csv_file)

        with tempfile.NamedTemporaryFile(suffix='.npy') as edge_file:
            np.save(edge_file.name, edges)

            with tempfile.NamedTemporaryFile(suffix='.npy') as cluster_file:

                generate_processing.generate_cluster_file(
                    mesh_file=self.mesh_file,
                    edge_file=edge_file.name,
                    number_of_clusters=6,
                    output_file=cluster_file.name,
                    max_iterations=1)

                visualization.draw_edges(edges_files_path=edge_file.name,
                                         poly_data_input=vtk_poly_data_input,
                                         poly_data_output=vtk_poly_data_output,
                                         edge_selection=[0, 100],  # edge selection should be disabled
                                         centroids_file_path=cluster_file.name,
                                         cluster_index=2,
                                         resolution=None)

        self.assertGreater(vtk_poly_data_output.GetNumberOfCells(), 0)

    def test_visualization_cluster_file_loads_correctly_on_cluster_selection(self):
        """Checks loading the cluster file and selecting a cluster work together"""
        import tempfile
        from ..file_ops import assemble_edges
        from ..utils import generate_processing

        vtk_poly_data_input = self.get_input_mesh()
        vtk_poly_data_output = vtk.vtkPolyData()

        # cluster_file = os.path.join(self.path_files, "clusters.npy")
        # creates the edge file

        edges = assemble_edges(nifti_path=self.nii_file,
                               csv_path=self.csv_file)

        with tempfile.NamedTemporaryFile(suffix='.npy') as edge_file:
            np.save(edge_file.name, edges)

            with tempfile.NamedTemporaryFile(suffix='.npy') as cluster_file:

                generate_processing.generate_cluster_file(
                    mesh_file=self.mesh_file,
                    edge_file=edge_file.name,
                    number_of_clusters=6,
                    output_file=cluster_file.name,
                    max_iterations=1)

                visualization.draw_edges(edges_files_path=edge_file.name,
                                         poly_data_input=vtk_poly_data_input,
                                         poly_data_output=vtk_poly_data_output,
                                         edge_selection=[0, 100],  # edge selection should be disabled
                                         centroids_file_path=cluster_file.name,
                                         cluster_index=1,
                                         resolution=None)

        self.assertGreater(vtk_poly_data_output.GetNumberOfCells(), 0)
