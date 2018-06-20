"""Some basic tests of the command line utilities"""

import unittest
import os
import numpy as np

from ..utils import generate_processing


class TestCommandLineTools(unittest.TestCase):

    def setUp(self):
        self.path_files = os.path.join(os.path.dirname(__file__), "test_data")
        self.nii_file = os.path.join(self.path_files, "map.nii")
        self.csv_file = os.path.join(self.path_files, "mat.csv")
        self.mesh_file = os.path.join(self.path_files, "fmri_lvls_csf_gm_bf_FIX_reinit.vtk")

    def test_generate_edge_file(self):
        """Basic check of the edge file generation"""

        import tempfile
        # the suffix is needed otherwise numpy will create another file with appropriate extension
        with tempfile.NamedTemporaryFile(suffix='.npy') as test_file:
            self.assertTrue(os.path.exists(test_file.name))
            self.assertEqual(os.path.getsize(test_file.name), 0)
            generate_processing.generate_edge_file(self.csv_file, self.nii_file, test_file.name)

            self.assertTrue(os.path.exists(test_file.name))
            self.assertGreater(os.path.getsize(test_file.name), 0)

            # check numpy can load the file
            edges = np.load(test_file.name)
            self.assertGreater(edges.shape[0], 0)
            self.assertEqual(edges.shape[1], 6)

    def test_generate_cluster_file(self):
        """Basic check of the kmeans/cluster file generation"""

        import tempfile
        # the suffix is needed otherwise numpy will create another file with appropriate extension
        with tempfile.NamedTemporaryFile(suffix='.npy') as test_file:
            generate_processing.generate_edge_file(self.csv_file, self.nii_file, test_file.name)

            # check numpy can load the file
            edges = np.load(test_file.name)
            self.assertGreater(edges.shape[0], 0)
            self.assertEqual(edges.shape[1], 6)

            # now we have the edge file, we do the same for the kmeans clustering
            with tempfile.NamedTemporaryFile(suffix='.npy') as test_file_centroids:
                self.assertTrue(os.path.exists(test_file_centroids.name))
                self.assertEqual(os.path.getsize(test_file_centroids.name), 0)

                generate_processing.generate_cluster_file(
                    mesh_file=self.mesh_file,
                    edge_file=test_file.name,
                    number_of_clusters=10,
                    output_file=test_file_centroids.name,
                    max_iterations=1)

                centroids, labels = np.load(test_file_centroids.name)
                self.assertEqual(centroids.shape[0], 10)
                self.assertEqual(centroids.shape[1], 6)
                self.assertEqual(labels.shape[0], edges.shape[0])
                self.assertEqual(len(labels.shape), 1)
