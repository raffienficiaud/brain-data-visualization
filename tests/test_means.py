import unittest, vtk
from .context import naive_k_means as nm
import numpy as np


class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.edges = np.array([[0, 0, 0, 1, 1, 1],
                               [2, 2, 2, 0, 0, 0],
                               [3, 3, 3, 4, 4, 4]])
        self.means = np.array([[1, 1, 1, 0, 0, 0],
                               [-1, -1, -1, 0, 0, 0]])

# testing helper functions

    def test_distance_function(self):
        means = np.array([[0, 0, 0, 0, 0, 0]])
        expected = np.array([0, 0, 0])
        labels = nm.distances(self.edges, means)[0]
        self.assertTrue(np.allclose(labels, expected))

    def test_create_flip_map(self):
        distances = np.array([[0, 1, 2], [2, 1, 0]])
        expected = np.array([0, 0, 1])
        flip_map = nm.create_flip_map(distances)
        self.assertTrue(np.allclose(flip_map, expected))

    def test_reorder_edges(self):
        flipped_edges = nm.reorder_edges(self.edges, [1, 0, 0])
        unflipped_edges = nm.reorder_edges(flipped_edges, [1, 0, 0])
        self.assertTrue(np.allclose(unflipped_edges, self.edges))
        pass

# testing full k-means

    def test_return_singular_mean(self):
        expected = np.array([[1., 1., 1., 2.33333333, 2.33333333,
                           2.33333333]])
        mean = nm.kmeans(1, self.edges)[0]
        self.assertTrue(np.allclose(mean, expected) or
                        np.allclose(mean, np.hstack((expected[:, 3:], expected[:, :3]))))

    def test_more_means_than_edges(self):
        means = nm.kmeans(3, self.means)[0]
        self.assertTrue(len(means) == 3)

    def test_kmeans(self):
        expected = np.array([[1.5, 1.5, 1.5, 0., 0., 0.],
                             [4., 4., 4., 3., 3., 3.]])
        flipped_expected = np.hstack((expected[:, 3:], expected[:, :3]))
        means = nm.kmeans(2, self.edges)[0]
        self.assertTrue(np.allclose(means, expected) or
                        np.allclose(means, np.flipud(expected)) or
                        np.allclose(means, flipped_expected) or
                        np.allclose(means, np.flipud(flipped_expected)) or
                        np.allclose(means, nm.reorder_edges(expected, [1, 0])) or
                        np.allclose(means, nm.reorder_edges(expected, [0, 1])) or
                        np.allclose(means, np.flipud(nm.reorder_edges(expected, [1, 0]))) or
                        np.allclose(means, np.flipud(nm.reorder_edges(expected, [0, 1]))))
