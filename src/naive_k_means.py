# -*- coding: UTF-8 -*-
"""
mpi_is_sw.brain_connectivity.naive_k_means
******************************************

This module defines the k-means clustering used for creating clusters and summarized
information for the set of edges. The implementation is pure python and may be slow,
hence the name `naive`.
"""

from __future__ import print_function
import numpy as np
import time


def kmeans(k, edges, max_iterations=100, save_file=-1):
    """Returns k-means clustering of a set of edges.

    This algorithm clusters the set of ``edges`` in ``k`` clusters, using the metric
    given by the :py:func:`.distance` function.


    :param k: an integer value denoting the number of means/clusters
    :param edges: a matrix of n-dimensional datapoints
    :param max_iterations: an integer value that defines the maximum number of
      iterations should convergence not be reached
    :returns: a 2-uples where the first element is a matrix of shape ``(k, n)`` containing
      the centroids/means of the clustered dataset, and the second element of the tuple
      being the assignments of the ``edges`` given as argument to the indexed centroids.

    """
    start_time = time.time()

    # Initialize means from the original data points.
    means = init_means(k, edges)

    # Actual algorithm step.
    for i in xrange(max_iterations):
        print("%s steps" % i)
        # Compute distances between all means and all edges, as well the
        # distances between flipped means and edges.
        # distance_matrix = distances(edges, means)

        # Find closest points for each mean generate a boolean flip_map from
        # the computed distances.
        labels, flip_map = distances(edges, means)

        # Reassamble the edges array with flipped and original versions of all
        # edges, following the flip_map.
        edges = reorder_edges(edges, flip_map)

        # Update mean positions with the mean values of the assigned points.
        updated_centroids = update_centroids(edges, labels, means, flip_map)

        # Check for convergence between updated and previous means.
        if convergent(means, updated_centroids):
            print("Convergence reached in %s seconds." % (time.time() - start_time))
            if save_file is not -1:
                np.save(save_file, (means, labels))
            return means, labels

        if i % 100 == 0:
            if save_file is not -1:
                np.save(save_file, (means, labels))
        means = updated_centroids

    print("Convergence not reachead after %s seconds." % (time.time() - start_time))
    if save_file is not -1:
        np.save(save_file, (means, labels))
    return means, labels


def init_means(k, edges):
    """Returns k means, sampled at random data points out of a given set of
    edges, as a matrix of shape (k, n).

    :param k: an integer value denoting the number of means/clusters
    :param edges: a matrix of edges to sample the initial means from
    :returns: a matrix of k data points from the original edges matrix
    """
    means = edges[(np.random.rand(k,) * len(edges)).astype(np.int32)]
    return means


def distances(edges, means):
    """Computes the distance between two set of edges.

    The distance is summing the distance between the end points of two edges,
    irrespective of the orientation of those edges.

    The fixed set is called ``means``. For each edge returns the index of
    the closest mean edge, as well as a boolean value marking whether or not
    edge_i has been closer to the flipped or original version of the mean in
    question.
    """
    # First we generate flipped versions of our means.
    flipped_means = np.hstack((means[:, 3:], means[:, :3]))

    indices = np.empty([edges.shape[0], ], dtype=np.int32)
    flip_map = np.empty([edges.shape[0], ], dtype=bool)

    for i, current_edge in enumerate(edges):
        original_distances = ((current_edge - means) ** 2).sum(axis=1)
        flipped_distances = ((current_edge - flipped_means) ** 2).sum(axis=1)

        indices[i] = np.minimum(original_distances, flipped_distances).argmin()
        flip_map[i] = original_distances[indices[i]
                                         ] > flipped_distances[indices[i]]

    return indices, flip_map


def label_elements(distances):
    """
    """
    # Now in order to return the distances we care about, i.e. the smallest...
    minimal_distances = np.minimum(distances[0], distances[1])

    # We don't care about the actual distances, instead we want to return the
    # index of the closest mean edge.
    labels = np.argmin(minimal_distances, axis=0)

    # Returns an array of length n, where n is the number of edges. Each value
    # in the array is an index of one of the mean edges.
    return labels


def create_flip_map(distances):
    """Compares two arrays column-wise and returns an array of integer indices
    denoting whether the first or second array contains the smaller value.

    :param distances: a matrix of shape (2, k, n) where k is the number of means
      and n is the number of edges each mean is compared to
    :returns: an array of integer values of 0, 1

    """

    # Create a flat array of all distances, original and flipped.
    flat_distances = distances.flatten()

    # Reshape the flat_distances array. distance_matrix[0] == original_distances
    # distance_matrix[1] == flipped_distances
    distance_matrix = flat_distances.reshape(2, flat_distances.shape[0] / 2)

    # Apply np.argmin to find whether the original or flipped version yields
    # the smaller distance.
    flip_map = np.argmin(distance_matrix, axis=0)

    return flip_map


def reorder_edges(edges, flip_map):
    """Combines an array ``edges`` of datapoints and its flipped copy into a
    single array with length equal to the length of the original ``edges``.
    Datapoints are chosen based on a boolean array ``flip_map``.

    :param edges: a matrix of shape (n, 6)
    :param flip_map: an integer/boolean array of length n
    :returns: a matrix of shape (n, 6)
    """
    flipped_edges = np.hstack((edges[:, 3:], edges[:, :3]))

    reordered_edges = np.array([edges[i] if flip_map[i] == 0 else flipped_edges[
                               i] for i in xrange(edges.shape[0])])

    return reordered_edges


def update_centroids(edges, labels, means, flip_map):
    """Returns an updated matrix of means. For each mean, sets it to the
    mean of all points assigned to it.

    :param edges: a matrix of n-dimensional edges to compute clusters of
    :param labels: a matrix of shape edges.shape with integer values in the
      half-open interval [0, k)
    :param means: a matrix of shape (k, 1, n) where k is the number of datapoints
      and n is the length of a single datapoint
    :returns: a matrix of shape (k, 1, n), containing the updated means
    """
    updated_means = np.empty(means.shape)
    for k in xrange(means.shape[0]):
        updated_means[k] = edges[labels == k].mean(
            axis=0) if edges[labels == k].shape[0] > 0 else means[k]

    return updated_means


def fetch_edges(mean_index, edges, means):
    """Returns a matrix containing all edges assigned to a mean_index.
    """
    labels = distances(edges, means)[0]
    assigned_edges = edges[labels == mean_index]
    return assigned_edges


def convergent(old_means, new_means):
    """Returns ``True`` if the two sets of edges are close enough.

    This would indicate that the convergence of the k-means clustering
    has been reached.
    """
    return np.allclose(old_means, new_means)
