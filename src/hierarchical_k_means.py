# -*- coding: UTF-8 -*-
import naive_k_means as nkm
import numpy as np


def hi_means(steps, edges):
    """This applies kmeans in a hierarchical fashion.

    :param edges:
    :param steps:
    :returns: a tuple of two arrays, ´´kmeans_history´´ containing a number of
      arrays of varying lengths and ´´labels_history´´, an array of length equal
      to edges.shape[0]
    """

    sub_edges = edges

    kmeans_history = []
    labels_history = []

    for _ in xrange(steps):
        kmeans = nkm.kmeans(sub_edges.shape[0] / 2, sub_edges)
        sub_edges = kmeans[0]

        kmeans_history += [kmeans[0]]
        labels_history += [kmeans[1]]

    kmeans_history = np.array(kmeans_history)
    labels_history = np.array(labels_history)

    return kmeans_history, labels_history


def fetch_mean_labels(k, kmeans_history, level):
    labels = kmeans_history[1]
    indices = np.array([k])

    for i in range(-1, level, -1):
        try:
            indices = np.argwhere(np.array([labels[i] == j for j in indices]))[:, 1]
        except IndexError:
            print 'whoops'
            return np.unique(np.array(indices))

    return np.unique(indices)


def fetch_labels(k, kmeans_history, offset=-1):
    """This fetches the original edges of a hierarchical mean.
    """
    labels = kmeans_history[1]
    passes = labels.shape[0]
    indices = np.array([k])

    for i in range(offset, -(passes + 1), -1):
        try:
            indices = np.array([np.nonzero(labels[i] == j)[0]
                                for j in indices])
            indices = np.hstack(indices)
        except IndexError:
            return np.array(indices)

    return indices


def assemble_labels(kmeans_history, level=-1):
    """Based on a set of hierarchical means, this generates labels for the original
    edges - moving down the hierarchy.
    """
    labels = np.array(range(kmeans_history[1][0].shape[0]))
    for i, m in enumerate(kmeans_history[0][level]):
        indices = fetch_labels(i, kmeans_history, level)
        if indices.shape[0] > 0:
            labels[indices] = i

    return labels
