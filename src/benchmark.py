# -*- coding: UTF-8 -*-

"""Support functions for extracting metrics on the computation.

"""

import numpy as np
from . import naive_k_means as nm
from . import hierarchical_k_means as hm


def get_single_edge_distance(mean, edges):
    """Calculates distances between a single mean edge and a set of edges.
    """
    flipped_mean = np.hstack([mean[3:], mean[:3]])
    org_dists = np.sqrt(((edges - mean) ** 2).sum(axis=1))
    flp_dists = np.sqrt(((edges - flipped_mean) ** 2).sum(axis=1))
    return np.minimum(org_dists, flp_dists)


def accumulate_distances(means, edges):
    """Calculates the total distance between all edges and all mean edges.
    """
    total_dist = 0
    for i, m in enumerate(means[0]):
        temp_dist = get_single_edge_distance(m, edges[means[1] == i])
        total_dist += temp_dist.sum()
    return total_dist


def match_edges(edges_0, edges_1):
    """Matches two sets of edges and returns the a fitted copy of `edges_1`.
    """
    flipped_edges_1 = np.hstack((edges_1[:, 3:], edges_1[:, :3]))
    matched_edges = []

    for _, cur_edge in enumerate(edges_0):
        org_dists = ((cur_edge - edges_1) ** 2).sum(axis=1)
        flp_dists = ((cur_edge - flipped_edges_1) ** 2).sum(axis=1)

        index = np.minimum(org_dists, flp_dists).argmin()
        matched_edges += [edges_1[index]] if org_dists[index] < flp_dists[index] else [flipped_edges_1[index]]
        edges_1 = np.delete(edges_1, index, axis=0)
        flipped_edges_1 = np.delete(flipped_edges_1, index, axis=0)

    return matched_edges


def benchmark_accumulated_distances(k, n, edges, trials=10):
    """Runs kmeans and hierarchical kmeans ´trials´ times. Prints out distances
    for each trial for kmeans, hierarchical kmeans and recalculated hierarchical

    kmeans respectively - returns the mean of total distances across all trials.
    :param k: number of clusters for the single kmeans approach
    :param n: number of passes for the hierarchical kmeans approach
    """
    si_dists = []
    hi_dists = []
    hi_re_dists = []
    for i in xrange(trials):
        si_means = nm.kmeans(k, edges)
        hi_means = hm.hi_means(edges, n)

        si_dist = accumulate_distances(si_means, edges)
        hi_dist = accumulate_distances((hi_means[0][-1], hm.assemble_labels(hi_means)), edges)
        hi_re_dist = accumulate_distances((hi_means[0][-1], nm.distances(edges, hi_means[0][-1])[0]), edges)

        si_dists += [si_dist]
        hi_dists += [hi_dist]
        hi_re_dists += [hi_re_dist]

    for i in xrange(trials):
        print si_dists[i], hi_dists[i], hi_re_dists[i]

    return np.mean(np.array(si_dists)), np.mean(np.array(hi_dists)), np.mean(np.array(hi_re_dists))
