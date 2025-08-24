import numpy as np

def watershed_algorithm(
    edgemetric,
    minimametric,
    stepnum,
    fracmaxh,
    neighbors,
    minh=None,
    maxh=None,
    random_seed=None
):
    """
    Watershed labeling based on smoothed gradient map and local minima.
    Args:
        edgemetric: np.ndarray (n_vertices,), the smoothed gradient map.
        minimametric: np.ndarray (n_vertices,), 1 for seed (minima), 0 otherwise.
        stepnum: int, number of steps.
        fracmaxh: float, fraction of max height to stop at (usually 1).
        neighbors: list of arrays, neighbors[v] is a 1D array of neighbors of v.
        minh: float or None, min threshold (defaults to min(edgemetric)).
        maxh: float or None, max threshold (defaults to max(edgemetric)).
        random_seed: int or None, for reproducibility.
    Returns:
        labels: np.ndarray (n_vertices,), each vertex labeled by watershed zone.
        watershed_zones: np.ndarray (n_vertices,), 1 if edge, 0 otherwise.
    """
    n_vertices = edgemetric.shape[0]
    if minh is None:
        minh = float(np.nanmin(edgemetric))
    if maxh is None:
        maxh = float(np.nanmax(edgemetric))
    stoph = maxh * fracmaxh
    step = (maxh - minh) / stepnum
    hiter = np.arange(minh, stoph + step, step)

    labels = np.zeros(n_vertices, dtype=np.int32)
    labelpos = np.where(minimametric == 1)[0]
    if random_seed is not None:
        np.random.seed(random_seed)
    labelnums = np.random.permutation(len(labelpos)) + 1
    labels[labelpos] = labelnums

    watershed_zones = np.zeros(n_vertices, dtype=np.int32)

    for hi in hiter:
        maskmetrics = (edgemetric < hi) & (labels == 0) & (watershed_zones == 0)
        maskpos = np.where(maskmetrics)[0]
        maskpos = np.random.permutation(maskpos)
        for m in maskpos:
            nodeneigh = neighbors[m]
            nodeneighlab = labels[nodeneigh]
            nonzero_lab = nodeneighlab[nodeneighlab != 0]
            if len(nonzero_lab) == 0:
                continue
            minlab = np.min(nonzero_lab)
            maxlab = np.max(nonzero_lab)
            if minlab != maxlab:
                watershed_zones[m] = 1
                labels[m] = 0
            else:
                labels[m] = minlab
    return labels, watershed_zones


def get_K_hop_neighbors(vertex_nbors, K=3):
    """
    For each vertex, find all vertices reachable within K hops.
    """
    N = len(vertex_nbors)
    K_neighbors = []
    for v in range(N):
        nbrs = set([v])
        current = set([v])
        for _ in range(K):
            next_nbrs = set()
            for u in current:
                next_nbrs.update(vertex_nbors[u])
            nbrs |= next_nbrs
            current = next_nbrs
        nbrs.discard(v)  
        K_neighbors.append(np.array(list(nbrs), dtype=int))
    maxlen = max(len(n) for n in K_neighbors)
    K_neighbors_arr = -np.ones((N, maxlen), dtype=int)
    for i, nbrs in enumerate(K_neighbors):
        K_neighbors_arr[i, :len(nbrs)] = nbrs
    return K_neighbors_arr
