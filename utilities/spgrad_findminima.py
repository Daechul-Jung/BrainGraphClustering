import numpy as np

def find_local_minima(data, K_neighbors):
    """
    Find local minima for each vertex based on the K-nearest neighbors.

    Args:
        data: 1D numpy array of shape (n_vertices,) - the smoothed gradient map.
        K_neighbors: 2D numpy array of shape (n_vertices, K) - neighbor indices for each vertex.
                     Use -1 or np.nan for padding neighbors that don't exist.

    Returns:
        minima_metric: 1D numpy array of shape (n_vertices,) - 1 for local minima, 0 otherwise.
    """
    n_vertices = data.shape[0]
    is_min = np.ones(n_vertices, dtype=bool)
    for k in range(K_neighbors.shape[1]):
        nbr = K_neighbors[:, k]
        # Set comparison value to Inf if neighbor index is -1 (no neighbor)
        nbr_data = np.full_like(data, np.inf)
        mask = (nbr >= 0) & (nbr < n_vertices)
        nbr_data[mask] = data[nbr[mask]]
        is_min = is_min & (data < nbr_data)
    return is_min.astype(np.int32)
