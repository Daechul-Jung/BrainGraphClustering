import numpy as np

def gradient_vertices_to_matrix(grad, vertex_nbors, n_neighbors=6):
    """
    For each vertex and each of its neighbors, 
    fill the matrix with the average of the gradient values.
    Output: grad_matrix (n_vertices, n_neighbors), with -1 or np.nan for missing neighbors.
    """
    n_vertices = len(vertex_nbors)
    grad_matrix = np.full((n_vertices, n_neighbors), np.nan, dtype=np.float32)
    for i in range(n_vertices):
        neighbors = vertex_nbors[i]
        for j, n_idx in enumerate(neighbors[:n_neighbors]):
            if n_idx >= 0:
                grad_matrix[i, j] = (grad[i] + grad[n_idx]) / 2.0
                
    return grad_matrix
