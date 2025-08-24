import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def CBIG_gwMRF_generate_components(lh_avg_mesh, rh_avg_mesh, lh_labels, rh_labels):
    """
    Computes connected components for a labeled vector.

    Args:
        lh_avg_mesh (dict): Left hemisphere mesh with 'vertices' and 'vertexNbors'.
        rh_avg_mesh (dict): Right hemisphere mesh with 'vertices' and 'vertexNbors'.
        lh_labels (torch.Tensor): Label vector for the left hemisphere.
        rh_labels (torch.Tensor): Label vector for the right hemisphere.

    Returns:
        tuple: (lh_ci, rh_ci, lh_sizes, rh_sizes)
            lh_ci: Connected components for the left hemisphere.
            rh_ci: Connected components for the right hemisphere.
            lh_sizes: Size of the corresponding components for the left hemisphere.
            rh_sizes: Size of the corresponding components for the right hemisphere.
    """
    def generate_components(avg_mesh, labels):
        n = avg_mesh['vertices'].shape[0]
        neighbors = min(len(nbors) for nbors in avg_mesh['vertexNbors'])  
        
        b = np.zeros(n * neighbors, dtype=int)
        c = np.zeros(n * neighbors, dtype=int)
        d = np.zeros(n * neighbors, dtype=int)

        if isinstance(labels, torch.Tensor):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)
        
        for i in range(12):
            current_nbors = avg_mesh['vertexNbors'][i] 
            num = neighbors - 1 
            start = i * neighbors
            end = start + num
            
            b[start:end] = i
            c[start:end] = np.array(current_nbors)[:num]
            d[start:end] = (labels_np[np.array(current_nbors[:num])] == labels_np[i]).astype(int)
        
        for i in range(12, n):
            current_nbors = avg_mesh['vertexNbors'][i]
            num = neighbors  
            start = i * neighbors
            end = start + num
            
            b[start:end] = i
            c[start:end] = current_nbors[:num]
            d[start:end] = (labels_np[np.array(current_nbors[:num])] == labels_np[i]).astype(int)
        
        indices_to_remove = [i * neighbors for i in range(12)]
        b = np.delete(b, indices_to_remove)
        c = np.delete(c, indices_to_remove)
        d = np.delete(d, indices_to_remove)
        
        g = coo_matrix((d, (b, c)), shape=(n, n))
        num_components, comp_labels = connected_components(csgraph=g, directed=False, return_labels=True)
        comp_sizes = np.bincount(comp_labels)
        
        return comp_labels, comp_sizes
    lh_ci, lh_sizes = generate_components(lh_avg_mesh, lh_labels)
    rh_ci, rh_sizes = generate_components(rh_avg_mesh, rh_labels)
    print(f'lh_ci: {lh_ci}')
    return lh_ci, rh_ci, lh_sizes, rh_sizes
