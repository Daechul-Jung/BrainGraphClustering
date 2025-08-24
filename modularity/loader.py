import numpy as np
import torch
from sklearn.decomposition import PCA

def build_gradient_adjacency_torch(vertex_nbors, grad, n_vertices, n_neighbors=6, symmetric=True):
    row, col, data = [], [], []
    for i in range(n_vertices):
        neighbors = vertex_nbors[i]
        for k in range(min(n_neighbors, len(neighbors))):
            j = neighbors[k]
            if j >= 0 and j < n_vertices:
                w = 1.0 - ((grad[i].item() + grad[j].item()) / 2.0)
                row.append(i)
                col.append(j)
                data.append(w)
                if symmetric and j != i:
                    row.append(j)
                    col.append(i)
                    data.append(w)
    idx = torch.tensor([row, col], dtype=torch.long)
    val = torch.tensor(data, dtype=torch.float32)
    adj = torch.sparse.FloatTensor(idx, val, torch.Size([n_vertices, n_vertices]))
    return adj

def compute_degree_torch(adj):
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # (N,)
    return deg

def normalize_adj_torch(adj):
    deg = compute_degree_torch(adj)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    idx = adj._indices()
    val = adj._values()
    d_i = deg_inv_sqrt[idx[0]]
    d_j = deg_inv_sqrt[idx[1]]
    norm_vals = val * d_i * d_j
    norm_adj = torch.sparse.FloatTensor(idx, norm_vals, adj.shape)
    return norm_adj

def positional_encoding(coords, num_frequencies=8):
    N, D = coords.shape
    encodings = [coords]
    for k in range(num_frequencies):
        for fn in [np.sin, np.cos]:
            encodings.append(fn(coords * (2.0 ** k)))
    encodings = np.concatenate(encodings, axis=1)
    return encodings

def build_feature_matrix(time_series, coords, pca_dim=64, pos_enc_dim=8):
    if pca_dim is not None:
        pca = PCA(n_components=pca_dim)
        ts_features = pca.fit_transform(time_series)
    else:
        ts_features = time_series
    pos_features = positional_encoding(coords, num_frequencies=pos_enc_dim)
    features = np.concatenate([ts_features, pos_features], axis=1)
    return features

def load_vertex_nbors_from_txt(path):
    # Each line: index n1 n2 n3 n4 n5 n6
    with open(path) as f:
        return [list(map(int, line.strip().split()[1:])) for line in f]

def load_grad_from_npy(path):
    return torch.from_numpy(np.load(path)).float()

def load_time_series_from_npy(path):
    return np.load(path)

def load_coords_from_npy(path):
    return np.load(path)

def load_hcp_subject(vertex_nbors, grad, time_series, coords,
                     n_neighbors=6, pca_dim=64, pos_enc_dim=8):
    n_vertices = len(vertex_nbors)
    adj = build_gradient_adjacency_torch(vertex_nbors, grad, n_vertices, n_neighbors)
    norm_adj = normalize_adj_torch(adj)
    deg = compute_degree_torch(adj)
    features = build_feature_matrix(time_series, coords, pca_dim=pca_dim, pos_enc_dim=pos_enc_dim)
    features_torch = torch.FloatTensor(features)
    return features_torch, adj, norm_adj, deg
