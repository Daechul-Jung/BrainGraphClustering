# loader.py
import numpy as np
import torch

# -------------------------------
# Mesh utilities
# -------------------------------

# def mask_indices_from_mesh(avg_mesh):
#     if "cortex_indices" in avg_mesh and avg_mesh["cortex_indices"] is not None:
#         V = int(np.asarray(avg_mesh["coords"]).shape[0])
#         mask = np.zeros(V, dtype=bool)
#         mask[np.asarray(avg_mesh["cortex_indices"])] = True
#         return mask
#     mars = np.asarray(avg_mesh["MARS_label"])
#     return mars != -1

def mask_indices_from_mesh(avg_mesh):
    # V = number of vertices actually present in this mesh object
    if "vertices" in avg_mesh:
        V = int(np.asarray(avg_mesh["vertices"]).shape[0])
    elif "coords" in avg_mesh:
        V = int(np.asarray(avg_mesh["coords"]).shape[0])
    else:
        raise KeyError(f"avg_mesh missing vertices/coords. keys={list(avg_mesh.keys())}")

    # If cortex_indices exists, try to use it safely
    if "cortex_indices" in avg_mesh and avg_mesh["cortex_indices"] is not None:
        ci = np.asarray(avg_mesh["cortex_indices"]).astype(np.int64)

        # Case A: looks like 1-based indexing (MATLAB)
        if ci.min() == 1 and ci.max() == V:
            ci = ci - 1  # convert to 0-based

        # Case B: indices refer to a larger "original" mesh than the current coords/vertices
        # -> current mesh is already reduced, so keep everything
        if ci.max() >= V:
            return np.ones(V, dtype=bool)

        mask = np.zeros(V, dtype=bool)
        mask[ci] = True
        return mask

    # Fallback if MARS_label exists
    if "MARS_label" in avg_mesh:
        mars = np.asarray(avg_mesh["MARS_label"])
        return mars != -1

    # If nothing exists, assume all vertices in this mesh are cortex
    return np.ones(V, dtype=bool)

# def neighbors_list_from_dict(vertexNbors, mask=None):
#     """
#     Convert avg_mesh['vertexNbors'] (dict: {i: np.array(neis)}) to a list of np.ndarrays.
#     If mask is given (boolean of length V_all), drop non-cortex vertices and remap indices.
#     """
#     if mask is None:
#         V = len(vertexNbors)
#         return [np.asarray(vertexNbors[i], dtype=np.int64) for i in range(V)], np.arange(V, dtype=np.int64)

#     idx_map = -np.ones(len(mask), dtype=np.int64)
#     idx_map[np.where(mask)[0]] = np.arange(mask.sum(), dtype=np.int64)

#     out = []
#     for i in np.where(mask)[0]:
#         nbrs = vertexNbors[i]
#         nbrs = np.asarray([idx_map[j] for j in nbrs if mask[j]], dtype=np.int64)
#         out.append(nbrs)
#     return out, idx_map

def neighbors_list_from_any(vertexNbors, mask=None):
    """
    Accepts vertexNbors as:
      - dict: {i: np.ndarray}
      - list: [np.ndarray, ...]
      - np.ndarray: shape (V, K) with -1/NaN for missing
    Returns:
      - list_of_neighbors (after masking & reindex)
      - idx_map (old->new, -1 for removed)
    """
    # Convert to a "list of arrays" in full space
    if isinstance(vertexNbors, dict):
        V = len(vertexNbors)
        full_list = [np.asarray(vertexNbors[i], dtype=np.int64) for i in range(V)]
    elif isinstance(vertexNbors, list):
        V = len(vertexNbors)
        full_list = [np.asarray(v, dtype=np.int64) for v in vertexNbors]
    else:
        arr = np.asarray(vertexNbors)
        V = arr.shape[0]
        full_list = []
        for i in range(V):
            row = arr[i]
            # drop invalid entries
            row = row[~np.isnan(row)] if np.issubdtype(row.dtype, np.floating) else row
            row = row.astype(np.int64)
            row = row[row >= 0]
            full_list.append(row)

    if mask is None:
        return full_list, np.arange(V, dtype=np.int64)

    keep_idx = np.where(mask)[0]
    idx_map = -np.ones(V, dtype=np.int64)
    idx_map[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int64)

    out = []
    for i_old in keep_idx:
        nbrs = full_list[i_old]
        nbrs_new = idx_map[nbrs]
        nbrs_new = nbrs_new[nbrs_new >= 0]
        out.append(nbrs_new.astype(np.int64))

    return out, idx_map


def normalize_rows_3d(coords, eps=1e-8):
    """Row-wise L2 normalize (V,3) coordinates."""
    if not torch.is_tensor(coords):
        coords = torch.tensor(coords, dtype=torch.float32)
    nrm = torch.sqrt((coords * coords).sum(dim=1, keepdim=True) + eps)
    return coords / nrm
# -------------------------------
# Build adjacency from per-vertex edge_density (gradient)
# -------------------------------
def build_edge_density_adjacency_from_mesh(
    vertexNbors_list,
    edge_density_vertex,      # np.ndarray or torch.Tensor, shape (V_c,)
    symmetric=True,
    clamp=True
):
    """
    Edge weights from per-vertex edge_density (gradient-like):
       g_ij = 0.5*(g_i + g_j)
       w_ij = 1 - g_ij      (optionally clamped to [0,1])
    """
    gv = edge_density_vertex.detach().cpu().numpy() if torch.is_tensor(edge_density_vertex) \
         else np.asarray(edge_density_vertex)
    V = len(vertexNbors_list)
    rows, cols, data = [], [], []

    for i in range(V):
        neis = vertexNbors_list[i]
        if neis.size == 0:
            continue
        gij = 0.5 * (gv[i] + gv[neis])
        wij = 1.0 - gij
        if clamp:
            wij = np.clip(wij, 0.0, 1.0)
        keep = neis > i
        if keep.any():
            rows.append(np.full(keep.sum(), i, dtype=np.int64))
            cols.append(neis[keep].astype(np.int64))
            data.append(wij[keep].astype(np.float32))

    if not data:
        idx = torch.zeros((2, 0), dtype=torch.long)
        val = torch.zeros((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (V, V)).coalesce()

    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    if symmetric:
        r = np.concatenate([rows, cols])
        c = np.concatenate([cols, rows])
        d = np.concatenate([data, data])
    else:
        r, c, d = rows, cols, data

    idx = torch.tensor([r, c], dtype=torch.long)
    val = torch.tensor(d, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (V, V)).coalesce()

# -------------------------------
# Mesh adjacency (unit weights) for spatial Laplacian
# -------------------------------
def build_mesh_adjacency_from_neighbors(vertexNbors_list, symmetric=True):
    """Unit-weight mesh adjacency from neighbor list (no function)."""
    rows, cols = [], []
    for i, neis in enumerate(vertexNbors_list):
        if neis.size == 0: continue
        rows.append(np.full(len(neis), i, dtype=np.int64))
        cols.append(neis.astype(np.int64))
    if not rows:
        V = len(vertexNbors_list)
        return torch.sparse_coo_tensor(torch.zeros((2,0),dtype=torch.long),
                                       torch.zeros((0,),dtype=torch.float32),
                                       (V, V)).coalesce()
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    if symmetric:
        r = np.concatenate([rows, cols]); c = np.concatenate([cols, rows])
    else:
        r, c = rows, cols
    v = np.ones_like(r, dtype=np.float32)
    idx = torch.tensor([r, c], dtype=torch.long)
    val = torch.tensor(v, dtype=torch.float32)
    V = len(vertexNbors_list)
    return torch.sparse_coo_tensor(idx, val, (V, V)).coalesce()


def compute_degree_torch(adj):
    return torch.sparse.sum(adj, dim=1).to_dense()

def normalize_adj_torch(adj, eps=1e-8):
    deg = compute_degree_torch(adj)
    dinv = torch.pow(deg + eps, -0.5)
    i, j = adj._indices()
    v = adj._values()
    vals = v * dinv[i] * dinv[j]
    return torch.sparse_coo_tensor(adj._indices(), vals, adj.shape).coalesce()

def laplacian_from_adj_torch(adj):
    V = adj.shape[0]
    deg = compute_degree_torch(adj).to(torch.float32)
    idx = torch.arange(V, dtype=torch.long)
    D = torch.sparse_coo_tensor(torch.stack([idx, idx]), deg, (V, V))
    return (D - adj).coalesce()

# -------------------------------
# High-level builder for one hemisphere
# -------------------------------

def build_hemi_inputs_from_edge_density_and_timeseries(avg_mesh, edge_density_all, time_series_all):
    # avg_mesh keys: cortex_indices, vertexNbors, faces, coords
    coords = avg_mesh["coords"]  # (V_mesh, 3)
    V_mesh = coords.shape[0]

    # time_series_all is full per-hemi (32492, T)
    V_full = time_series_all.shape[0]

    # If shapes differ, use cortex_indices to map full->mesh space
    if V_full != V_mesh:
        if "cortex_indices" not in avg_mesh or avg_mesh["cortex_indices"] is None:
            raise RuntimeError(
                f"Need avg_mesh['cortex_indices'] to map full ({V_full}) -> mesh ({V_mesh}), "
                f"but it's missing. keys={list(avg_mesh.keys())}"
            )

        idx = np.asarray(avg_mesh["cortex_indices"], dtype=np.int64)

        # handle potential MATLAB 1-based indices
        if idx.min() == 1 and idx.max() == V_full:
            idx = idx - 1

        # sanity
        if idx.max() >= V_full:
            raise RuntimeError(f"cortex_indices out of range: max={idx.max()} V_full={V_full}")

        # Slice full space -> mesh space
        X_f = time_series_all[idx].float() if torch.is_tensor(time_series_all) \
              else torch.from_numpy(time_series_all[idx]).float()

        g_vertex = edge_density_all[idx] if not torch.is_tensor(edge_density_all) \
                   else edge_density_all[idx]

    else:
        # already aligned
        X_f = time_series_all.float() if torch.is_tensor(time_series_all) \
              else torch.from_numpy(time_series_all).float()
        g_vertex = edge_density_all

    # Spatial features are already in mesh space
    Y_s = normalize_rows_3d(coords)  # (V_mesh, 3)

    # Neighbors: in your new avg_mesh, vertexNbors should already be mesh-space
    vertexNbors = avg_mesh["vertexNbors"]
    if isinstance(vertexNbors, dict):
        vertexNbors_list = [np.asarray(vertexNbors[i], dtype=np.int64) for i in range(V_mesh)]
    else:
        vertexNbors_list = [np.asarray(n, dtype=np.int64) for n in vertexNbors]

    # Build graphs
    adj_f = build_edge_density_adjacency_from_mesh(vertexNbors_list, g_vertex, symmetric=True, clamp=True)
    norm_adj = normalize_adj_torch(adj_f)

    adj_mesh = build_mesh_adjacency_from_neighbors(vertexNbors_list, symmetric=True)
    L_s = laplacian_from_adj_torch(adj_mesh)

    return X_f, Y_s, adj_f, norm_adj, L_s

# def build_hemi_inputs_from_edge_density_and_timeseries(avg_mesh, edge_density_all, time_series_all):
#     """
#     avg_mesh: dict from CBIG_ReadNCAvgMesh
#     edge_density_all: (V_all,) np/torch
#     time_series_all: (V_all, T) np/torch (already pre-normalized in your pipeline)
#     Returns:
#         X_f: torch.FloatTensor (V_c, T)         # functional features for functional GCN
#         Y_s: torch.FloatTensor (V_c, 3)         # spatial features for spatial GCN (normalized coords)
#         adj_f: torch.sparse (V_c,V_c)           # from edge_density (1 - avg gradient)
#         norm_adj: torch.sparse (V_c,V_c)        # D^{-1/2} A D^{-1/2} (used by both GCNs)
#         L_s: torch.sparse (V_c,V_c)             # Laplacian of pure mesh (unit weights)
#     """
#     # Mask & neighbor remap
#     mask = mask_indices_from_mesh(avg_mesh)
#     vertexNbors_list, _ = neighbors_list_from_any(avg_mesh['vertexNbors'], mask=mask)

#     # Slice inputs to cortex
#     if torch.is_tensor(time_series_all):
#         X_f = time_series_all[mask].float()
#     else:
#         X_f = torch.from_numpy(time_series_all[mask]).float()

#     coords_c = avg_mesh['vertices'][mask]     # torch.Tensor (V_c, 3)
#     Y_s = normalize_rows_3d(coords_c)         # normalized spatial features

#     g_vertex_c = edge_density_all[mask] if not torch.is_tensor(edge_density_all) else edge_density_all[mask]

#     # Graph from edge_density (for modularity & message passing)
#     adj_f = build_edge_density_adjacency_from_mesh(vertexNbors_list, g_vertex_c, symmetric=True, clamp=True)
#     norm_adj = normalize_adj_torch(adj_f)

#     # Spatial Laplacian from unit mesh
#     adj_mesh = build_mesh_adjacency_from_neighbors(vertexNbors_list, symmetric=True)
#     L_s = laplacian_from_adj_torch(adj_mesh)

#     return X_f, Y_s, adj_f, norm_adj, L_s