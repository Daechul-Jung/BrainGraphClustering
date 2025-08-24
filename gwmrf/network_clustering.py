# models/network_clustering.py
import numpy as np
import nibabel as nib
import torch
from typing import List, Tuple, Dict, Optional


def _gifti_to_data(path: str) -> torch.Tensor:
    """(V, T) float32 unit-normalized per-vertex time series"""
    gii = nib.load(path)
    X = torch.tensor(np.stack([d.data for d in gii.darrays], axis=1), dtype=torch.float32)
    eps = 1e-8
    X = X - X.mean(dim=1, keepdim=True)
    X = X / (X.std(dim=1, keepdim=True) + eps)
    X = X / (X.norm(p=2, dim=1, keepdim=True) + eps)
    return X

def _parcel_timecourses(
    lh_ts: torch.Tensor, rh_ts: torch.Tensor,
    lh_labels: torch.Tensor, rh_labels: torch.Tensor,
    lh_cortex_mask: torch.Tensor, rh_cortex_mask: torch.Tensor,
    L: int
) -> np.ndarray:
    """
    Build parcel-averaged time courses for a single subject.
    Returns (L, T) float32.
    """
    # Keep cortex vertices only
    lh_ts_c = lh_ts[lh_cortex_mask]
    rh_ts_c = rh_ts[rh_cortex_mask]
    lh_lab_c = lh_labels[lh_cortex_mask].long()
    rh_lab_c = rh_labels[rh_cortex_mask].long()

    T = lh_ts_c.shape[1]
    out = torch.zeros((L, T), dtype=torch.float32)
    counts = torch.zeros(L, dtype=torch.int32)

    # Left
    for l in torch.unique(lh_lab_c):
        idx = (lh_lab_c == l).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            out[l] = lh_ts_c[idx].mean(dim=0)
            counts[l] += 1

    # Right
    for l in torch.unique(rh_lab_c):
        idx = (rh_lab_c == l).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            out[l] = rh_ts_c[idx].mean(dim=0) if counts[l] == 0 else (out[l] + rh_ts_c[idx].mean(dim=0)) / 2.0
            counts[l] += 1

    # If a parcel happened to be empty, leave zeros (rare after your empty‑parcel fix)
    return out.numpy()

def _corrcoef_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise correlation: X is (L, T) -> (L, L) correlation matrix."""
    # Normalize rows
    X = X - X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    X = X / std
    return (X @ X.T) / X.shape[1]

def _binarize_top_p(M: np.ndarray, p: float = 0.10, symmetric: bool = True) -> np.ndarray:
    """Keep top p proportion of off-diagonal entries as 1, else 0. Diagonal set to 0."""
    L = M.shape[0]
    A = M.copy()
    np.fill_diagonal(A, -np.inf)
    if symmetric:
        # Use upper triangle to set a single threshold
        triu = A[np.triu_indices(L, k=1)]
        k = max(1, int(np.floor(p * triu.size)))
        thresh = np.partition(triu, -k)[-k]
        B = (M >= thresh).astype(np.uint8)
        B = np.triu(B, 1)
        B = B + B.T
    else:
        flat = A.flatten()
        flat = flat[~np.isinf(flat)]
        k = max(1, int(np.floor(p * flat.size)))
        thresh = np.partition(flat, -k)[-k]
        B = (M >= thresh).astype(np.uint8)
    np.fill_diagonal(B, 0)
    return B

# --------- vMF mixture on the sphere (rows as features) ---------

def _unit_rows(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / nrm

def _approx_kappa_bar(Rbar: float, D: int) -> float:
    """
    Fast approximation of kappa given mean resultant length Rbar in D dims.
    For D>=3: Banerjee et al. (2005) approx.
    """
    if Rbar < 1e-6:
        return 0.0
    return Rbar * (D - Rbar**2) / (1 - Rbar**2)

def spherical_kmeans(X: np.ndarray, K: int, n_init: int = 10, max_iter: int = 200, seed: int = 0):
    """
    Spherical k-means (cosine similarity). Returns labels and centroids (unit vectors).
    This is equivalent to vMF with shared kappa and equal mixing proportions.
    """
    rng = np.random.default_rng(seed)
    X = _unit_rows(X)
    N, D = X.shape

    best_inertia = np.inf
    best = None

    for _ in range(n_init):
        # kmeans++ on sphere (simple version)
        centroids = X[rng.choice(N, size=1, replace=False)]
        for _ in range(1, K):
            sim = X @ centroids.T  # cosine
            dist = 1.0 - np.max(sim, axis=1)
            probs = dist / (dist.sum() + 1e-9)
            next_idx = rng.choice(N, p=probs)
            centroids = np.vstack([centroids, X[next_idx]])

        for _it in range(max_iter):
            sim = X @ centroids.T
            labels = np.argmax(sim, axis=1)
            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                sel = (labels == k)
                if np.any(sel):
                    Ck = X[sel].mean(axis=0)
                    nrm = np.linalg.norm(Ck) + 1e-9
                    new_centroids[k] = Ck / nrm
                else:
                    # Re-seed empty cluster
                    new_centroids[k] = X[rng.integers(0, N)]
            if np.allclose(new_centroids, centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        # inertia on sphere: sum(1 - max cosine)
        inertia = np.sum(1.0 - np.max(X @ centroids.T, axis=1))
        if inertia < best_inertia:
            best_inertia = inertia
            best = (labels.copy(), centroids.copy())

    return best

def vmf_mixture_em(X: np.ndarray, K: int, max_iter: int = 200, tol: float = 1e-5, seed: int = 0):
    """
    Simple vMF-EM with diagonal responsibilities (no covariance—vMF is spherical).
    Starts from spherical k-means; estimates cluster-specific kappas.
    Returns labels, mu (unit centroids), kappa (per cluster), resp (N,K).
    """
    rng = np.random.default_rng(seed)
    X = _unit_rows(X)
    N, D = X.shape

    lab, mu = spherical_kmeans(X, K, n_init=5, max_iter=100, seed=seed)
    R = np.eye(K)[lab]  # hard resp as init
    pi = R.mean(axis=0)

    kappas = np.zeros(K)
    for k in range(K):
        sel = (lab == k)
        if np.any(sel):
            Rk_vec = X[sel].sum(axis=0)
            Rk = np.linalg.norm(Rk_vec)
            Rbar = Rk / (sel.sum())
            kappas[k] = _approx_kappa_bar(Rbar, D)
        else:
            kappas[k] = 0.0

    prev_ll = -np.inf
    for _ in range(max_iter):
        # E-step: log π_k + κ_k xᵢ·μ_k
        dot = X @ mu.T  # (N,K)
        loglik_comp = np.log(pi + 1e-12)[None, :] + (dot * kappas[None, :])
        # stabilize
        m = loglik_comp.max(axis=1, keepdims=True)
        resp = np.exp(loglik_comp - m)
        resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-12)

        # M-step: update π, μ, κ
        Nk = resp.sum(axis=0) + 1e-12
        pi = Nk / N
        # μ_k = normalize(Σ_i r_ik xᵢ)
        new_mu = np.zeros_like(mu)
        new_kappa = np.zeros_like(kappas)
        for k in range(K):
            Rk_vec = (resp[:, k:k+1] * X).sum(axis=0)
            Rk = np.linalg.norm(Rk_vec)
            if Rk < 1e-8:
                new_mu[k] = mu[k]
                new_kappa[k] = 0.0
            else:
                new_mu[k] = Rk_vec / (Rk + 1e-9)
                Rbar = Rk / (Nk[k] + 1e-12)
                new_kappa[k] = _approx_kappa_bar(Rbar, D)

        mu = new_mu
        kappas = new_kappa

        # log-likelihood (up to constants): Σ_i log Σ_k π_k exp(κ_k xᵢ·μ_k)
        ll = (m + np.log(resp.sum(axis=1) + 1e-12) + 0.0).sum()
        if np.abs(ll - prev_ll) < tol * (np.abs(prev_ll) + 1.0):
            break
        prev_ll = ll

    labels = resp.argmax(axis=1)
    return labels, mu, kappas, resp

# --------- Top-level pipeline ---------

def build_group_binary_connectome(
    subjects_lh_func: List[str],
    subjects_rh_func: List[str],
    lh_labels: torch.Tensor, rh_labels: torch.Tensor,  # full-length vertex labels (with -1 on medial), values in [0,L)
    lh_cortex_mask: torch.Tensor, rh_cortex_mask: torch.Tensor,
    L: int,
    top_p: float = 0.10
) -> np.ndarray:
    """
    Implements: parcel-mean -> per-subject LxL corr -> binarize top p -> average over subjects.
    Returns (L, L) float32 matrix in [0,1] representing edge prevalence across subjects.
    """
    assert len(subjects_lh_func) == len(subjects_rh_func)
    N = len(subjects_lh_func)
    acc = np.zeros((L, L), dtype=np.float32)

    for s in range(N):
        lh_ts = _gifti_to_data(subjects_lh_func[s])
        rh_ts = _gifti_to_data(subjects_rh_func[s])
        PT = _parcel_timecourses(lh_ts, rh_ts, lh_labels, rh_labels, lh_cortex_mask, rh_cortex_mask, L)  # (L,T)
        C = _corrcoef_rows(PT)  # (L,L)
        B = _binarize_top_p(C, p=top_p, symmetric=True).astype(np.float32)
        acc += B

    return acc / max(1, N)

def cluster_parcels_into_networks(
    group_binary_connectome: np.ndarray,
    K: int,
    method: str = "vmf"
) -> np.ndarray:
    """
    Feature for each parcel i is row i of the group matrix (with i-th diag zero), unit-normalized.
    Cluster rows into K networks using vMF-EM (default) or spherical-kmeans.
    Returns integer labels in [0, K).
    """
    A = group_binary_connectome.copy()
    np.fill_diagonal(A, 0.0)
    X = _unit_rows(A)  # (L, L)
    if method == "vmf":
        labels, _, _, _ = vmf_mixture_em(X, K, max_iter=200, tol=1e-5, seed=42)
    else:
        labels, _ = spherical_kmeans(X, K, n_init=10, max_iter=200, seed=42)
    return labels.astype(np.int32)

def parcel_to_vertex_network_labels(
    lh_parcel_labels: torch.Tensor, rh_parcel_labels: torch.Tensor,
    parcel_to_network: np.ndarray,
    lh_cortex_mask: torch.Tensor, rh_cortex_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map parcel-level network labels back to full-vertex label vectors (per hemisphere).
    Returns two tensors of shape (V_hemi,) with -1 at medial.
    """
    Lh = lh_parcel_labels.max().item() + 1 if lh_parcel_labels.numel() else 0
    Rh = rh_parcel_labels.max().item() + 1 if rh_parcel_labels.numel() else 0
    assert (Lh + Rh) == parcel_to_network.shape[0]

    # Build per-vertex outputs
    lh_v = torch.full_like(lh_parcel_labels, -1, dtype=torch.long)
    rh_v = torch.full_like(rh_parcel_labels, -1, dtype=torch.long)

    # Map cortex vertices: read which parcel they belong to, then map parcel->network
    lh_p = lh_parcel_labels[lh_cortex_mask].long().cpu().numpy()
    rh_p = rh_parcel_labels[rh_cortex_mask].long().cpu().numpy()
    lh_net = parcel_to_network[:Lh][lh_p]
    rh_net = parcel_to_network[Lh:][rh_p]

    lh_v[lh_cortex_mask] = torch.from_numpy(lh_net).long()
    rh_v[rh_cortex_mask] = torch.from_numpy(rh_net).long()
    return lh_v, rh_v
