# main.py
import argparse
from pathlib import Path
import torch
import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modularity.loader import build_hemi_inputs_from_edge_density_and_timeseries
from modularity.utils import visualize_labels, _to_numpy, visualize_labels_int
from modularity.trainer import Trainer

def run_one_hemi(
    name,
    hemi,                    # "lh" or "rh"
    processed_dir: Path,
    func_path: Path,
    edge_density_path: Path,
    opt,
    n_clusters,
    mu_spatial,
    epochs,
    device
):
    print(f"\n=== Processing hemisphere: {name} ({hemi}) ===")

    # Load precomputed aggregated mesh
    mesh_path = processed_dir / f"avg_mesh_{hemi}.pt"
    avg_mesh = torch.load(mesh_path, map_location="cpu", weights_only=False)
    # Load features
    X_all = torch.load(func_path, map_location="cpu", weights_only=False)   # (V, T)
    g_all = np.load(edge_density_path)                  # (V,)
    # Build model inputs
    X_f, Y_s, adj_f, norm_adj, L_s = build_hemi_inputs_from_edge_density_and_timeseries(
        avg_mesh, g_all, X_all
    )

    # Move to device
    X_f = X_f.to(device)
    Y_s = Y_s.to(device)
    adj_f = adj_f.to(device)
    norm_adj = norm_adj.to(device)
    L_s = L_s.to(device)

    # Mask for "full length" label visualization
    nonmedial_wall_idx = get_cortex_mask(avg_mesh)  # <-- we’ll define helper below

    opt_local = dict(opt)
    opt_local["num_feature_f"] = X_f.shape[1]   # T
    opt_local["num_feature_s"] = Y_s.shape[1]   # 3

    trainer = Trainer(
        device=device,
        opt=opt_local,
        n_clusters=n_clusters,
        norm_adj=norm_adj,
        adj_f=adj_f,
        L_s=L_s,
        mu_spatial=mu_spatial,
    )

    assignments, loss_history = trainer.fit(X_f, Y_s, n_epochs=epochs)

    labels = trainer.cluster_labels(assignments)
    labels = np.asarray(labels, dtype=np.int32)  # (V_cortex,)

    # labels is mesh-space length V_mesh (29696)
    labels = np.asarray(labels, dtype=np.int32)

    V_full = X_all.shape[0]  # should be 32492
    vis_labels = np.full((V_full,), -1, dtype=np.int32)

    idx = np.asarray(avg_mesh["cortex_indices"], dtype=np.int64)

    # convert 1-based -> 0-based if needed
    if idx.min() == 1 and idx.max() == V_full:
        idx = idx - 1

    # now assign
    vis_labels[idx] = labels

    # # Full-length labels with medial wall = -1 (length V_all)
    # vis_labels = np.full(nonmedial_wall_idx.shape[0], -1, dtype=np.int32)
    # vis_labels[nonmedial_wall_idx] = labels

    # Save
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / f"{name}_{hemi}_labels_int.npy", labels)
    np.save(results_dir / f"{name}_{hemi}_labels_full_int.npy", vis_labels)

    print(f"{name} ({hemi}): done. clusters used={len(np.unique(labels))}")
    return labels

def get_cortex_mask(avg_mesh):
    V = avg_mesh["coords"].shape[0]   # 29696
    return np.ones(V, dtype=bool)
# def get_cortex_mask(avg_mesh):
#     """
#     Returns boolean mask over V_all that matches the cortex vertices used by loader.
#     We prefer 'cortex_indices' if present (from intersection mesh),
#     otherwise fall back to MARS_label != -1.
#     """
#     if "cortex_indices" in avg_mesh and avg_mesh["cortex_indices"] is not None:
#         idx = np.asarray(avg_mesh["cortex_indices"])
#         mask = np.zeros(int(np.asarray(avg_mesh["coords"]).shape[0]), dtype=bool)
#         mask[idx] = True
#         return mask

#     mars = np.asarray(avg_mesh["MARS_label"])
#     return mars != -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--clusters", type=int, default=200) ## Or 17
    p.add_argument("--mu_spatial", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cuda", type=str, default='cuda:7')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    PROCESSED_DIR = Path("/cranberry/daechul/data/processed")

    lh_func_path = PROCESSED_DIR / "group_mean_lh_func.pt"
    rh_func_path = PROCESSED_DIR / "group_mean_rh_func.pt"
    lh_edge_density = PROCESSED_DIR / "group_mean_lh_edge_density.npy"
    rh_edge_density = PROCESSED_DIR / "group_mean_rh_edge_density.npy"

    opt = {
        "num_feature": 3,
        "hidden_dim": 1024,
        "collapse_regularization": 0.1,
        "dropout_rate": 0.1,
        "activation": "selu",
        "skip_connection": True,
        "lr": args.lr,
        "cuda": args.cuda,
    }

    lh_labels = run_one_hemi(
        name="LH",
        hemi="lh",
        processed_dir=PROCESSED_DIR,
        func_path=lh_func_path,
        edge_density_path=lh_edge_density,
        opt=opt,
        n_clusters=args.clusters,
        mu_spatial=args.mu_spatial,
        epochs=args.epochs,
        device=args.cuda
    )

    rh_labels = run_one_hemi(
        name="RH",
        hemi="rh",
        processed_dir=PROCESSED_DIR,
        func_path=rh_func_path,
        edge_density_path=rh_edge_density,
        opt=opt,
        n_clusters=args.clusters,
        mu_spatial=args.mu_spatial,
        epochs=args.epochs,
        device=args.cuda
    )

    np.save(PROCESSED_DIR / "lh_labels.npy", lh_labels)
    np.save(PROCESSED_DIR / "rh_labels.npy", rh_labels)

