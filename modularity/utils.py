# modularity/utils_train.py
import os
import math
import torch
import numpy as np
from typing import Iterable, Optional, Dict, Any

def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters() if not trainable_only else (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)

def tensor_bytes(x: torch.Tensor) -> int:
    try:
        return x.element_size() * x.numel()
    except Exception:
        return 0

def model_memory_bytes(model: torch.nn.Module, include_buffers: bool = True) -> int:
    bytes_params = sum(tensor_bytes(p) for p in model.parameters())
    bytes_buffers = sum(tensor_bytes(b) for b in model.buffers()) if include_buffers else 0
    return bytes_params + bytes_buffers

def pretty_size(n_bytes: int) -> str:
    if n_bytes == 0: return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    k = int(math.floor(math.log(n_bytes, 1024)))
    k = min(k, len(units) - 1)
    return f"{n_bytes / (1024 ** k):.2f} {units[k]}"

def summarize_model(model: torch.nn.Module) -> Dict[str, Any]:
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    mem = model_memory_bytes(model, include_buffers=True)
    return {
        "parameters_total": total,
        "parameters_trainable": trainable,
        "memory_total_bytes_including_buffers": mem,
        "memory_total_human": pretty_size(mem),
    }

def print_model_summary(model: torch.nn.Module, header: Optional[str] = None):
    if header:
        print("=" * len(header))
        print(header)
        print("=" * len(header))
    s = summarize_model(model)
    print(f"Total params      : {s['parameters_total']:,}")
    print(f"Trainable params  : {s['parameters_trainable']:,}")
    print(f"Estimated footprint (params+buffers): {s['memory_total_human']}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    epoch: int,
                    save_dir: str,
                    tag: Optional[str] = None,
                    extra: Optional[dict] = None) -> str:
    """
    Saves weights + (optionally) optimizer. Returns the filepath.
    """
    ensure_dir(save_dir)
    name = f"epoch{epoch:04d}" + (f"_{tag}" if tag else "")
    path = os.path.join(save_dir, f"{name}.pt")
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict()
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _faces_to_pv(faces):
    """
    PyVista expects a flat array like: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    If faces is (F,3), convert it.
    If it's already flat VTK-style, return as-is.
    """
    faces = _to_numpy(faces)
    if faces.ndim == 2 and faces.shape[1] == 3:
        f = np.concatenate([np.full((faces.shape[0], 1), 3, dtype=faces.dtype), faces], axis=1)
        return f.ravel()
    # already flat (or another valid format)
    return faces.ravel()

def visualize_labels(vertices, faces, labels, hemisphere, tag="final", out_dir="results/clustering/visualize"):
    import os, numpy as np, torch, pyvista as pv
    from matplotlib import cm, colors as mplcolors

    def to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    def faces_to_vtk(f):
        f = to_np(f)
        if f.ndim == 2 and f.shape[1] == 3:
            f = np.concatenate([np.full((f.shape[0], 1), 3, dtype=f.dtype), f], axis=1).ravel()
        else:
            f = f.ravel()
        return f

    try: pv.start_xvfb()
    except Exception: pass

    V = to_np(vertices)
    F = faces_to_vtk(faces)
    L = np.asarray(labels)  # full-length (V,) with -1 for medial wall and 0..K-1 for clusters

    # convert medial wall to NaN (so it’s not colored)
    L_plot = L.astype(float)
    L_plot[L_plot < 0] = np.nan

    # how many clusters?
    K = int(np.nanmax(L_plot)) + 1 if np.any(~np.isnan(L_plot)) else 0

    mesh = pv.PolyData(V, F)
    mesh["labels"] = L_plot

    # discrete colormap with K colors (tab20 has up to 20; extend if needed)
    base_cmap = cm.get_cmap("tab20", max(K, 1))  # ensure at least 1
    cmap = mplcolors.ListedColormap(base_cmap(np.arange(base_cmap.N)))

    # annotations for integer ticks
    annotations = {i: str(i) for i in range(K)}

    os.makedirs(os.path.join(out_dir, hemisphere), exist_ok=True)
    png = os.path.join(out_dir, hemisphere, f"mesh_visualization_{tag}_with_{K}clusters.png")

    pl = pv.Plotter(off_screen=True, window_size=(1280, 960))
    pl.add_mesh(
        mesh,
        scalars="labels",
        cmap=cmap,
        categories=True,              # <- treat as discrete categories
        annotations=annotations,      # <- label integers on the colorbar
        nan_color="lightgray",        # <- medial wall (NaN) color
        show_edges=False,
        clim=(-0.5, K - 0.5) if K > 0 else None,  # <- crisp bins centered on ints
    )
    pl.show(screenshot=png)
    print(f"[VIS] Saved: {png}")

def visualize_labels_int(vertices, faces, labels_int, hemisphere,
                         tag="final", out_dir="results/clustering/visualize"):
    import os, numpy as np, torch, pyvista as pv
    from matplotlib import cm, colors as mplcolors

    def to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    def faces_to_vtk(f):
        f = to_np(f)
        if f.ndim == 2 and f.shape[1] == 3:
            f = np.concatenate([np.full((f.shape[0], 1), 3, dtype=f.dtype), f], axis=1).ravel()
        else:
            f = f.ravel()
        return f

    try: pv.start_xvfb()
    except Exception: pass

    V  = to_np(vertices)
    F  = faces_to_vtk(faces)
    Li = np.asarray(labels_int, dtype=np.int32)  # **integers**, length V

    # K = number of clusters (exclude -1)
    valid = Li >= 0
    K = Li[valid].max() + 1 if valid.any() else 0

    # make mesh and attach **integer** scalars
    mesh = pv.PolyData(V, F)
    mesh["labels_int"] = Li

    # remove medial wall points (label == -1)
    # (threshold keeps points with labels in [0, K-1])
    sub = mesh.threshold((0, max(K-1, 0)), scalars="labels_int") if K > 0 else mesh

    # discrete colormap and annotations
    base_cmap = cm.get_cmap("tab20", max(K, 1))
    cmap = mplcolors.ListedColormap(base_cmap(np.arange(base_cmap.N)))
    annotations = {i: str(i) for i in range(K)}

    os.makedirs(os.path.join(out_dir, hemisphere), exist_ok=True)
    png = os.path.join(out_dir, hemisphere, f"mesh_visualization_{tag}_with_{K}clusters.png")

    pl = pv.Plotter(off_screen=True, window_size=(1280, 960))
    pl.add_mesh(
        sub,
        scalars="labels_int",       # still **int**
        cmap=cmap,
        categories=True,            # treat ints as categories
        annotations=annotations,    # integer tick labels
        show_edges=False,
        clim=(-0.5, K - 0.5) if K > 0 else None,  # crisp bins on integers
    )
    pl.show(screenshot=png)
    print(f"[VIS] Saved: {png}")