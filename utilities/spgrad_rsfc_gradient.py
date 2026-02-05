# utilities/spgrad_rsfc_gradient.py 
## fixed version
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from nibabel.cifti2 import Cifti2Image, Cifti2Header, ScalarAxis, SeriesAxis
import subprocess
from utilities.spgrad_findminima import find_local_minima
from utilities.spgrad_watershed_algorithm import watershed_algorithm, get_K_hop_neighbors


def _load_roi_or_label_mask(lh_path: Path, rh_path: Path) -> np.ndarray:
    """
    Return boolean mask of length 64984 (True = medial wall / exclude).
    Preferred input: atlasroi.shape.gii (cortex=1, medial=0).
    Fallback input: aparc.label.gii (medial often -1).
    """
    def read_first_array(gii_path: Path) -> np.ndarray:
        g = nib.load(str(gii_path))
        return np.asarray(g.darrays[0].data)

    lh = read_first_array(lh_path)
    rh = read_first_array(rh_path)
    lh_u = np.unique(lh)
    rh_u = np.unique(rh)

    def looks_like_atlasroi(u):
        # atlasroi typically {0,1} or {0,1,2} depending on file; definitely not thousands of parcels
        return (u.size <= 10) and np.all(np.isin(u, [0, 1, 2]))

    if looks_like_atlasroi(lh_u) and looks_like_atlasroi(rh_u):
        # atlasroi: cortex usually 1, medial wall 0
        lh_mask = (lh == 0)
        rh_mask = (rh == 0)
    else:
        # label: medial often -1; also treat 0 as non-cortex if present
        lh_mask = (lh == -1) | (lh == 0)
        rh_mask = (rh == -1) | (rh == 0)
    # Heuristic: atlasroi is typically {0,1} float; aparc labels are ints with -1 for medial wall.
    # if np.issubdtype(lh.dtype, np.floating) or np.issubdtype(rh.dtype, np.floating):
    #     # atlasroi-like
    #     lh_mask = (lh == 0)
    #     rh_mask = (rh == 0)
    # else:
    #     # label-like fallback (common: -1 medial wall)
    #     lh_mask = (lh == -1) | (lh == 0)
    #     rh_mask = (rh == -1) | (rh == 0)

    mask = np.concatenate([lh_mask, rh_mask]).astype(bool)
    if mask.shape[0] != 64984:
        raise ValueError(f"Mask length expected 64984, got {mask.shape[0]}")
    return mask


def _build_vertex_neighbors_from_faces(faces: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    nbrs = [set() for _ in range(n_vertices)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        nbrs[a].update([b, c])
        nbrs[b].update([a, c])
        nbrs[c].update([a, b])
    return [np.fromiter(s, dtype=np.int32) for s in nbrs]


def _restrict_neighbors_to_valid(vertex_nbors_full: list[np.ndarray], valid_idx: np.ndarray, n_full: int) -> list[np.ndarray]:
    """
    Given neighbors in full space [0..n_full-1], restrict to valid_idx vertices and reindex to [0..n_valid-1].
    """
    old2new = -np.ones((n_full,), dtype=np.int32)
    old2new[valid_idx] = np.arange(valid_idx.shape[0], dtype=np.int32)

    valid_set = set(valid_idx.tolist())
    out = []
    for old_v in valid_idx:
        nb = vertex_nbors_full[int(old_v)]
        nb_valid = [old2new[int(u)] for u in nb if int(u) in valid_set]
        out.append(np.asarray(nb_valid, dtype=np.int32))
    return out


def spgrad_rsfc_gradient(
    lh_time_series_file,
    rh_time_series_file,
    lh_surf,
    rh_surf,
    lh_midsurf,
    rh_midsurf,
    lh_mask_file,
    rh_mask_file,
    mesh="fs_LR_32k",
    sub_FC=10,
    sub_verts=200,
    output_dir="output",
):
    """
    Speed-up RSFC gradient -> watershed edges -> edge density (single-run edge map here).

    Key properties of THIS implementation:
      - Uses atlasroi (preferred) to define medial wall mask
      - Writes FULL-LENGTH outputs (64984) with medial wall = 0
      - Uses ROI during smoothing to avoid bleeding across medial wall
    """

    out = Path(output_dir)
    tmp_dir = out / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1) Merge L/R metric time series into a CIFTI dtseries (cortex-only)
    merged = tmp_dir / "merged.dtseries.nii"
    cmd = (
        f'wb_command -cifti-create-dense-timeseries "{merged}" '
        f'-left-metric "{lh_time_series_file}" -right-metric "{rh_time_series_file}" '
        f'-timestep 1 -timestart 0'
    )
    if not merged.exists():
        os.system(cmd)

    dt_img = nib.load(str(merged))
    dt_data = dt_img.get_fdata(dtype=np.float32)  # typically (T, N)
    if dt_data.ndim != 2:
        raise ValueError(f"Unexpected dtseries shape: {dt_data.shape}")

    T, N = dt_data.shape[0], dt_data.shape[1]
    if N != 64984:
        raise ValueError(f"Expected 64984 cortex vertices (L+R), got {N}")

    data_full = dt_data.T  # (N, T)

    # 2) Medial wall mask
    mask = _load_roi_or_label_mask(Path(lh_mask_file), Path(rh_mask_file))  # True=exclude
    valid_idx = np.where(~mask)[0]
    data = data_full[valid_idx, :]
    n_valid = data.shape[0]

    # 3) Subsampling parameters
    rng = np.random.default_rng(seed=5049)
    n2 = max(1, n_valid // int(sub_FC))
    n1 = max(1, n_valid // int(sub_verts))
    inds_FC = rng.choice(n_valid, size=n2, replace=False)
    inds_verts = rng.choice(n_valid, size=n1, replace=False)

    # split blocks
    a, b = 3, 10
    iter_a = a + (n1 % a != 0)
    iter_b = b + (n_valid % b != 0)
    bs_a = max(1, n1 // a)
    bs_b = max(1, n_valid // b)

    # normalize t_series
    t_series = data[inds_FC, :]
    t_series = t_series - t_series.mean(axis=1, keepdims=True)
    mag_t = np.linalg.norm(t_series, axis=1, keepdims=True)
    mag_t[mag_t == 0] = 1e-12

    # We'll write FC_simi blocks as FULL-LENGTH CIFTI so wb_command gradient works cleanly
    original_brain_axis = dt_img.header.get_axis(1)

    intermediates = []
    for i in range(iter_a):
        start_a = i * bs_a
        end_a = (i + 1) * bs_a if i < iter_a - 1 else n1
        sel_verts = inds_verts[start_a:end_a]
        if sel_verts.size == 0:
            continue

        # FC_A: (n2, block_cols)
        s_series = data[sel_verts, :].T
        s_series = s_series - s_series.mean(axis=0, keepdims=True)
        mag_s = np.linalg.norm(s_series, axis=0, keepdims=True)
        mag_s[mag_s == 0] = 1e-12

        FC_A = (t_series @ s_series) / (mag_t @ mag_s)

        # FC similarity block: (n_valid, block_cols)
        FC_simi = np.zeros((n_valid, FC_A.shape[1]), dtype=np.float32)

        # compute FC_B in blocks
        for j in range(iter_b):
            start_b = j * bs_b
            end_b = (j + 1) * bs_b if j < iter_b - 1 else n_valid
            if end_b <= start_b:
                continue

            sb = data[start_b:end_b, :].T
            sb = sb - sb.mean(axis=0, keepdims=True)
            mag_sb = np.linalg.norm(sb, axis=0, keepdims=True)
            mag_sb[mag_sb == 0] = 1e-12

            FC_B = (t_series @ sb) / (mag_t @ mag_sb)

            FAc = FC_A - FC_A.mean(axis=0, keepdims=True)
            FBc = FC_B - FC_B.mean(axis=0, keepdims=True)
            mag_a = np.linalg.norm(FAc, axis=0, keepdims=True)
            mag_b = np.linalg.norm(FBc, axis=0, keepdims=True).T
            mag_a[mag_a == 0] = 1e-12
            mag_b[mag_b == 0] = 1e-12

            simi = (FBc.T @ FAc) / (mag_b @ mag_a)
            FC_simi[start_b:end_b, :] = simi.astype(np.float32)

        # Expand to full length (N=64984), medial wall = 0
        FC_full = np.zeros((N, FC_simi.shape[1]), dtype=np.float32)
        FC_full[valid_idx, :] = FC_simi

        # Write as dtseries where "time axis" is block columns
        data_arr = FC_full.T  # (block_cols, N)
        series_ax = SeriesAxis(start=0.0, step=1.0, size=data_arr.shape[0])
        hdr = Cifti2Header.from_axes((series_ax, original_brain_axis))
        out_block = tmp_dir / f"FC_simi_block_{i+1}.dtseries.nii"
        Cifti2Image(data_arr, hdr).to_filename(str(out_block))
        intermediates.append(out_block)

    # 4) Compute gradients for each block via wb_command, then average
    grads = []
    for ds in intermediates:
        out_grad = tmp_dir / f"{ds.stem}_grad.dtseries.nii"
        if not out_grad.exists():
            cmdg = (
                f'wb_command -cifti-gradient "{ds}" COLUMN "{out_grad}" '
                f'-left-surface "{lh_surf}" -right-surface "{rh_surf}"'
            )
            os.system(cmdg)

        img = nib.load(str(out_grad))
        g = img.get_fdata(dtype=np.float32)  # (block_cols, N)
        grads.append(g)

    if len(grads) == 0:
        raise RuntimeError("No gradient blocks produced.")

    all_grads = np.concatenate(grads, axis=0)  # (sum_block_cols, N)
    mean_grad_full = all_grads.mean(axis=0)    # (N,)

    # Write local gradient (full length)
    local_grad_path = out / "local_gradient_map.dscalar.nii"
    final_ax = ScalarAxis(["Gradient"])
    final_hdr = Cifti2Header.from_axes((final_ax, original_brain_axis))
    Cifti2Image(mean_grad_full[np.newaxis, :], final_hdr).to_filename(str(local_grad_path))

   
    sigma = 2.55
    smoothed_grad_path = out / f"local_gradient_map_smooth{sigma}.dscalar.nii"
    if not smoothed_grad_path.exists():
        cmd_no_roi = (
        f'wb_command -cifti-smoothing "{local_grad_path}" {sigma} {sigma} COLUMN "{smoothed_grad_path}" '
        f'-left-surface "{lh_midsurf}" -right-surface "{rh_midsurf}"'
        )
        subprocess.run(cmd_no_roi, shell=True, check=True)
        

    sm_grad_full = nib.load(str(smoothed_grad_path)).get_fdata(dtype=np.float32).ravel()
    if sm_grad_full.shape[0] != N:
        raise ValueError("Smoothed gradient length mismatch.")

    # Work in valid space for watershed
    sm_grad = sm_grad_full[valid_idx]

    # 6) Build neighbors restricted to valid vertices
    lh_mesh = nib.load(str(lh_surf))
    rh_mesh = nib.load(str(rh_surf))
    lh_faces = np.asarray(lh_mesh.darrays[1].data, dtype=np.int32)
    rh_faces = np.asarray(rh_mesh.darrays[1].data, dtype=np.int32)

    lh_n = lh_mesh.darrays[0].data.shape[0]
    rh_faces_off = rh_faces + lh_n
    faces_full = np.vstack([lh_faces, rh_faces_off]).astype(np.int32)

    vertex_nbors_full = _build_vertex_neighbors_from_faces(faces_full, N)
    neighbors_valid = _restrict_neighbors_to_valid(vertex_nbors_full, valid_idx, N)

    # 7) Watershed edges
    K = 3
    K_neighbors = get_K_hop_neighbors(neighbors_valid, K=K)
    minimametric = find_local_minima(sm_grad, K_neighbors)

    labels, watershed_zones = watershed_algorithm(
        sm_grad,
        minimametric,
        stepnum=50,
        fracmaxh=1.0,
        neighbors=neighbors_valid,
        minh=float(np.min(sm_grad)),
        maxh=float(np.max(sm_grad)),
        random_seed=42,
    )

    edge_valid = watershed_zones.astype(np.float32)  # (n_valid,)

    # Expand to full space
    edge_full = np.zeros((N,), dtype=np.float32)
    edge_full[valid_idx] = edge_valid

    # Write edge density FULL LENGTH
    out_edge = out / "gradients_edge_density.dscalar.nii"
    edge_ax = ScalarAxis(["EdgeDensity"])
    edge_hdr = Cifti2Header.from_axes((edge_ax, original_brain_axis))
    Cifti2Image(edge_full[np.newaxis, :], edge_hdr).to_filename(str(out_edge))

    return {
        "local_gradient": str(local_grad_path),
        "smoothed_gradient": str(smoothed_grad_path),
        "edge_density": str(out_edge),
    }
