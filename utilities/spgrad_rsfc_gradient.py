import os
import numpy as np
import nibabel as nib
from pathlib import Path
from nipype.interfaces.workbench.base import WBCommand
from nibabel.cifti2 import Cifti2Image, Cifti2Header, ScalarAxis, SeriesAxis
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nipype.interfaces.workbench.base import WBCommand
from utilities.read_avg_mesh import *
from utilities.spgrad_findminima import find_local_minima
from utilities.spgrad_watershed_algorithm import watershed_algorithm, get_K_hop_neighbors
from nibabel.cifti2 import Cifti2Image, Cifti2Header, ScalarAxis, SeriesAxis

def spgrad_rsfc_gradient(
    lh_time_series_file, rh_time_series_file, 
    lh_surf, rh_surf,
    lh_midsurf, rh_midsurf,   # add midsurface file paths here for smoothing!
    lh_label, rh_label,
    mesh='fs_LR_32k', medial_mask=None, 
    sub_FC=10, sub_verts=200, 
    output_dir='output'
    ):
    """
    Compute RSFC gradient, smooth, and output both the local gradient map and an edge density map.
    """

    # --------- 1. ORIGINAL GRADIENT PIPELINE ---------
    out = Path(output_dir)
    tmp_dir = out / 'tmp'
    dump_dir = out / 'dump'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    merged = tmp_dir / 'merged.dtseries.nii'
    merge_cmd = WBCommand(
    f'wb_command -cifti-create-dense-timeseries "{merged}" '
    f'-left-metric "{lh_time_series_file}" -right-metric "{rh_time_series_file}" -timestep 0 -timestart 0'
    )
    # merge_cmd = WBCommand(
    #     f'wb_command -cifti-create-dense-timeseries {merged} '
    #     f'-left-metric {lh_time_series_file} -right-metric {rh_time_series_file} -timestep 0 -timestart 0'
    # )
    merge_cmd.run()
    dt_img = nib.load(merged)
    dt_data = dt_img.get_fdata(dtype=np.float32)
    data = dt_data.T
    n_vert, n_tp = data.shape
    if n_vert not in (64984, 91282):
        raise ValueError(f'Unexpected number of vertices: {n_vert}')

    # Medial mask
    if medial_mask is None:
        def load_label(gii_path):
            gii = nib.load(gii_path)
            return np.asarray(gii.darrays[0].data, dtype=int)
        lh_label_path = Path(lh_label) if isinstance(lh_label, str) else lh_label
        rh_label_path = Path(rh_label) if isinstance(rh_label, str) else rh_label
        lh_label = load_label(lh_label_path)
        rh_label = load_label(rh_label_path)
        mask = np.concatenate([lh_label == -1, rh_label == -1])
    else:
        mask = np.asarray(medial_mask, dtype=bool)
        if mask.size != n_vert:
            raise ValueError('medial_mask length must match number of vertices')

    data = data[~mask, :]
    n_valid = data.shape[0]
    rng = np.random.default_rng(seed=5049)
    inds_FC = rng.choice(n_valid, size=n_valid // sub_FC, replace=False)
    inds_verts = rng.choice(n_valid, size=n_valid // sub_verts, replace=False)
    a, b = 3, 10
    n2 = len(inds_FC)
    n1 = len(inds_verts)
    iter_a = a + (n1 % a != 0)
    iter_b = b + (n_valid % b != 0)
    bs_a = n1 // a
    bs_b = n_valid // b

    t_series = data[inds_FC, :]
    t_series -= t_series.mean(axis=1, keepdims=True)
    mag_t = np.linalg.norm(t_series, axis=1)[:, None]
    original_brain_axis = dt_img.header.get_axis(1)
    brain_axis = original_brain_axis[~mask]

    intermediates = []
    for i in range(iter_a):
        start_a = i * bs_a
        end_a = (i + 1) * bs_a if i < iter_a - 1 else n1
        sel_verts = inds_verts[start_a:end_a]
        
        # compute FC_A
        s_series = data[sel_verts, :].T
        s_series -= s_series.mean(axis=0)
        mag_s = np.linalg.norm(s_series, axis=0)[None, :]
        mag_t[mag_t == 0] = 1e-12
        mag_s[mag_s == 0] = 1e-12
        FC_A = (t_series @ s_series) / (mag_t @ mag_s)
        FC_simi = np.zeros((n_valid, FC_A.shape[1]), dtype=np.float32)
        
        # iterate FC_B blocks
        for j in range(iter_b):
            start_b = j * bs_b
            end_b = (j + 1) * bs_b if j < iter_b - 1 else n_valid
            sb = data[start_b:end_b, :].T
            sb -= sb.mean(axis=0)
            mag_sb = np.linalg.norm(sb, axis=0)[None, :]
            mag_sb[mag_sb == 0] = 1e-12
            FC_B = (t_series @ sb) / (mag_t @ mag_sb)
            FAc = FC_A - FC_A.mean(axis=0)
            FBc = FC_B - FC_B.mean(axis=0)
            mag_a = np.linalg.norm(FAc, axis=0)[None, :]
            mag_b = np.linalg.norm(FBc, axis=0)[:, None]
            mag_a[mag_a == 0] = 1e-12
            mag_b[mag_b == 0] = 1e-12
            simi = (FBc.T @ FAc) / (mag_b @ mag_a)
            FC_simi[start_b:end_b, :] = simi.astype(np.float32)

        data_arr = FC_simi.T
        series_ax = SeriesAxis(start=0.0, step=0.0, size=data_arr.shape[0])
        hdr = Cifti2Header.from_axes((series_ax, brain_axis))
        out_block = tmp_dir / f'FC_simi_block_{i+1}.dtseries.nii'
        Cifti2Image(data_arr, hdr).to_filename(str(out_block))
        intermediates.append(out_block)

    grads = []
    print('start to run gradient command line')
    for ds in intermediates:
        base = ds.stem
        print(base)
        out_grad = tmp_dir / f'{base}_grad.dtseries.nii'
        cmd = WBCommand(
            f'wb_command -cifti-gradient "{ds}" COLUMN "{out_grad}" '
            f'-left-surface "{lh_surf}" -right-surface "{rh_surf}"'
        )
        
        cmd.run()
        img = nib.load(str(out_grad))
        grads.append(img.get_fdata(dtype=np.float32))
        print('run gradient command line successfully')

    all_grads = np.concatenate(grads, axis=0)
    mean_grad = all_grads.mean(axis=0)
    print(mean_grad.shape, all_grads.shape)
    data_arr = mean_grad[np.newaxis, :]
    final_ax = ScalarAxis(['Gradient'])
    final_hdr = Cifti2Header.from_axes((final_ax, brain_axis))
    local_grad_path = out / 'local_gradient_map.dscalar.nii'
    Cifti2Image(data_arr, final_hdr).to_filename(str(local_grad_path))
    print("Wrote final gradient to", local_grad_path)

    sigma = 2.55
    smoothed_grad_path = out / f'local_gradient_map_smooth{sigma}.dscalar.nii'
    cmd = WBCommand(
        f'wb_command -cifti-smoothing "{local_grad_path}" {sigma} {sigma} COLUMN "{smoothed_grad_path}" '
        f'-left-surface "{lh_midsurf}" -right-surface "{rh_midsurf}"'
    )
    # cmd = WBCommand(
    #     f'wb_command -cifti-smoothing {local_grad_path} {sigma} {sigma} COLUMN {smoothed_grad_path} '
    #     f'-left-surface {lh_midsurf} -right-surface {rh_midsurf}'
    # )
    cmd.run()
    print(f"Smoothed gradient map written to {smoothed_grad_path}")
    sm_grad = nib.load(str(smoothed_grad_path)).get_fdata(dtype=np.float32).ravel()
    print("Calculating edge density using watershed algorithm...")
  
    lh_mesh = nib.load(lh_surf)
    rh_mesh = nib.load(rh_surf)
    lh_faces = lh_mesh.darrays[1].data
    rh_faces = rh_mesh.darrays[1].data
    if lh_faces.max() < rh_faces.min():
        faces = np.vstack([lh_faces, rh_faces])
    else:
        lh_vertices_count = lh_mesh.darrays[0].data.shape[0]
        rh_faces_offset = rh_faces + lh_vertices_count
        faces = np.vstack([lh_faces, rh_faces_offset])
    vertex_nbors = build_masked_vertex_nbors(faces, ~mask) 
    K = 3
    K_neighbors = get_K_hop_neighbors(vertex_nbors, K=K)  
    max_K = K_neighbors.shape[1]
    minimametric = find_local_minima(sm_grad, K_neighbors)
    stepnum = 50
    fracmaxh = 1.0
    minh = float(np.min(sm_grad))
    maxh = float(np.max(sm_grad))
    labels, watershed_zones = watershed_algorithm(
        sm_grad, minimametric, stepnum, fracmaxh, vertex_nbors, minh, maxh, random_seed=42
    )

    edge_density = watershed_zones.astype(np.float32)

    # Write to CIFTI
    final_ax = ScalarAxis(['EdgeDensity'])
    final_hdr = Cifti2Header.from_axes((final_ax, brain_axis))
    out_edge = out / 'gradients_edge_density.dtseries.nii'
    Cifti2Image(edge_density[np.newaxis, :], final_hdr).to_filename(str(out_edge))
    print("Wrote edge density to", out_edge)

    return {
        "local_gradient": str(local_grad_path),
        "smoothed_gradient": str(smoothed_grad_path),
        "edge_density": str(out_edge)
    }
