import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from nipype.interfaces.workbench.base import WBCommand

from utilities.download_batch import download_batch
from utilities.spgrad_rsfc_gradient import spgrad_rsfc_gradient
from utilities.create_border import gradient_vertices_to_matrix
from utilities.read_avg_mesh import CBIG_AvgMesh_intersection
from utilities.prepare_func import load_subject_func_timeseries, normalize_subject_timeseries
# ========== CONFIGURATION ==========

RAW_DATA_DIR = Path("")
PROCESSED_DIR = Path("")
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
EDGE_DIR = PROCESSED_DIR / "edge_density"
FUNC_DIR = PROCESSED_DIR / "group_func"
MESH_DIR = PROCESSED_DIR / "mesh"


EDGE_DIR.mkdir(exist_ok=True)
FUNC_DIR.mkdir(exist_ok=True)
MESH_DIR.mkdir(exist_ok=True)


LOGFILE = PROCESSED_DIR / "processed_subjects.txt"
GPU_DATA_ROOT = ""
GPU_RESULTS_ROOT = ""
REMOTE_DONE_LOG = f"{GPU_RESULTS_ROOT}/processed_subjects.txt"
LOCAL_DATA_ROOT = RAW_DATA_DIR  
GPU_USERNAME = "daechul"
GPU_HOST = ""

BATCH_SIZE = 10  

# ========== HELPERS ==========

def subject_outputs_exist(subject: str) -> bool:
    lh = EDGE_DIR / f"{subject}_lh_edge_density.npy"
    rh = EDGE_DIR / f"{subject}_rh_edge_density.npy"
    # optionally also check FUNC_DIR files:
    # fl = FUNC_DIR / f"{subject}_lh.pt"
    # fr = FUNC_DIR / f"{subject}_rh.pt"
    return lh.exists() and rh.exists()

def run_output_exists(subject: str, run_name: str) -> bool:
    run_dir = EDGE_DIR / subject / run_name
    return (run_dir / "gradients_edge_density.dtseries.nii").exists()


def get_processed_subjects(logfile):
    if not logfile.exists():
        return set()
    with open(logfile) as f:
        return set(line.strip() for line in f if line.strip())

def log_processed_subject(logfile, subject):
    with open(logfile, 'a') as f:
        f.write(f"{subject}\n")

def clear_download_dir():
    for p in RAW_DATA_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    print(f"[clean] Cleared {RAW_DATA_DIR}")

def list_all_subjects_from_csv(csv_file):
    import csv
    with open(csv_file) as f:
        return [row[0] for row in csv.reader(f)]

# ========== PROCESSING FUNCTIONS ==========

def process_one_subject(subj_dir, subject):
    if subject_outputs_exist(subject):
        print(f"[skip] {subject} already has final outputs locally.")
        return True
    
    fs_dir = subj_dir / "hcp1200" / "fsaverage_LR32k"
    rest_dir = subj_dir / "hcp1200" / "rest"
    if not (fs_dir.exists() and rest_dir.exists()):
        print(f"Skipping {subject}: missing fs or rest dir.")
        return

    def get_fs_paths(fs_dir, subject):
        return {
            "lh_surf": fs_dir / f"{subject}.L.sphere.32k_fs_LR.surf.gii",
            "rh_surf": fs_dir / f"{subject}.R.sphere.32k_fs_LR.surf.gii",
            "lh_midsurf": fs_dir / f"{subject}.L.midthickness.32k_fs_LR.surf.gii",
            "rh_midsurf": fs_dir / f"{subject}.R.midthickness.32k_fs_LR.surf.gii",
            "lh_label": fs_dir / f"{subject}.L.aparc.32k_fs_LR.label.gii",
            "rh_label": fs_dir / f"{subject}.R.aparc.32k_fs_LR.label.gii",
        }
    fs_paths = get_fs_paths(fs_dir, subject)

    lh_edge_maps, rh_edge_maps = [], []
    for run_dir in sorted(rest_dir.iterdir()):
        run_name = run_dir.name
        run_output_dir = EDGE_DIR / subject / run_name
        if run_output_exists(subject, run_name):
            print(f"[run-skip] {subject} {run_name} already has gradients.")
            edge_file = run_output_dir / "gradients_edge_density.dtseries.nii"
            edge_density = nib.load(str(edge_file)).get_fdata()[0]
            lh_n = rh_n = 32492
            lh_edge_maps.append(edge_density[:lh_n])
            rh_edge_maps.append(edge_density[lh_n:])
            continue

        dtseries_file = run_dir / f"{run_name}_Atlas.dtseries.nii"
        if not dtseries_file.exists():
            continue
        lh_func = run_dir / f"{run_name}_Atlas.L.func.gii"
        rh_func = run_dir / f"{run_name}_Atlas.R.func.gii"
        if not lh_func.exists() or not rh_func.exists():
            print('separating')
            cmd = WBCommand(
                f'wb_command -cifti-separate "{dtseries_file}" COLUMN '
                f'-metric CORTEX_LEFT "{lh_func}" '
                f'-metric CORTEX_RIGHT "{rh_func}"'
            )
            cmd.run()

        run_output_dir = EDGE_DIR / subject / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        result = spgrad_rsfc_gradient(
            lh_func, rh_func,
            fs_paths['lh_surf'], fs_paths['rh_surf'],
            fs_paths['lh_midsurf'], fs_paths['rh_midsurf'],
            lh_label=fs_paths['lh_label'], rh_label=fs_paths['rh_label'],
            mesh='fs_LR_32k', medial_mask=None,
            sub_FC=10, sub_verts=200,
            output_dir=run_output_dir
        )
        edge_density_file = Path(result["edge_density"])
        edge_density = nib.load(str(edge_density_file)).get_fdata()[0]
        lh_n = 32492  
        rh_n = 32492
        lh_edge_maps.append(edge_density[:lh_n])
        rh_edge_maps.append(edge_density[lh_n:])

    if not lh_edge_maps or not rh_edge_maps:
        print(f"No edge maps for {subject}")
        return

    mean_lh_edge = np.mean(np.stack(lh_edge_maps), axis=0)
    mean_rh_edge = np.mean(np.stack(rh_edge_maps), axis=0)
    np.save(EDGE_DIR / f"{subject}_lh_edge_density.npy", mean_lh_edge)
    np.save(EDGE_DIR / f"{subject}_rh_edge_density.npy", mean_rh_edge)

    # Save normalized time series for this subject (optional but recommended)
    subj_ts_lh = load_subject_func_timeseries(subj_dir, hemi='L')
    subj_ts_rh = load_subject_func_timeseries(subj_dir, hemi='R')
    if subj_ts_lh is not None:
        norm_lh = normalize_subject_timeseries(subj_ts_lh)
        torch.save(norm_lh, FUNC_DIR / f"{subject}_lh.pt")
    if subj_ts_rh is not None:
        norm_rh = normalize_subject_timeseries(subj_ts_rh)
        torch.save(norm_rh, FUNC_DIR / f"{subject}_rh.pt")
        
    return True

def process_batch(subject_list):
    for subject in subject_list:
        subj_dir = RAW_DATA_DIR / subject
        try:
            ok = process_one_subject(subj_dir, subject)
        except Exception as e:
            print(f"[error] {subject}: {e}")
            ok = False
        if ok:
            log_processed_subject(LOGFILE, subject)
            print(f"Processed {subject}")
        else:
            print(f"[not-logged] {subject} not completed.")
        
    print("[batch] cleanup raw")
    clear_download_dir()
# ========== AGGREGATION FUNCTIONS ==========

def aggregate_all_results():
    # Edge density group average
    lh_files = sorted((EDGE_DIR).glob("*_lh_edge_density.npy"))
    rh_files = sorted((EDGE_DIR).glob("*_rh_edge_density.npy"))
    if not lh_files or not rh_files:
        print("No subject-level edge-density files found; skipping aggregation.")
        return
    lh_all = np.stack([np.load(f) for f in lh_files], axis=0)
    rh_all = np.stack([np.load(f) for f in rh_files], axis=0)
    group_lh = lh_all.mean(axis=0)
    group_rh = rh_all.mean(axis=0)
    np.save(PROCESSED_DIR / "group_mean_lh_edge_density.npy", group_lh)
    np.save(PROCESSED_DIR / "group_mean_rh_edge_density.npy", group_rh)
    print("Saved final group mean edge densities.")

    # Time series group average
    lh_func_files = sorted((FUNC_DIR).glob("*_lh.pt"))
    rh_func_files = sorted((FUNC_DIR).glob("*_rh.pt"))
    lh_all_func = torch.stack([torch.load(f) for f in lh_func_files], dim=0)
    rh_all_func = torch.stack([torch.load(f) for f in rh_func_files], dim=0)
    group_lh_func = lh_all_func.mean(dim=0)
    group_rh_func = rh_all_func.mean(dim=0)
    torch.save(group_lh_func, PROCESSED_DIR / "group_mean_lh_func.pt")
    torch.save(group_rh_func, PROCESSED_DIR / "group_mean_rh_func.pt")
    print("Saved group mean fMRI time series.")

    group_lh = np.load(PROCESSED_DIR / "group_mean_lh_edge_density.npy")
    group_rh = np.load(PROCESSED_DIR / "group_mean_rh_edge_density.npy")

    avg_mesh_lh = CBIG_AvgMesh_intersection(
        RAW_DATA_DIR.parent / "processed" / "edge_density",  
        'lh', surf_type='inflated'
    )
    avg_mesh_rh = CBIG_AvgMesh_intersection(
        RAW_DATA_DIR.parent / "processed" / "edge_density",
        'rh', surf_type='inflated'
    )
    torch.save(avg_mesh_lh, PROCESSED_DIR / "avg_mesh_lh.pt")
    torch.save(avg_mesh_rh, PROCESSED_DIR / "avg_mesh_rh.pt")
    print("Saved group-level average mesh (lh/rh).")

    lh_border_matrix = gradient_vertices_to_matrix(group_lh, avg_mesh_lh['vertexNbors'])
    rh_border_matrix = gradient_vertices_to_matrix(group_rh, avg_mesh_rh['vertexNbors'])

    np.save(PROCESSED_DIR / "lh_border_matrix.npy", lh_border_matrix)
    np.save(PROCESSED_DIR / "rh_border_matrix.npy", rh_border_matrix)
    print("Saved group-level border matrices (lh/rh).")

# ========== MAIN ==========

def main():
    all_subjects = list_all_subjects_from_csv("utilities/data/1200_ids.csv")  
    processed_set = get_processed_subjects(LOGFILE)
    subjects_to_process = [
    s for s in all_subjects
        if (s not in processed_set) and (not subject_outputs_exist(s))
    ]
    print(f"Total subjects to process: {len(subjects_to_process)}")

    for i in range(0, len(subjects_to_process), BATCH_SIZE):
        batch = subjects_to_process[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1} of {((len(subjects_to_process)-1)//BATCH_SIZE)+1}")
        print(f'new batch')
        download_batch(batch, GPU_DATA_ROOT, LOCAL_DATA_ROOT, GPU_USERNAME,GPU_HOST)  
        process_batch(batch)       # Process and save to PROCESSED_DIR
        clear_download_dir()       # Delete batch to free space

    aggregate_all_results()

if __name__ == "__main__":
    main()
