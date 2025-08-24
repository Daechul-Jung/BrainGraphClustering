import torch
import numpy as np
import nibabel as nib
from pathlib import Path

def gifti_to_data(gifti_path):
    
    gii = nib.load(str(gifti_path))
    data = torch.tensor(np.stack([d.data for d in gii.darrays], axis=1), dtype=torch.float32)
    
    return data

def load_subject_func_timeseries(subject_dir, hemi='L'):
    """
    Loads and concatenates all runs' func.gii for a subject and hemisphere.
    """
    rest_dir = subject_dir / "hcp1200" / "rest"
    func_files = sorted(rest_dir.glob(f"*/rfMRI_*_Atlas.{hemi}.func.gii"))
    if not func_files:
        return None
    data_list = [gifti_to_data(f) for f in func_files]
    all_data = torch.cat(data_list, dim=1)  
    return all_data

def normalize_subject_timeseries(data, epsilon=1e-10):
    """
    Demean, z-score, and l2-normalize each vertex's time series.
    """
    data = data - data.mean(dim=1, keepdim=True)
    data = data / (data.std(dim=1, keepdim=True) + epsilon)
    lh_norm = torch.norm(data, p=2, dim=1, keepdim=True) + epsilon
    data = data / lh_norm
    return data

def aggregate_group_timeseries(subject_base_dir, output_dir, hemi='L'):
    """
    For all subjects, load, normalize, sum, and normalize again for group-level.
    """
    subject_dirs = [d for d in Path(subject_base_dir).iterdir() if d.is_dir() and (d / "hcp1200" / "rest").exists()]
    group_sum = None
    n_subjects = 0

    for subj_dir in subject_dirs:
        subj_ts = load_subject_func_timeseries(subj_dir, hemi)
        if subj_ts is None:
            continue
        subj_ts_norm = normalize_subject_timeseries(subj_ts)
        if group_sum is None:
            group_sum = torch.zeros_like(subj_ts_norm)
        group_sum += subj_ts_norm
        n_subjects += 1
        print(f"  {subj_dir.name}: {subj_ts_norm.shape}")

    if group_sum is None or n_subjects == 0:
        print(f"No usable subjects for hemisphere {hemi}")
        return None

    # Final l2 normalization
    group_ts = group_sum / n_subjects
    group_ts = group_ts - group_ts.mean(dim=1, keepdim=True)
    group_ts = group_ts / (group_ts.std(dim=1, keepdim=True) + 1e-10)
    group_norm = torch.norm(group_ts, p=2, dim=1, keepdim=True) + 1e-10
    group_ts = group_ts / group_norm

    out_path = Path(output_dir) / f"group_{hemi}_func_time_series.pt"
    torch.save(group_ts, out_path)
    print(f"Saved group {hemi} time series to {out_path}, shape: {group_ts.shape}, n_subjects: {n_subjects}")
    return group_ts

if __name__ == "__main__":
    subject_base_dir = ""
    output_dir = ""
    Path(output_dir).mkdir(exist_ok=True)
    group_lh = aggregate_group_timeseries(subject_base_dir, output_dir, hemi='L')
    group_rh = aggregate_group_timeseries(subject_base_dir, output_dir, hemi='R')
