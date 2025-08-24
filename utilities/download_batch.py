import subprocess
from pathlib import Path
import shutil

EXCLUDES = [
    "--exclude", "*/MNINonLinear/*",
    "--exclude", "*/T1w/*",
    "--exclude", "*/rest/*_MSMAll*.dtseries.nii",
    "--exclude", "*/rest/*_hp2000_clean.dtseries.nii",
    "--exclude", "*/rest/*_hp_preclean.dtseries.nii",
    "--exclude", "*/rest/*_hp*.dtseries.nii",
    "--exclude", "*/rest/*_Atlas_MSMAll*.dtseries.nii",
    # keep only the vanilla Atlas.dtseries.nii
]

RSYNC_FILTERS = [
    # start with nothing
    "--exclude", "*",
    # include subject root
    "--include", "*/",
    # include HCP structure
    "--include", "hcp1200/",
    "--include", "hcp1200/fsaverage_LR32k/",
    "--include", "hcp1200/rest/",
    # include necessary fsaverage_LR32k surfaces and labels
    "--include", "hcp1200/fsaverage_LR32k/*sphere*.surf.gii",
    "--include", "hcp1200/fsaverage_LR32k/*midthickness*.surf.gii",
    "--include", "hcp1200/fsaverage_LR32k/*aparc*.label.gii",
    # include only the vanilla Atlas CIFTI per run
    "--include", "hcp1200/rest/*/*_Atlas.dtseries.nii",
    # everything else excluded
    "--exclude", "*",
]

def _has_free_space(path: Path, min_free_gb: float) -> bool:
    total, used, free = shutil.disk_usage(str(path))
    return free >= min_free_gb * (1024 ** 3)


def download_batch(subject_list, gpu_data_root, local_data_root, gpu_username, gpu_host, min_free_gb = 5):
    """
    Copy HCP subject directories from GPU server to local disk.
    Args:
        subject_list (list): Subject IDs
        gpu_data_root (str): Path on GPU server, e.g., '/banana/daechul/data/downloaded'
        local_data_root (str or Path): Path on local desktop, e.g., '/media/daechul/My Passport/data/downloaded'
        gpu_username (str): Username on GPU server
        gpu_host (str): Host/IP of GPU server
    """
    
    local_root = Path(local_data_root)
    local_root.mkdir(parents=True, exist_ok=True)
    
    if not _has_free_space(local_root, min_free_gb):
        raise RuntimeError(f"Insufficient free space in {local_root} (< {min_free_gb} GB). Free some space and retry.")
    
    base_cmd = ["rsync", "-avz", "--progress",
                "--ignore-existing",            
                "--partial", "--partial-dir=.rsync-partial", 
                "--prune-empty-dirs"]     
    base_cmd = [
        "rsync", "-avz", "--progress",
        "--ignore-existing",
        "--partial", "--partial-dir=.rsync-partial",
        "--prune-empty-dirs",
        *RSYNC_FILTERS,
    ]

    for subject in subject_list:
        remote_path = f"{gpu_username}@{gpu_host}:{gpu_data_root}/{subject}/"
        print(" ".join(base_cmd + [remote_path, str(local_root)]))
        try:
            subprocess.run(base_cmd + [remote_path, str(local_root)], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"[rsync] failed for subject {subject}. Likely disk is full or network hiccup. "
                f"Free space or retry this subject."
            ) from e
            
    # for subject in subject_list: