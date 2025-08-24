import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities.read_avg_mesh import *
from utilities.spgrad_rsfc_gradient import *
import nibabel as nib
from pathlib import Path
from gwMRF_set_params import *
from trainer import *
import argparse
from utilities.create_border import *
from gwMRF_generate_components import *
import torch

from network_clustering import (
    build_group_binary_connectome,
    cluster_parcels_into_networks,
    parcel_to_vertex_network_labels
)


def main():
    
    parser = argparse.ArgumentParser(description="Build data and perform clustering using CBIG gwMRF pipeline.")
    
    parser.add_argument('--input_fullpaths', type=str, required = True,
                        help="Path to file containing full paths for subject's data (e.g., .func.gii files).")
    parser.add_argument('--output_path', type=str, required = True,
                        help="Directory where output will be saved.")
    parser.add_argument('--start_idx', type=int, default=1,
                        help="Start index (also used as seed).")
    parser.add_argument('--end_idx', type=int, default=1,
                        help="End index.")
    parser.add_argument('--num_left_cluster', type=int, default = 400,
                        help="Number of clusters for left hemisphere.")
    parser.add_argument('--num_right_cluster', type=int, default = 400,
                        help="Number of clusters for right hemisphere.")
    parser.add_argument('--smoothcost', type=float, default=5000,
                        help="Smoothness cost in the MRF.")
    parser.add_argument('--num_iterations', type=int, default=7,
                        help="Number of iterations per random initialization.")
    parser.add_argument('--num_runs', type=int, default=2,
                        help="Number of random initializations.")
    parser.add_argument('--start_gamma', type=int, default=50000,
                        help="Starting gamma value.")
    parser.add_argument('--exponential', type=float, default=15.0,
                        help="Exponential parameter.")
    parser.add_argument('--iter_reduce_gamma', type=int, default=300,
                        help="Parameter for reducing gamma per iteration.")
    
    args = parser.parse_args()
    ##### should reset #####
    lh_func_path = Path('/home/daechul/home/cbig/code/data/data/downloaded/100610/hcp1200/rest/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas.L.func.gii')
    rh_func_path = Path('/home/daechul/home/cbig/code/data/data/downloaded/100610/hcp1200/rest/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas.R.func.gii')
    
    ############### SET Parameters ##############
    params = CBIG_gwMRF_set_prams()
    #############################################
    ####### SET AVG_MESH for each hemisphere ######
    # lh_avg_mesh, _ = CBIG_AvgMesh_intersection(params, hemi='lh' ,mesh_name=params['fsaverage'], surf_type='inflated', label='cortex')
    # rh_avg_mesh, _ = CBIG_AvgMesh_intersection(params, hemi='rh' ,mesh_name=params['fsaverage'], surf_type='inflated', label='cortex')
    # lh_time = Path('code/data/data/downloaded/100610/hcp1200/rest/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas.L.func.gii')
    # rh_time = Path('code/data/data/downloaded/100610/hcp1200/rest/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas.R.func.gii')    
    # lh_surf = Path('code/data/data/downloaded/100610/hcp1200/fsaverage_LR32k/100610.L.inflated.32k_fs_LR.surf.gii')
    # rh_surf = Path('code/data/data/downloaded/100610/hcp1200/fsaverage_LR32k/100610.R.inflated.32k_fs_LR.surf.gii')
    # subject = "100610"
    # base_dir = "/home/daechul/home/cbig/code/data/data/downloaded/100610/hcp1200/fsaverage_LR32k"
    # lh_surf_path = f"{base_dir}/{subject}.L.sphere.32k_fs_LR.surf.gii"
    # rh_surf_path = f"{base_dir}/{subject}.R.sphere.32k_fs_LR.surf.gii"
    # lh_midsurf = f"{base_dir}/{subject}.L.midthickness.32k_fs_LR.surf.gii"
    # rh_midsurf = f"{base_dir}/{subject}.R.midthickness.32k_fs_LR.surf.gii"
    # lh_label = f"{base_dir}/{subject}.L.aparc.32k_fs_LR.label.gii"
    # rh_label = f"{base_dir}/{subject}.R.aparc.32k_fs_LR.label.gii"
    # lh_medial_wall = lh_avg_mesh['MARS_label'] == -1
    # rh_medial_wall = rh_avg_mesh['MARS_label'] == -1
    #################################
    
    ########################### Replace here to just loading the gradient and other which are already calculated ############
    PROCESSED_DIR = Path("/media/daechul/My Passport/processed")
    lh_avg_mesh = torch.load(PROCESSED_DIR / "avg_mesh_lh.pt")
    rh_avg_mesh = torch.load(PROCESSED_DIR / "avg_mesh_rh.pt")
    lh_avg_mesh['MARS_label'][lh_avg_mesh['MARS_label'] == 0] = -1
    rh_avg_mesh['MARS_label'][rh_avg_mesh['MARS_label'] == 0] = -1
    
    lh_local_grad = torch.from_numpy(
        np.load(PROCESSED_DIR / "group_mean_lh_edge_density.npy")
    ).float()  
    rh_local_grad = torch.from_numpy(
        np.load(PROCESSED_DIR / "group_mean_rh_edge_density.npy")
    ).float()  

    # lh_full_grad = torch.zeros(len(lh_avg_mesh['MARS_label']), dtype=torch.float32)
    # rh_full_grad = torch.zeros(len(rh_avg_mesh['MARS_label']), dtype=torch.float32)

    # lh_not_medial = torch.where(lh_avg_mesh['MARS_label'] != -1)[0]
    # rh_not_medial = torch.where(rh_avg_mesh['MARS_label'] != -1)[0]

    # lh_full_grad[lh_not_medial] = lh_local_grad
    # rh_full_grad[rh_not_medial] = rh_local_grad

    try:
        lh_grad_matrix = np.load(PROCESSED_DIR / "lh_border_matrix.npy")
        rh_grad_matrix = np.load(PROCESSED_DIR / "rh_border_matrix.npy")
    except FileNotFoundError:
        # fall back to recomputing from the vectors you just loaded
        lh_grad_matrix = gradient_vertices_to_matrix(lh_local_grad.numpy(),
                                                    lh_avg_mesh['vertexNbors'])
        rh_grad_matrix = gradient_vertices_to_matrix(rh_local_grad.numpy(),
                                                    rh_avg_mesh['vertexNbors'])
    ##########################################################################################################################
    trainer = Trainer(num_left_cluster=args.num_left_cluster, 
                      num_right_cluster=args.num_right_cluster,
                      lh_func_path=lh_func_path,
                      rh_func_path=rh_func_path,
                      lh_grad = lh_grad_matrix,
                      rh_grad = rh_grad_matrix,
                      lh_avg_mesh = lh_avg_mesh,
                      rh_avg_mesh = rh_avg_mesh,
                      surf_type='inflated')
    
    params, results = trainer.prepare_clustering()
    lh_vertex_parcels = results['lh_label']         # shape (V_lh,)
    rh_vertex_parcels = results['rh_label']         # shape (V_rh,)
    Lh = int(lh_vertex_parcels[lh_vertex_parcels!=-1].max().item() + 1)
    Rh = int(rh_vertex_parcels[rh_vertex_parcels!=-1].max().item() + 1)
    L = Lh + Rh
    assert L == 400, f"Expected 400 parcels, found {L}"

    # Cortex masks (True where not medial)
    lh_cortex_mask = (lh_vertex_parcels != -1)
    rh_cortex_mask = (rh_vertex_parcels != -1)

   
    subjects_lh = [str(lh_func_path)]
    subjects_rh = [str(rh_func_path)]  

    # 3) Build group-level binarized connectome (Yeo2011-style averaging of top-10% edges)
    group_bin = build_group_binary_connectome(
        subjects_lh_func=subjects_lh,
        subjects_rh_func=subjects_rh,
        lh_labels=lh_vertex_parcels, rh_labels=rh_vertex_parcels,
        lh_cortex_mask=lh_cortex_mask, rh_cortex_mask=rh_cortex_mask,
        L=L, top_p=0.10
    )

    # 4) Cluster parcels into 7 and 17 networks with vMF mixture
    labels_7  = cluster_parcels_into_networks(group_bin, K=7,  method="vmf")   # shape (L,)
    labels_17 = cluster_parcels_into_networks(group_bin, K=17, method="vmf")   # shape (L,)

    # 5) Map parcel-level network IDs back to vertex space for visualization/saving
    lh_net7,  rh_net7  = parcel_to_vertex_network_labels(
        lh_vertex_parcels, rh_vertex_parcels, labels_7,  lh_cortex_mask, rh_cortex_mask
    )
    lh_net17, rh_net17 = parcel_to_vertex_network_labels(
        lh_vertex_parcels, rh_vertex_parcels, labels_17, lh_cortex_mask, rh_cortex_mask
    )

    # 6) Save as .label.gii (or .func.gii) for easy viewing
    import nibabel as nib
    def save_label_gii(arr: torch.Tensor, out_path: str):
        dl = nib.gifti.GiftiLabelTable()
        da = nib.gifti.GiftiDataArray(arr.cpu().numpy().astype(np.int32), intent='NIFTI_INTENT_LABEL')
        g = nib.gifti.GiftiImage(darrays=[da], labeltable=dl)
        nib.save(g, out_path)

    outdir = os.path.join(args.output_path, "networks")
    os.makedirs(outdir, exist_ok=True)
    save_label_gii(lh_net7,  os.path.join(outdir, "lh.networks7.label.gii"))
    save_label_gii(rh_net7,  os.path.join(outdir, "rh.networks7.label.gii"))
    save_label_gii(lh_net17, os.path.join(outdir, "lh.networks17.label.gii"))
    save_label_gii(rh_net17, os.path.join(outdir, "rh.networks17.label.gii"))

    # 7) (optional) also save parcel-level CSVs for analysis
    np.savetxt(os.path.join(outdir, "parcel_networks7.csv"),  labels_7,  fmt="%d", delimiter=",")
    np.savetxt(os.path.join(outdir, "parcel_networks17.csv"), labels_17, fmt="%d", delimiter=",")
    
if __name__ == "__main__":
    main()