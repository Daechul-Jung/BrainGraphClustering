import torch
import numpy as np
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities.read_avg_mesh import *
from utilities.spgrad_rsfc_gradient import *
import nibabel as nib
from model import *
from gwMRF_set_params import *


class Trainer:
    def __init__(self, num_left_cluster, num_right_cluster, 
                 lh_func, rh_func, 
                 lh_grad, rh_grad, 
                 lh_avg_mesh, rh_avg_mesh,
                 surf_type):
        
        torch.manual_seed(5489)
        np.random.seed(5489)
        
        # Should be defined first
        self.lh_avg_mesh = lh_avg_mesh
        self.rh_avg_mesh = rh_avg_mesh
        
        ### Based on avg_mesh, define 
        self.lh_model = None
        self.rh_model = None
       
        self.lh_func = lh_func
        self.rh_func = rh_func
        self.num_left_parcel = num_left_cluster
        self.num_right_parcel = num_right_cluster
        self.lh_grad = lh_grad
        self.rh_grad = rh_grad
        self.surf_type = surf_type if surf_type is not None else 'inflated'
        
        self.params = CBIG_gwMRF_set_prams()
        
    ##############################  Basic Preparation for parameters and models and data #############################
    
    def set_models(self, ):
        """
        Defining models for each (left and right) hemisphere
        """
        self.lh_model = gwMRF(self.num_left_parcel, self.lh_avg_mesh, self.params, 'lh', self.lh_grad)
        self.rh_model = gwMRF(self.num_right_parcel, self.rh_avg_mesh, self.params, 'rh', self.rh_grad)
        

   
    def get_spatial_data(self, hemi):
        """
        Getting Spatial information which is normalized 
        
        Args:
            hemi (str): left or right hemisphere

        Returns:
            spatial_data (torch.tensor): Normalized spatial information (x, y, z coordinate) (V, 3)
        """
        if hemi == 'lh':
            spatial_data = self.lh_avg_mesh['vertices'][self.lh_avg_mesh['mesh_data']]
            spatial_data /= torch.sqrt(torch.sum(spatial_data ** 2, dim=1, keepdim=True))
            
        elif hemi == 'rh':
            spatial_data = self.rh_avg_mesh['vertices'][self.rh_avg_mesh['mesh_data']]
            spatial_data /= torch.sqrt(torch.sum(spatial_data ** 2, dim=1, keepdim=True))
            
        return spatial_data
        
    # def divide_local_grad(self):
    #     """
    #     Divide the local gradient for left and right hemisphere and store it as parameters
    #     """
    #     left_right_bar = len(torch.where(self.lh_avg_mesh['MARS_label'] != -1)[0])
    #     lh_grad = torch.tensor(nib.load(self.grad_path).get_fdata()[0][: left_right_bar], dtype = torch.float32)
    #     rh_grad = torch.tensor(nib.load(self.grad_path).get_fdata()[0][left_right_bar :], dtype = torch.float32)
        
    #     lh_not_medial = self.lh_avg_mesh['MARS_label'] != -1
    #     rh_not_medial = self.rh_avg_mesh['MARS_label'] != -1
        
    #     lh_full_grad = torch.tensor(np.zeros(len(self.lh_avg_mesh['MARS_label']), dtype = np.float32), dtype = torch.float32)
    #     rh_full_grad = torch.tensor(np.zeros(len(self.rh_avg_mesh['MARS_label']), dtype = np.float32), dtype = torch.float32)
        
    #     lh_full_grad[lh_not_medial] = lh_grad
    #     rh_full_grad[rh_not_medial] = rh_grad
                
    #     self.lh_local_grad = lh_full_grad
    #     self.rh_local_grad = rh_full_grad
        
    def set_all(self, ):
        self.set_models()
       
    #########################################################################################################
        
    def prepare_clustering(self):
        """
        Same as CBIG_gwMRF_graph_cut_clustering. From this part, Setting and preparing the clustering. In this part, we can choose how many subjects would be in the clustering performance 
        and save as json file for later 
        """
        log_file, start_idx, end_idx = self.params['fileID'], self.params['start_index'], self.params['start_index'] + self.params['runs'] - 1
        results = 0
        log_file.write(f'will iterate from initialization {start_idx} to initialization {end_idx}')
        for i in range(self.params['start_index'], self.params['start_index'] + self.params['runs']):
            self.params['seed'] = i

            energy_file = os.path.join(
                self.params['output_folder'], f"{self.params['output_name']}_seed_{i}_Energy.py"
            )
            result_file = os.path.join(
                self.params['output_folder'], f"{self.params['output_name']}_seed_{i}.py"
            )

            if os.path.isfile(energy_file):
                log_file.write(f"File for seed {i} already exists, skipping!\n")
                continue
            else:
                results = self.clustering_iter_split()

                self.save_to_json_file(result_file, {"results": results, "prams": self.params})

                if "E" in results:
                    energy_data = {"Energy": results["E"]}
                    self.save_to_json_file(energy_file, energy_data, variable_name="Energy")
                else:
                    log_file.write(f"Warning: 'results' does not have key 'E'\n")
                    
        log_file.close()  
        
        return self.params, results
    
    def clustering_iter_split(self):
        """
        corresponding to CBIG_gwMRF_graph_cut_clustering_iter_split and perform clustering for left and right hemisphere
        """
        self.set_all()   #### Set models, avg_mesh(for each hemisphere) and local gradient(for each hemisphere)
        
        l1 = torch.where(self.lh_avg_mesh['MARS_label'] != -1)[0]
        
        r1 = torch.where(self.rh_avg_mesh['MARS_label'] != -1)[0]
        
        self.params['dim'] -= 1
        
        if self.params['separate_hemispheres'] == 1 or self.params['local_concentration'] > 0.00:
            if self.params['skip_left'] == 0:
                print(f"Processing left hemisphere: {self.params['lh_avg_file']}")
                if self.params['pca'] == 0:
                    lh_time = self.lh_func   #### Should be time series 
                    if lh_time.shape[0] > len(l1):
                        lh_time = lh_time[l1, :]
                else:
                    lh_time = self.lh_func
                    lh_spatial = self.get_spatial_data('lh')

                print("Computing parameters in split brains")

                self.params['cluster'] = self.params['left_cluster']

                if self.lh_model.params['pca'] == 1:
                    loss, results_lh = self.clustering_wrapper(hemi = 'lh', x_time = lh_time, x_spatial = lh_spatial)
                
                print(results_lh['full_label'])
                results['lh_label'] = results_lh['full_label']
                results['lh_final_likeli'] = results_lh['final_likeli']
                results['lh_likeli_pos'] = results_lh['likeli_pos']
            
            else:
                results = ...
            # Right Hemisphere
            print('\n right hemisphere')
            
        if self.params['separate_hemispheres'] == 1 or self.params['local_concentration'] > 0.00:
            if self.params['skip_right'] == 0:
                print(f"Processing right hemisphere: {self.params['rh_avg_file']}")
                if self.params['pca'] == 0:
                    rh_time = self.rh_func
                    if rh_time.shape[0] > len(r1):
                        rh_time = rh_time[r1, :]
                else: 
                    rh_time = self.rh_func
                    rh_spatial = self.get_spatial_data('rh')
                    
                if self.rh_model.params['pca'] == 1:
                    loss, result_rh = self.clustering_wrapper(hemi='rh', x_time = rh_time, x_spatial = rh_spatial)
                    
                results['rh_label'] = results_rh['full_label']
                results['rh_final_likeli'] = results_rh['final_likeli']
                results['rh_likeli_pos'] = results_rh['likeli_pos']
            else:
                results_rh = ...
                
            results['D'] = results_rh['D'] + results_lh['D']
            results['S'] = results_rh['S'] + results_lh['S']
            results['E'] = results_rh['E'] + results_lh['E']
            results['UnormalizedE'] = results_rh['UnormalizedE'] + results['UnormalizedE']
            results['gamma'] = np.concatenate([results['gamma'], results_rh['gamma']])
            results['kappa'] = np.concatenate([results['kappa'], results_rh['kappa']])
            
        return results
    
    def clustering_wrapper(self, hemi, x_time, x_spatial):
        """
        Main clustering same as new_kappa_prod in original. In this function, we do all MAP1, 2, 3. MAP3 for initializing labels, then MAP1 for updating kappa and mu till converge and finally MAP2 for updating tau till zero or converge (Not theoretically gaurantee) 
        Args:
            
        """
        
        if hemi == 'lh':
            model = self.lh_model
            local_grad = self.lh_grad
        else:
            model = self.rh_model
            local_grad = self.rh_grad
        cortex_vertices = np.count_nonzero([model.avg_mesh['MARS_label'] != -1])
        # if not model.params['potts']:
        #     if hemi == 'lh':
        #         model.build_neighborhood_gradient(self.lh_grad)
        #     elif hemi == 'rh':
        #         model.build_neighborhood_gradient(self.rh_grad)
        idx_cortex_vertices = torch.where(model.avg_mesh['MARS_label'] != -1)[0]
        model.params['fileID'].write("some log message\n")
        ###################### MAP 3 #############################
        labels = model.initiate_labels(x_time, local_grad)
        initial_label = labels.clone()
        for i in range(0, 5, 2):
            for j in range(1000):
                tau_head = model.update_tau(labels)
                if (j > 1) and torch.equal(model.tau, tau_head):
                    break
                else:
                    model.tau = tau_head
                    model.params['graphCutIterations'] = 10 ** (i + 2)
                    labels, energy, stats = self.clustering(model, labels, x_time, x_spatial)
                    model.tau = results['tau']
        results = {
        'initial_full_label': torch.zeros(model.avg_mesh['vertices'].shape[0], dtype=torch.long)
        }
        results['initial_full_label'][model.avg_mesh['MARS_label'] != -1] = initial_label[
            :cortex_vertices
        ]
        #################### End of MAP3 #########################
        
        ###################### MAP 2 #############################
        # results['initial_full_label'] = torch.zeros(model.avg_mesh['vertices'].shape[0], dtype=torch.long)
        # results['initial_full_label'][model.avg_mesh['MARS_label'] != -1] = torch.tensor(initial_label[:cortex_vertices], dtype=torch.long)
        
        if model.params.get('reduce_gamma', 1) == 1:
            labels, energy, stats = self.clustering_reduce_tau(model, labels, x_time, x_spatial)
        else:
            # compute once more to fill energy/stats
            labels, energy, stats = self.clustering(model, labels, x_time, x_spatial)

        # collect final
        results.update({
            'full_label': labels,
            'D': energy['D'],
            'S': energy['S'],
            'UnormalizedE': energy['E'],
            'final_likeli': stats.get('final_likeli', None),
            'kappa': stats.get('kappa', None),
            'E': energy['E'],
            'gamma': stats.get('tau', None),
            'likeli_pos': None,   # optional if you want to store spatial-only term
            'tau': stats.get('tau', None),
        })
        likelihood = stats.get('likelihood_mean', None)
            
        return likelihood, results 
    
    def clustering(self, model, labels, x_time, x_spatial):
        """
        This function is the same as graph_cut_clutsering_split_standard in the original function. In this function, we will compute
        u_global, u_spatial, and v_grad and then finally do graph cut for getting labels. 

        Args:
            model (gwMRF): gwMRF model for specific(left or right) hemisphere
            labels (torch.tensor): labels for every vertex
        """
        
        curr_loss = float('inf')
        pr = model.params
        cortex_mask = (model.avg_mesh['MARS_label'] != -1)
        Nc = int(cortex_mask.sum())
        term_pct = float(pr.get('termination', 1.0))      # percent (like MATLAB)
        max_iters = int(pr.get('iterations', 10))
        
        
        def energy_of(lbl, ug, us):
            # data (unary) term at assigned labels
            unary = -(ug + us)  # negative log-lik proxy, row-shifted ok
            keep = lbl[cortex_mask].long()
            idx = torch.arange(Nc, device=unary.device)
            D = unary[idx, keep].sum().item()
            # smoothness term via gradient-weighted pairwise
            S, _ = model.V_grad(lbl)
            S = float(S)
            return D, S, (D + S)
        prev_E = float('inf')
        likelihood_mean = None
        final_likeli_vec = None
        
        print('till here, it works')
        for j in range(model.params['iterations']):
            prev_loss = curr_loss
            u_global = model.U_global(x_time, labels)
            u_spatial = model.U_spatial(x_spatial, labels)
            
            v_grad, neighbors = model.V_grad(labels)
            labels = model.graphcut_update(u_global, u_spatial)
            curr_loss = model.loss(v_grad, u_global, u_spatial, labels)
            
            D, S, E = energy_of(labels, u_global, u_spatial)

            # simple likelihood stat (mean assigned row values before the minus)
            keep = labels[cortex_mask].long()
            idx = torch.arange(Nc, device=u_global.device)
            final_likeli_vec = (u_global[idx, keep]).detach().cpu().numpy()
            likelihood_mean = float((u_global[idx, keep] + u_spatial[idx, keep]).mean().item())

            # early stop
            improv = (prev_E / E - 1.0) * 100.0 if np.isfinite(prev_E) and E > 0 else 0.0
            labels = labels
            if abs(improv) < term_pct:
                break
            prev_E = E

        energy = {'D': D, 'S': S, 'E': E}
        stats  = {
            'likelihood_mean': likelihood_mean,
            'final_likeli': final_likeli_vec,
            'tau': model.tau.clone(),
            'kappa': model.kappa.clone(),
        }
        return labels, energy, stats
            
    
    def clustering_reduce_tau(self, model, labels, x_time, x_spatial):
        """
        This function is the same as graph_cut_clutsering_split_reduce in the original function. In this function, we will compute
        u_global, u_spatial, and v_grad and then finally do graph cut for getting labels.
        
        Args:
            model (gwMRF): gwMRF model for each hemisphere 
            labels (torch.tensor): _description_
        """
        pr = model.params
        reduce_speed = int(pr.get('reduce_speed', 5))
        iters = int(pr.get('iter_reduce_gamma', 300))

        best_labels, best_energy, best_stats = labels, None, None

        for _k in range(iters):
            tau_prev = model.tau.clone()
            model.tau = model.tau / reduce_speed
            model.tau[model.tau <= 1000] = 0 

            # 1) one MAP1 pass at current tau
            labels, energy, stats = self.clustering(model, labels, x_time, x_spatial)

            # 2) refine tau by connectivity; increase graph-cut iterations
            for i in range(0, 5, 2):
                for _j in range(1000):
                    tau_head = model.update_tau(labels)
                    if torch.equal(model.tau, tau_head):
                        break
                    model.tau = tau_head
                    pr['graphCutIterations'] = 10 ** (i + 2)
                    labels, energy, stats = self.clustering(model, labels, x_time, x_spatial)

            # stop if mean(tau) didn't reduce
            if torch.mean(model.tau) >= torch.mean(tau_prev):
                model.tau = tau_prev
                break

            best_labels, best_energy, best_stats = labels, energy, stats

        return best_labels, (best_energy or energy), (best_stats or stats)
            
def save_intermediate(self):
    """
    Save a human-readable snapshot of the current trainer/model state.
    Creates a timestamped folder under <output_folder>/intermediate with:
      - summary.json (settings + tau/kappa stats)
      - README.txt (plain-text summary)
      - *_tau.csv and *_kappa.csv (one value per row)
      - optional lh.labels.gii / rh.labels.gii if self._last_labels_* present
    Returns: path (str) to the created snapshot directory.
    """
    import time, csv
    from pathlib import Path

    out_root = Path(self.params.get('output_folder', './outputs')) / "intermediate"
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snap_dir = out_root / f"{stamp}"
    snap_dir.mkdir(parents=True, exist_ok=False)

    def _arr(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _stats(arr: np.ndarray):
        if arr is None:
            return None
        return {
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
            "mean": float(np.nanmean(arr)) if arr.size else None,
            "std": float(np.nanstd(arr)) if arr.size else None,
            "n": int(arr.size),
        }

    def _dump_csv(path, header, arr):
        if arr is None:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([header])
            for v in arr.tolist():
                w.writerow([v])

    def _hemi_block(name: str, mdl):
        if mdl is None:
            return {}
        tau = _arr(mdl.tau)
        kappa = _arr(mdl.kappa)
        _dump_csv(snap_dir / f"{name}_tau.csv", "tau", tau)
        _dump_csv(snap_dir / f"{name}_kappa.csv", "kappa", kappa)
        return {
            "hemi": getattr(mdl, "hemi", name),
            "L": int(getattr(mdl, "num_parcels", 0)),
            "c": int(getattr(mdl, "c", 0)) if hasattr(mdl, "c") else None,
            "k": int(mdl.k.item()) if hasattr(mdl, "k") and torch.is_tensor(mdl.k) else getattr(mdl, "k", None),
            "tau_stats": _stats(tau),
            "kappa_stats": _stats(kappa),
        }

    summary = {
        "timestamp": stamp,
        "output_name": self.params.get("output_name", ""),
        "settings": {
            "iterations": int(self.params.get("iterations", 0)),
            "smoothcost": float(self.params.get("smoothcost", 0.0)),
            "reduce_gamma": int(self.params.get("reduce_gamma", 0)),
            "iter_reduce_gamma": int(self.params.get("iter_reduce_gamma", 0)),
            "reduce_speed": int(self.params.get("reduce_speed", 5)),
            "graphCutIterations": int(self.params.get("graphCutIterations", 0)),
        },
        "left": _hemi_block("lh", getattr(self, "lh_model", None)),
        "right": _hemi_block("rh", getattr(self, "rh_model", None)),
    }

    # Optionally save intermediate label maps if caller stored them:
    try:
        import nibabel as nib
        def _save_label_gii(vec, out_path):
            arr = np.asarray(vec, dtype=np.int32)
            da = nib.gifti.GiftiDataArray(arr, intent='NIFTI_INTENT_LABEL')
            nib.save(nib.gifti.GiftiImage(darrays=[da]), str(out_path))

        if hasattr(self, "_last_labels_lh") and self._last_labels_lh is not None:
            _save_label_gii(self._last_labels_lh, snap_dir / "lh.labels.gii")
            summary["left"]["labels_saved"] = "lh.labels.gii"
        if hasattr(self, "_last_labels_rh") and self._last_labels_rh is not None:
            _save_label_gii(self._last_labels_rh, snap_dir / "rh.labels.gii")
            summary["right"]["labels_saved"] = "rh.labels.gii"
    except Exception as e:
        summary["label_save_error"] = str(e)

    # Write JSON (human-readable) and a short README.txt
    with open(snap_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(snap_dir / "README.txt", "w") as f:
        f.write("gwMRF intermediate snapshot\n")
        f.write(f"time: {stamp}\n")
        f.write(f"output_name: {summary['output_name']}\n\n")
        f.write("[settings]\n")
        for k, v in summary["settings"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n[left]\n")
        for k, v in summary["left"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n[right]\n")
        for k, v in summary["right"].items():
            f.write(f"  {k}: {v}\n")

    return str(snap_dir)