import torch
import numpy as np
import os
import sys
from scipy.special import iv, ive
from gwMRF_generate_components import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utilities.read_avg_mesh import *
from utilities.spgrad_rsfc_gradient import *
from gwMRF_set_params import *
import torch.nn.functional as F
import pygco

class gwMRF:
    def __init__(self, L, avg_mesh, params, hemisphere, border_matrix):
        self.mu = torch.zeros((L, 1200)) # Parameter for time series 
        self.nu = torch.zeros((L, 3)) # Parameter for spatial data
        self.kappa = torch.zeros((L, )) # 
        self.tau = torch.zeros((L, )) + 50000
        self.num_parcels = L
        self.avg_mesh = avg_mesh
        self.border_matrix = border_matrix
        self.params = params
        self.c = 150000
        self.k = torch.tensor(15, dtype=torch.long)
        self.hemi = hemisphere
        self.build_sparse_gradient_matrix()
        
        
    def cal_kappa(self, mu_l, y_l):
        
        M = y_l.shape[1]
        N = y_l.shape[0]
        gamma_l = torch.sum(torch.matmul(y_l, mu_l)) / N
        kappa = ((M-2)*gamma_l) + ((M-1) * gamma_l) / (2*(M-2))
        return kappa
    
    def vmf_logpdf(self, x, mu, kappa):
        
        d = x.shape[1]
        bessel = iv(d/2 - 1, kappa.item())  
        log_C = 0
        
        if d == 1200:
            log_C = ((d/2 - 1) * torch.log(kappa) - (d/2) * torch.log( torch.tensor(2 * torch.pi)) - torch.log(torch.tensor(bessel)))  
            
        elif d == 3:
            log_C = torch.log(kappa) - ( torch.log(torch.tensor( 2 * torch.pi )) + kappa ) 
            
        dots = x @ mu

        return log_C + kappa * dots    
    
    def U_global(self, x, labels):
        
        mask = self.avg_mesh['MARS_label'] != -1
        labels = labels[mask]
        u_global = torch.zeros((len(labels), self.num_parcels))

        for l in range(self.num_parcels):
            
            mask_idx = torch.where(labels == l)[0]
            if mask_idx.numel() == 0:
                
                kappa_l = self.kappa[l]
                mu_l = self.mu[l]
            else:
                y_l = self.x_time[mask_idx]  
                sum_y = y_l.sum(dim=0)
                mu_l  = sum_y / (torch.norm(sum_y, p=2) + 1e-8)
                kappa_l = self.cal_kappa(mu_l, y_l)
            
            if int(kappa_l.item()) == 0:
                kappa_l = torch.tensor(500, dtype=torch.long)
                
            self.kappa[l] = kappa_l
            self.mu[l] = mu_l
            
            u_global_l = self.vmf_logpdf(x[mask], mu_l, kappa_l)
            u_global[:, l] = u_global_l
            
        u_global[u_global == torch.inf] = 1000
        max_likeli = torch.max(u_global, dim = 1, keepdim=True)[0]
        u_global -= max_likeli
        
        u_global[torch.isnan(u_global)] = -300
        u_global[u_global == -torch.inf] = -300
        
        return u_global
    
    def U_spatial(self, x, labels):
        
        mask = self.avg_mesh['MARS_label'] != -1
        labels = labels[mask]
        u_spatial = torch.zeros((len(labels), self.num_parcels))
        
        for l in range(self.num_parcels):
            
            mask_idx = torch.where(labels == l)[0]
            if mask_idx.numel() == 0:
                nu_l = self.nu[l]
            else:
                s_l = x[mask_idx]
                sum_s = s_l.sum(dim = 0)
                nu_l = sum_s / sum_s.norm()
                self.nu[l] = nu_l
                
            u_spatial_l = self.vmf_logpdf(x[mask], nu_l, self.tau[l])
            u_spatial[:, l] = u_spatial_l
            
        max_likeli = torch.max(u_spatial, dim = 1, keepdim=True)[0]
        u_spatial -= max_likeli
        
        return u_spatial
    
    def expo(self, grad):
        return self.c * (torch.exp(-self.k * grad) - torch.exp(-self.k))
    
    def V_grad(self, labels):
        
        labels_c = labels[self._cortex_mask]  

        r, c, w = self._nbr_row, self._nbr_col, self._nbr_w
        diff_mask = (labels_c[r] != labels_c[c]).float()  
        penalties = torch.from_numpy(w) * diff_mask       

        total_pen = penalties.sum()

        P = torch.zeros((self._Nc, self._Nc), dtype=torch.float32)
        P[r, c] = penalties

        return total_pen, P
    
    # def V_grad2(self, labels, local_grad):
    #     v_grad = 0
    #     v_grad_neighbors = torch.zeros((len(labels), len(labels)), dtype= torch.float32)
    #     medial = torch.where(self.avg_mesh['MARS_label'] == -1)[0]
    
    #     for vertex, neighbors in enumerate(self.avg_mesh['vertexNbors']):
    #         for neigh in neighbors:
    #             if neigh not in medial and vertex not in medial:
    #                 if labels[vertex] != labels[neigh]:
    #                     grad = abs(local_grad[vertex] - local_grad[neigh])
    #                     penalty = self.c * self.expo(grad)
    #                     v_grad += penalty
    #                     v_grad_neighbors[vertex, neigh] = penalty
                        
    #     return v_grad, v_grad_neighbors
    
    def loss(self, v_grad, u_global, u_spatial, labels):
        mask = self.avg_mesh['MARS_label'] != -1        
        labels_kept = labels[mask]                          

        unary = -(u_global + u_spatial)             

        idx = torch.arange(unary.size(0))  
        total_unary_cost = unary[idx, labels_kept].sum()

        return total_unary_cost + v_grad
    
    def graphcut_update(self, u_global, u_spatial):
        """
        Returns new label array of shape 
        """
        # if not hasattr(self, '_nbr_w'):
        #     self.build_neighborhood_gradient()
        
        mask = self._cortex_mask
        eps = 1e-8
        N, L = u_global.shape
        
        ######################## VERSION 1 ###########################
        # if u_spatial != None:
        #     u_global = u_global.numpy()
        #     u_spatial = u_spatial.numpy()
            
        #     print(f'u_global before scaling: \n{u_global}')
        #     data_cost = -(u_global * 100000 + eps) / N + eps
            
        #     print(f'data_cost: \n{data_cost}')
        #     pos_cost = -(u_spatial + eps) / N + eps
        #     print(f'pos_cost: {pos_cost}')
        #     unary_cost = data_cost + pos_cost
        #     unary_cost = unary_cost.round().astype(np.int32)
        #     print(f'unary_cost: \n{unary_cost}')
            
        # else :  ### When it is initial time
        #     u_global = u_global.numpy()
        #     unary_cost = -(u_global * 200)
        #     unary_cost = unary_cost.round().astype(np.int32)
        #     print(f'unary_cost: \n{unary_cost}')
        ##############################################################
        
        ######################## VERSION 2 ###########################
        # if u_spatial != None:
        #     u_global = -u_global / 300   #### N previously
        #     print(f'u_global before softmax only with dividing: \n{u_global}')
        #     soft_global = F.softmax(u_global, dim = 1)
        #     soft_global_min, _ = soft_global.min(dim = 1, keepdim = True)
            
        #     u_spatial = -u_spatial / N
        #     soft_spatial = F.softmax(u_spatial, dim = 1)
        #     soft_spatial_min, _ = soft_spatial.min(dim = 1, keepdim = True)
        #     print(f'soft_global: \n{soft_global}\nsoft_spatial: \n{soft_spatial}')
        #     unary_cost = soft_global * (1 / soft_global_min) * 5000 + soft_spatial * (1 / soft_spatial_min) * 5000   ### Originally 5000 for both 
        #     unary_cost = unary_cost.numpy()
        #     unary_cost = unary_cost.round().astype(np.int32)
        #     print(f'unary_cost: \n{unary_cost}')
            
        # else:
        #     u_global = -u_global / 300
        #     print(f'u_global before softmax only with dividing: {u_global}')
        #     soft_global = F.softmax(u_global, dim = 1)
        #     soft_min, _ = soft_global.min(dim = 1, keepdim = True)
        #     print(f'soft global: \n{soft_global} \n soft_min: \n{soft_min}')
        #     unary_cost = soft_global * (1 / soft_min) * 10000
        #     # print(f'Unary cost before round: \n{unary_cost}')
        #     unary_cost = unary_cost.numpy().round().astype(np.int32)
        #     print(f'Unary cost after round: \n{unary_cost}')
            
        #############################################################
        
        ######################## VERSION 3 ###########################
        if u_spatial != None:
            u_global = u_global.numpy()
            u_spatial = u_spatial.numpy()
            
            print(f'u_global before scaling: \n{u_global}')
            data_cost = -(u_global * 100) ## Originally 100
            
            print(f'data_cost: \n{data_cost}')
            pos_cost = -(u_spatial) 
            print(f'pos_cost: \n{pos_cost}')
            unary_cost = (data_cost + pos_cost) * 10 + 100000   ### Originally remove 10 and one 0 from 100000     ////// data + pos + 10000 originally, now scale up 10 times
            #### If values are too high, graph cut does not work at all
            unary_cost = unary_cost.round().astype(np.int32)
            print(f'unary_cost: \n{unary_cost}')
            
        else :  ### When it is initial time
            u_global = u_global.numpy()
            unary_cost = -(u_global * 2000)  ### previous 200, now scale up to 2000
            unary_cost = unary_cost.round().astype(np.int32)
            print(f'unary_cost: \n{unary_cost}')
            
        ##############################################################
        
        smooth = np.ones((L, L), dtype=np.float32)
        scale = self.params['smoothcost'] * 100000.0 / N   ### remove one 0, originally 100000
        np.fill_diagonal(smooth, 0)
        smooth =np.round(smooth * scale).astype(int)
        rows, cols, w = self._nbr_row, self._nbr_col, self._nbr_w        
        scale_factor = 5  ### try 50
        ####  Originally 10 but decreased to 1 for testing. 1 does not work at all, it returns 134.
        ####  Rather than 10, 5 works better. We need to scale all factors according to the number of parcels
        ####  But still returning 385 ~ 390 labels 
        edges = np.column_stack([rows, cols]).astype(np.int32)
        weights = (w * scale_factor).round().astype(np.int32).reshape(-1, 1)
        print(f'weights: \n{weights}')
        edges_full = np.hstack([edges, weights])
        
        edges_full = np.ascontiguousarray(edges_full, dtype = np.int32)
        unary_cost = np.ascontiguousarray(unary_cost, dtype = np.int32)
        smooth = np.ascontiguousarray(smooth, dtype = np.int32)
        n_iter = 10000
        
        print(f'smooth cost: \n{smooth}')
        
        print('Do graph cut')
        labels_c = pygco.cut_from_graph(edges_full, unary_cost, smooth, n_iter, algorithm = 'expansion')
        print('graph cut ended')
        # print(np.unique(labels_c))
        print(f'length of labels after graph cut: {len(labels_c)}')
        full = torch.full_like(self.avg_mesh['MARS_label'], 0, dtype=torch.long)
        full[mask] = torch.from_numpy(labels_c).long()
        
        return full
    
    def build_sparse_gradient_matrix(self):
        MARS_label = self.avg_mesh['MARS_label'].cpu().numpy() if hasattr(self.avg_mesh['MARS_label'], "cpu") else self.avg_mesh['MARS_label']
        cortex_mask = (MARS_label != 0)
        idx_cortex_vertices = np.where(cortex_mask)[0]
        vertex_nbors = self.avg_mesh['vertexNbors']
        n_vertices = len(MARS_label)
        max_nbors = self.border_matrix.shape[0]  # should match

        rows, cols, vals = [], [], []

        for i in range(12, n_vertices):
            nbors = vertex_nbors[i]
            for k in range(min(6, len(nbors))):
                neighbor = nbors[k]
                weight = self.stableE(self.border_matrix[k, i])
                rows.append(i)
                cols.append(neighbor)
                vals.append(weight)

        for i in range(12):
            nbors = vertex_nbors[i]
            for k in range(min(5, len(nbors))):
                neighbor = nbors[k]
                weight = self.stableE(self.border_matrix[k, i])
                rows.append(i)
                cols.append(neighbor)
                vals.append(weight)

        # Only cortex-cortex
        orig_to_cortex = {orig: idx for idx, orig in enumerate(idx_cortex_vertices)}
        cortex_rows, cortex_cols, cortex_vals = [], [], []
        for r, c, v in zip(rows, cols, vals):
            if r in orig_to_cortex and c in orig_to_cortex:
                cortex_rows.append(orig_to_cortex[r])
                cortex_cols.append(orig_to_cortex[c])
                cortex_vals.append(v)

        self._nbr_row = np.array(cortex_rows, dtype=np.int32)
        self._nbr_col = np.array(cortex_cols, dtype=np.int32)
        self._nbr_w   = np.array(cortex_vals, dtype=np.float32)
        print(self._nbr_w)
        self._cortex_mask = cortex_mask
        self._Nc = len(idx_cortex_vertices)
        
    
    def assign_empty_clsuter(self, labels, ):
        low_grad_idx = torch.where(self.local_grad < 0.05)[0]
        empty = []
    
        for i in range(self.num_parcels):
            idx = (labels == i) 
            if sum(idx) == 0:
                empty.append(i) 
                
        assigned_vertices = []
        
        if len(empty) != 0:
            assigned_vertices = np.random.choice(low_grad_idx, size = len(empty), replace=False)
            labels[assigned_vertices] = torch.tensor(empty, dtype=torch.long)
            
        for i in empty:
            self.tau[i] = self.params['start_gamma']
            
        return labels
        

    def initiate_labels(self, x_time, local_grad):
        
        print('initiate labels')
        
        mask = self.avg_mesh['MARS_label'] != -1
        cortex_idx = torch.where(mask)[0]
        cortex_grad = local_grad[mask]
        Nc = cortex_idx.numel()
        
        low_grad_idx = torch.where(cortex_grad < 0.05)[0]
                
        if low_grad_idx.numel() < self.num_parcels:
            low_grad_idx = torch.arange(Nc)
        
        chosen = np.random.choice(low_grad_idx.numpy(), size=self.num_parcels, replace=False)
        seed_in_mask = torch.from_numpy(chosen).long()
        seed_vertices = cortex_idx[seed_in_mask]  # L vertices
        d = x_time.shape[1]
        mu = x_time[seed_vertices] 
        
        kappa = torch.tensor(1800 if 2000 < d < 10000 else (12500 if d >= 10000 else 500), dtype = torch.float32)
        likeli = torch.zeros((len(cortex_idx), self.num_parcels))
        self.mu = mu
        
        for l, mu_l in enumerate(mu):
            likeli[:, l] = self.vmf_logpdf(x = x_time[mask], mu = mu_l, kappa=kappa)
            
        max_likeli = torch.max(likeli, dim = 1, keepdim=True)[0]
        self.kappa = torch.zeros_like(self.kappa, dtype=torch.float32) + kappa
        likeli -= max_likeli
        
        print(f'likelihood in init: {likeli}')
        cortex_labels = self.graphcut_update(likeli, None)  
        print(f'length of unique values of labels in initiate labels: {len(np.unique(cortex_labels))} and {len(cortex_labels)}')
        # self.check_length_labels(cortex_labels)
        
        if len(np.unique(cortex_labels)) < self.num_parcels:
            cortex_labels = self.assign_empty_clsuter(cortex_labels)
        
        return cortex_labels
    
    def check_length_labels(self, labels):
        
        for l in range(self.num_parcels):
            length = len(torch.where(labels == l)[0])
            
            print(f'{l}-th parcels length: {length}')
            
            
    def update_tau(self, labels, ):
        cortex_mask = self.avg_mesh['MARS_label'] != -1
        cortex_vertices = np.count_nonzero([self.avg_mesh['MARS_label'] != -1])
        full_label = torch.zeros(self.avg_mesh['vertices'].shape[0], dtype=torch.long)
        
        full_label[cortex_mask] = torch.tensor(labels[:cortex_vertices], dtype=torch.long)
        lh_ci, _, _, _ = CBIG_gwMRF_generate_components(self.avg_mesh, self.avg_mesh, full_label, full_label)
    
        lh_ci = lh_ci[cortex_mask]

        bin_vector = torch.zeros(self.params['cluster'], dtype=torch.long)
        for l in range(self.params['cluster']):
            idx = (labels == l)
            unique_val = np.unique(lh_ci[idx])
            bin_vector[l] = 1 if len(unique_val) > 1 else 0

        gamma_head = self.tau.clone()
        gamma_head[(bin_vector == 1) & (self.tau == 0)] = 1000
        # gamma_head[(bin_vector == 1) & (gamma != 0)] *= prams['reduce_speed']
        print(f'Tau head in UpdateGamma: {gamma_head}')
        
        return gamma_head
    
    def stableE(self, x):
        x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x
        eps = 1e-9
        x = torch.clamp(x, 0, 1)
        x = self.expo(x)

        condition1 = torch.isinf(x) & (x > 0)
        condition2 = (~torch.isreal(x)) & (torch.real(x) > 0)
        condition3 = torch.isinf(x) & (x < 0)
        condition4 = (~torch.isreal(x)) & (torch.real(x) < 0)
        
        x[condition1] = self.expo(1 - eps)
        x[condition2] = self.expo(1 - eps)
        x[condition3] = self.expo(0 - eps)
        x[condition4] = self.expo(0 + eps)
        
        return x