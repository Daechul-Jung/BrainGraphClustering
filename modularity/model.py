# model.py
import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modularity.gnn import *
from modularity.dmon import DMoN

def laplacian_smoothness(S, L):
    LS = torch.sparse.mm(L, S)
    return torch.trace(S.T @ LS)

class CombinedModel(nn.Module):
    """
    Two heads:
      - functional head (DMoN_f): uses adjacency 'adj_f' -> modularity term (-Q_f) + collapse_f
      - spatial   head (DMoN_s): NO modularity; only collapse_s
    Total loss adds: + mu_spatial * Tr(C_s^T L_s C_s)
    """
    def __init__(self, opt, n_clusters, norm_adj, adj_f, laplacian_s, mu_spatial=0.0):
        super().__init__() 
        hid = opt['hidden_dim']
        self.encoder_f = GCN(
            in_features=opt['num_feature_f'],  # T
            out_features=n_clusters,
            num_layers=2,
            hidden_features=hid,
            activation=opt.get('activation', 'selu'),
            skip_connection=opt.get('skip_connection', True),
            dropout=opt.get('dropout_rate', 0.0),
        )
        self.encoder_s = GCN(
            in_features=opt['num_feature_s'],  # 3
            out_features=n_clusters,
            num_layers=2,
            hidden_features=hid,
            activation=opt.get('activation', 'selu'),
            skip_connection=opt.get('skip_connection', True),
            dropout=opt.get('dropout_rate', 0.0),
        )

        self.dmon_f = DMoN(
            n_clusters,
            collapse_regularization=opt.get('collapse_regularization', 0.1),
            dropout_rate=opt.get('dropout_rate', 0.0),
            do_unpooling=False
        )
        self.dmon_s = DMoN(
            n_clusters,
            collapse_regularization=opt.get('collapse_regularization', 0.1),
            dropout_rate=opt.get('dropout_rate', 0.0),
            do_unpooling=False
        )

        self.norm_adj = norm_adj      
        self.adj_f = adj_f              
        self.Ls = laplacian_s          
        self.mu_spatial = float(mu_spatial)

    def forward(self, x_f, x_s):
        z_f = self.encoder_f(x_f, self.norm_adj)
        z_s = self.encoder_s(x_s, self.norm_adj)

        _, C_f, _, spec_f, col_f = self.dmon_f(z_f, self.adj_f) 
        _, C_s, _, _,   col_s = self.dmon_s(z_s, adjacency=None) 

        # Spatial smoothness on C_s
        lap_s = x_f.new_tensor(0.0)
        if self.Ls is not None and self.mu_spatial > 0.0:
            LS = torch.sparse.mm(self.Ls, C_s)
            lap_s = torch.trace(C_s.T @ LS)  # Tr(C_s^T L_s C_s)        

        # Final loss
        loss = spec_f + self.dmon_f.collapse_regularization * col_f \
             + self.mu_spatial * lap_s \
             + self.dmon_s.collapse_regularization * col_s

        return (z_f, z_s, C_f, C_s, spec_f, col_f, lap_s, col_s, loss)
