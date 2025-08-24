import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from gnn import *
from dmon import *


class CombinedModel(torch.nn.Module):
    def __init__(self, opt, n_clusters, norm_adj, adj):
        super().__init__()
        self.encoder = GCN(
            in_features=opt['num_feature'],
            out_features=opt['hidden_dim'],
            activation=opt.get('activation', 'selu'),
            skip_connection=opt.get('skip_connection', True)
        )
        self.dmon = DMoN(
            n_clusters,
            collapse_regularization=opt['collapse_regularization'],
            dropout_rate=opt['dropout_rate'],
            do_unpooling=False
        )
        self.norm_adj = norm_adj
        self.adj = adj

    def forward(self, x):
        emb = self.encoder(x, self.norm_adj)
        pooled, assignments, dmon_loss = self.dmon(emb, self.adj)
        return emb, pooled, assignments, dmon_loss
