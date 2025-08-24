import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DMoN(nn.Module):
    def __init__(self, n_clusters, collapse_regularization=0.1, dropout_rate=0.0, do_unpooling=False):
        super().__init__()
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate
        self.do_unpooling = do_unpooling

        self.transform = nn.Sequential(
            nn.LazyLinear(n_clusters),
            nn.Dropout(dropout_rate)
        )
        self._init = False

    def forward(self, features, adjacency):
        
        if not self._init:
            lin = self.transform[0]
            with torch.no_grad():
                _ = lin(features)
                nn.init.orthogonal_(lin.weight)
                lin.bias.zero_()
            self._init = True

        S = F.softmax(self.transform(features), dim=1)
        sizes = S.sum(dim=0)
        Snorm = S / sizes.unsqueeze(0)
        
        deg = torch.sparse.sum(adjacency, dim=0).to_dense().unsqueeze(1)
        m2 = deg.sum() 
        AS = torch.sparse.mm(adjacency, S)
        Gp = AS.T @ S
        dl = S.T @ deg
        dr = deg.T @ S
        null = (dl @ dr) / m2
        spec = -torch.trace(Gp - null) / m2
        col = torch.norm(sizes) / features.shape[0] * math.sqrt(self.n_clusters) - 1.0
        total_loss = spec + self.collapse_regularization * col
        Hp = (Snorm.T @ features)
        Hp = F.selu(Hp)
        
        if self.do_unpooling:
            Hp = Snorm @ Hp
        return Hp, S, total_loss