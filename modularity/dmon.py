# dmon.py
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

    def forward(self, features, adjacency=None):
        # Lazy init
        if not self._init:
            lin = self.transform[0]
            with torch.no_grad():
                _ = lin(features)
                nn.init.orthogonal_(lin.weight)
                lin.bias.zero_()
            self._init = True

        S = F.softmax(self.transform(features), dim=1)     # (V,K)
        sizes = S.sum(dim=0)                               # (K,)
        Snorm = S / (sizes.unsqueeze(0) + 1e-8)

        # Collapse/balance (same form you used)
        collapse = torch.norm(sizes) / features.shape[0] * math.sqrt(self.n_clusters) - 1.0

        # If adjacency is provided, compute -Q (modularity loss); else 0
        spec = torch.tensor(0.0, device=features.device)
        if adjacency is not None:
            deg = torch.sparse.sum(adjacency, dim=0).to_dense().unsqueeze(1)  # (V,1)
            m2 = deg.sum() + 1e-8
            AS = torch.sparse.mm(adjacency, S)
            Gp = AS.T @ S
            dl = S.T @ deg
            dr = deg.T @ S
            null = (dl @ dr) / m2
            spec = -torch.trace(Gp - null) / m2  # -Q(C)

        # DMoN's own total (if you need it), but we'll build our final loss in model.py
        total = spec + self.collapse_regularization * collapse

        # pooled (optional)
        Hp = (Snorm.T @ features)
        Hp = F.selu(Hp)
        if self.do_unpooling:
            Hp = Snorm @ Hp

        return Hp, S, total, spec, collapse
