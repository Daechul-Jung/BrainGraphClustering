import torch
import torch.nn as nn
import torch.nn.functional as F

################################## Multi-layer
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation='selu', skip_connection=True, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=True)
        self.skip_connection = skip_connection
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        if activation == 'selu':   
            self.act = nn.SELU()
        elif activation == 'relu': 
            self.act = nn.ReLU()
        elif activation is None:   
            self.act = nn.Identity()
        else: 
            raise ValueError(f"Unsupported activation: {activation}")
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, features, norm_adj):
        h_lin = self.lin(features)             # (V, out)
        h_prop = torch.sparse.mm(norm_adj, h_lin)  # (V, out)

        if self.skip_connection and features.shape[1] == h_lin.shape[1]:
            h = h_prop + h_lin
        else:
            h = h_prop

        return self.act(self.dropout(h))
        h = features.matmul(self.weight)
        Ah = torch.sparse.mm(norm_adj, h)
        h = (h * self.skip_weight.unsqueeze(0) + Ah) if self.skip_connection else Ah
        return self.act(self.dropout(h))

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, activation='selu', skip_connection=True, dropout=0.0):
        super().__init__()
        layers = []
        
        dims = [in_features] + [hidden_features]*(num_layers-1) + [out_features]
        for l in range(num_layers):
            layers.append(GCNLayer(
                in_features=dims[l],
                out_features=dims[l+1],
                activation=activation,
                skip_connection=skip_connection and dims[l]==dims[l+1],
                dropout=dropout
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, norm_adj):
        for layer in self.layers:
            x = layer(x, norm_adj)
        return x