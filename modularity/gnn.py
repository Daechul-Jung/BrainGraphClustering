import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_features, out_features, activation='selu', skip_connection=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.skip_connection = skip_connection

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        if self.skip_connection:
            self.skip_weight = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_buffer('skip_weight', torch.zeros(out_features))

        if activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        if self.skip_connection:
            nn.init.ones_(self.skip_weight)

    def forward(self, features, norm_adj):
        h = features.matmul(self.weight)
        Ah = torch.sparse.mm(norm_adj, h)
        if self.skip_connection:
            h = h * self.skip_weight.unsqueeze(0) + Ah
        else:
            h = Ah
        h = h + self.bias.unsqueeze(0)
        return self.activation(h)
