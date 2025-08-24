import torch
import torch.nn.functional as F

def feature_masking(x, mask_prob=0.2):
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask

def feature_noise(x, noise_scale=0.1):
    return x + noise_scale * torch.randn_like(x)

def random_edge_dropout(adj, drop_prob=0.1):
    idx = adj._indices()
    val = adj._values()
    mask = torch.rand(val.shape) > drop_prob
    idx_new = idx[:, mask]
    val_new = val[mask]
    return torch.sparse.FloatTensor(idx_new, val_new, adj.shape)

def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0) 
    similarity_matrix = torch.matmul(representations, representations.T)
    mask = torch.eye(2*N, dtype=torch.bool).to(z1.device)
    similarity_matrix.masked_fill_(mask, -9e15)
    positives = torch.cat([torch.diag(similarity_matrix, N), torch.diag(similarity_matrix, -N)], dim=0)
    negatives = similarity_matrix[~mask].view(2*N, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2*N, dtype=torch.long).to(z1.device)
    logits = logits / temperature
    loss = F.cross_entropy(logits, labels)
    return loss
