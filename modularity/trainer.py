import torch
import torch.nn as nn
from model import CombinedModel

class Trainer:
    def __init__(self, opt, n_clusters, adj, deg, norm_adj, gnn_type='GCN'):
        self.model = CombinedModel(opt, n_clusters, adj, deg, gnn_type)
        self.opt = opt
        self.device = torch.device('cuda' if opt['cuda'] else 'cpu')
        self.model.to(self.device)

    def fit(self, features, norm_adj, adj, n_epochs=20):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt['lr'])
        features = features.to(self.device)
        norm_adj = norm_adj.to(self.device)
        adj = adj.to(self.device)
        self.model.train() 
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            _, assignments, loss = self.model(features, norm_adj, adj)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1:3d}/{n_epochs}, Loss: {loss.item():.6f}")
        return assignments.detach().cpu().numpy()

    def cluster_labels(self, assignments):
        # Returns hard cluster assignments for each node
        return assignments.argmax(axis=1)
