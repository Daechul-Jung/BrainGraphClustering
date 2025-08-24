import argparse
from loader import (load_vertex_nbors_from_txt, load_grad_from_npy,
                    load_time_series_from_npy, load_coords_from_npy, load_hcp_subject)
from utils import *
from model  import *
from trainer import Trainer

if __name__ == "__main__":
    ##### Loading functional time series and graph structure ######
    vertex_nbors = load_vertex_nbors_from_txt('lh_neighbors.txt')
    grad = load_grad_from_npy('lh_grad.npy')
    time_series = load_time_series_from_npy('lh_timeseries.npy')
    coords = load_coords_from_npy('lh_coords.npy')
    features, adj, norm_adj, deg = load_hcp_subject(vertex_nbors, grad, time_series, coords,
                                                    n_neighbors=6, pca_dim=64, pos_enc_dim=8)
    ###### 
    opt = {
        'num_feature': features.shape[1],
        'hidden_dim': 64,
        'collapse_regularization': 0.1,
        'dropout_rate': 0.1,
        'activation': 'selu',
        'skip_connection': True
    }
    n_clusters = 400 ### or try 7
    lambda_mod = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, adj, norm_adj = features.to(device), adj.to(device), norm_adj.to(device)

    model = CombinedModel(opt, n_clusters, norm_adj, adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------- Training -------
    for epoch in range(1, 201):
        model.train()
        # Data Augmentation
        x1 = feature_masking(features, mask_prob=0.2)
        x1 = feature_noise(x1, noise_scale=0.1)
        x2 = feature_masking(features, mask_prob=0.2)
        x2 = feature_noise(x2, noise_scale=0.1)
        # Forward passes (same normalized adjacency)
        z1, _, _, _ = model(x1)
        z2, _, _, _ = model(x2)
        contrast_loss = nt_xent_loss(z1, z2, temperature=0.5)
        # Clustering and modularity loss (use one view)
        _, _, _, modularity_loss = model(x1)
        # Combine losses
        total_loss = contrast_loss + lambda_mod * modularity_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Total: {total_loss.item():.4f} | Contrast: {contrast_loss.item():.4f} | Modularity: {modularity_loss.item():.4f}")