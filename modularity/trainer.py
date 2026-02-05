import torch
import torch.nn as nn
import os, sys
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modularity.model import CombinedModel
from modularity.utils import print_model_summary, save_checkpoint, ensure_dir


class Trainer:
    def __init__(self, device, opt, n_clusters, norm_adj, adj_f, L_s, mu_spatial=0.0):
        self.device = device
        self.model = CombinedModel(
            opt=opt,
            n_clusters=n_clusters,
            norm_adj=norm_adj.to(self.device),
            adj_f=adj_f.to(self.device),
            laplacian_s=L_s.to(self.device),
            mu_spatial=mu_spatial
        ).to(self.device)
        self.opt = opt
        
        
    def fit(self,
            X_f,                 # functional features 
            X_s,                 # spatial features 
            n_epochs: int = 20,
            lr: float = None,
            ckpt_dir: str = "./checkpoints",
            save_every: int = 0,              
            save_epochs: tuple = (),          
            save_optimizer: bool = True,
            plot_path: str = "./training_loss.png"):

        X_f = X_f.to(self.device)
        X_s = X_s.to(self.device)

        # Optimizer
        lr = lr if lr is not None else self.opt['lr']
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(X_f, X_s)   # initializes lazy params
        self.model.train()
        lr = lr if lr is not None else self.opt['lr']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print_model_summary(self.model, header="CombinedModel Summary (after init)")
        loss_history = []

        # make sure save dir exists
        ensure_dir(ckpt_dir)
        is_main = (int(os.environ.get("RANK", "0")) == 0 or int(os.environ.get("LOCAL_RANK", "0")) == 0)
        pbar = tqdm(range(1, n_epochs + 1), total=n_epochs, disable=not is_main, dynamic_ncols=True,
                    leave=True,                    # keep the final bar
                    mininterval=0.2,               # throttle refresh
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
        for epoch in pbar:
            optimizer.zero_grad()
            zf, zs, C_f, C_s, spec_f, col_f, lap_s, col_s, loss = self.model(X_f, X_s)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            pbar.set_description(f"Epoch {epoch}/{n_epochs}")
            pbar.set_postfix({
                "loss": f"{loss.item():.4g}",
                "-Q_f": f"{spec_f.item():.4g}",
                "col_f": f"{col_f.item():.4g}",
                "lap_s": f"{lap_s.item():.4g}",
                "col_s": f"{col_s.item():.4g}"
            })

            # Periodic print (optional; tqdm already shows progress)
            if epoch % 10 == 0 or epoch == 1:
                pbar.write(
                    f"Epoch {epoch:3d}/{n_epochs} | "
                    f"loss {loss.item():.6f} | "
                    f"-Q_f {spec_f.item():.6f} | "
                    f"col_f {col_f.item():.6f} | "
                    f"lap_s {lap_s.item():.6f} | "
                    f"col_s {col_s.item():.6f}"
                )

            should_save_periodic = (save_every > 0 and epoch % save_every == 0)
            should_save_specific = (epoch in set(save_epochs))

            if should_save_periodic or should_save_specific:
                path = save_checkpoint(
                    model=self.model,
                    optimizer=optimizer if save_optimizer else None,
                    epoch=epoch,
                    save_dir=ckpt_dir,
                    tag=None,
                    extra={"loss": loss.item()}
                )
                print(f"[CKPT] Saved: {path}")

        if 0 not in set(save_epochs):
            path = save_checkpoint(
                model=self.model,
                optimizer=optimizer if save_optimizer else None,
                epoch=n_epochs,
                save_dir=ckpt_dir,
                tag="final",
                extra={"loss": loss_history[-1]}
            )
            print(f"[CKPT] Saved final: {path}")

        # Plot loss curve with log-scaled y if values are large
        try:
            plt.figure()
            plt.plot(range(1, n_epochs + 1), loss_history)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"[PLOT] Saved loss curve to: {plot_path}")
        except Exception as e:
            print(f"[WARN] Could not plot loss: {e}")

        # Return last soft assignment for convenience
        assignments = C_f.detach().cpu().numpy()
        return assignments, loss_history
    
    @torch.no_grad()
    def cluster_labels(self, assignments):
        return assignments.argmax(axis=1)
