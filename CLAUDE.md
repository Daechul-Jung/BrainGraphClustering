# CLAUDE.md — BrainGraphClustering

AI assistant guide for the BrainGraphClustering codebase. Read this before making changes.

---

## Project Overview

Research codebase implementing two cortical parcellation methods for HCP resting-state fMRI data:

1. **Deep Modularity (DMoN)** — proposed method in `modularity/`: GNN-based approach combining functional modularity optimization with spatial smoothness regularization.
2. **gwMRF** — classical baseline in `gwmrf/`: von Mises-Fisher Markov Random Field with graph-cut inference.

Shared preprocessing utilities live in `utilities/`.

---

## Repository Structure

```
BrainGraphClustering/
├── modularity/                  # Proposed DMoN-based method
│   ├── main.py                  # Entry point: per-hemisphere pipeline
│   ├── model.py                 # CombinedModel: dual-head GCN + DMoN
│   ├── gnn.py                   # GCN layers with skip connections
│   ├── dmon.py                  # DMoN: soft assignments + modularity loss
│   ├── trainer.py               # Training loop (Adam, checkpointing)
│   ├── loader.py                # Graph construction from edge density
│   └── utils.py                 # Visualization and checkpoint utilities
│
├── gwmrf/                       # gwMRF baseline
│   ├── main.py                  # Entry point: gwMRF pipeline
│   ├── model.py                 # vMF likelihoods + graph-cut inference
│   ├── trainer.py               # MAP3/MAP1/MAP2 optimization schedule
│   ├── network_clustering.py    # Parcel-to-network clustering (7/17 networks)
│   ├── gwMRF_set_params.py      # Parameter configuration
│   └── gwMRF_generate_components.py  # Connected component analysis
│
├── utilities/                   # Shared preprocessing utilities
│   ├── prepare_func.py          # Load & normalize fMRI time series
│   ├── prepare_all.py           # Full preparation pipeline
│   ├── read_avg_mesh.py         # Surface mesh I/O and neighbor computation
│   ├── spgrad_rsfc_gradient.py  # RSFC gradient computation (wb_command)
│   ├── spgrad_watershed_algorithm.py  # Watershed boundary detection
│   ├── spgrad_findminima.py     # Local minima for seed selection
│   ├── create_border.py         # Border/boundary utilities
│   └── download_batch.py        # HCP data download helpers
│
├── README.md
└── CLAUDE.md                    # This file
```

---

## Running the Code

### Deep Modularity (proposed method)

```bash
python modularity/main.py \
    --epochs 200 \
    --clusters 200 \
    --mu_spatial 0.01 \
    --lr 1e-3 \
    --cuda cuda:0
```

Key CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 200 | Training epochs |
| `--clusters` | 200 | Number of cortical parcels |
| `--mu_spatial` | 0.01 | Weight on spatial smoothness loss |
| `--lr` | 1e-3 | Adam learning rate |
| `--cuda` | `cuda:0` | Device (`cpu` for CPU-only) |

**Expected input files** (loaded from current directory by default):
- `group_mean_lh_func.pt` / `group_mean_rh_func.pt` — PyTorch tensors, shape `(V, T)` (fMRI time series)
- `group_mean_lh_edge_density.npy` / `group_mean_rh_edge_density.npy` — NumPy arrays, shape `(V,)` (gradient maps)
- `avg_mesh_lh.pt` / `avg_mesh_rh.pt` — mesh dicts with keys `coords`, `faces`, `vertexNbors`, `MARS_label`

**Outputs** (saved to current directory):
- `{LH,RH}_{lh,rh}_labels_int.npy` — cortex-only cluster labels (0-indexed)
- `{LH,RH}_{lh,rh}_labels_full_int.npy` — full-surface labels (medial wall vertices = -1)
- `checkpoints/` — model weights (`.pt` files)
- `training_loss.png` — loss curve

### gwMRF (baseline)

```bash
python gwmrf/main.py \
    --input_fullpaths /path/to/data.txt \
    --output_path /path/to/results \
    --num_left_cluster 400 \
    --num_right_cluster 400
```

---

## Architecture Details

### Deep Modularity (modularity/)

**CombinedModel** (`model.py`) has two parallel branches:

```
fMRI time series (V×T)  → GCN encoder → DMoN head → soft assignment C_f
Coordinates (V×3)       → GCN encoder → DMoN head → soft assignment C_s
```

**Loss function:**
```
L = -Q(C_f) + μ · Tr(C_s^T L_s C_s) + λ_f · R(C_f) + λ_s · R(C_s)
```
- `-Q(C_f)`: negative modularity (maximize within-parcel connectivity vs null model)
- `Tr(C_s^T L_s C_s)`: Laplacian smoothness on spatial assignments
- `R(·)`: collapse regularization to prevent empty clusters

**GCN layers** (`gnn.py`): two-layer graph convolution with SELU activation, dropout (0.1), and skip connections (when input/output dims match).

**DMoN module** (`dmon.py`): softmax-normalized cluster assignments, uses `LazyLinear` for deferred weight initialization.

**Graph construction** (`loader.py`):
- Functional adjacency: `A_ij = 1 - g_ij` where `g_ij` is gradient/edge density
- Normalized adjacency: `Â = D^{-1/2} A D^{-1/2}`
- Spatial Laplacian: `L_s = D_s - A_s` from mesh topology

**Inference:** hard labels via `argmax` over soft assignments `C_f`.

### gwMRF (gwmrf/)

Three-stage optimization (`trainer.py`):
1. **MAP3** — initialize via graph cuts with random seeds
2. **MAP1** — iterate κ/μ (kappa/mu) updates to convergence
3. **MAP2** — gradually reduce τ (spatial concentration) via gamma schedule

Likelihood model (`model.py`): von Mises-Fisher distribution on normalized time series and vertex coordinates.

Graph cuts use `pygco` (Potts model, alpha-expansion).

Network assignment (`network_clustering.py`): builds group-level binary connectome (top 10% edges), applies vMF-EM or spherical k-means to map parcels → Yeo 7/17 networks.

---

## Key Conventions

### Data Formats

- **Meshes:** HCP `fs_LR 32k` surface (32,492 vertices per hemisphere). Medial wall vertices are masked with `MARS_label` and excluded from clustering.
- **fMRI:** z-score + L2 normalized per vertex before use (see `utilities/prepare_func.py`).
- **Labels:** 0-indexed integers. Medial wall vertices get label `-1` in full-surface outputs.
- **PyTorch tensors:** functional data loaded with `torch.load(..., weights_only=False)`.
- **NumPy arrays:** gradient maps loaded with `np.load(...)`.

### Sparse Tensors

Adjacency matrices are stored as `torch.sparse_coo_tensor`. Operations on them require `.to_dense()` before most arithmetic. Keep adjacency matrices sparse when passing to GCN — they are used in `torch.spmm`.

### Device Handling

Always pass `device` through from `main.py` to all model/data operations. Do not hardcode `.cuda()` — use `.to(device)` throughout.

### Checkpoints

`utils.py` provides `save_checkpoint(state, path)` and `load_checkpoint(path, model, optimizer)`. The checkpoint dict includes:
- `epoch`, `loss`: training metadata
- `model_state_dict`, `optimizer_state_dict`: weights

### Visualization

`utils.py` wraps `pyvista` for surface mesh visualization. Visualization is optional — skip gracefully if `pyvista` is not installed. Training loss curves are saved with `matplotlib` to `training_loss.png`.

---

## Dependencies

No `requirements.txt` exists. Dependencies are:

```
python >= 3.8
torch >= 1.9         # Neural networks, sparse tensors
numpy                # Numerical computing
scipy                # Sparse matrix utilities
nibabel              # GIFTI/CIFTI/NIFTI neuroimaging I/O
tqdm                 # Progress bars
matplotlib           # Loss curve plots
pyvista              # (optional) 3D surface visualization
pygco                # Graph cuts — required for gwMRF only
```

**External binary:**
- `wb_command` (HCP Workbench) — required only for gradient computation in `utilities/spgrad_rsfc_gradient.py`; called via `subprocess`.

Install example:
```bash
pip install torch numpy scipy nibabel tqdm matplotlib pyvista
# pygco: follow https://github.com/amueller/gco_python for installation
```

---

## Testing

No automated test suite exists. Validation is manual:
- Inspect `training_loss.png` for convergence (loss should decrease and stabilize).
- Check that label arrays have the expected shape `(V_cortex,)` and value range `[0, n_clusters-1]`.
- Use `utils.py` visualization to render parcellations on the mesh.

---

## Common Pitfalls

1. **Medial wall:** always filter using `MARS_label` (or `cortex_indices`) before clustering. The medial wall is not valid cortex and must be excluded from graph construction.

2. **Empty clusters:** DMoN's collapse regularization (`λ_f`, `λ_s`) prevents this during training. If clusters collapse post-hoc during `argmax`, increase `λ` or reduce `--clusters`.

3. **Memory:** at 32k vertices with `T=1200` time points, the functional data matrix is ~150 MB per hemisphere on GPU. Reduce `T` if OOM errors occur.

4. **pygco build:** `pygco` requires a C compiler and may need manual compilation. Only needed for `gwmrf/`; the `modularity/` method has no such dependency.

5. **wb_command path:** `spgrad_rsfc_gradient.py` calls `wb_command` via `subprocess`. Ensure `wb_command` is on `$PATH`.

6. **Input path defaults:** `modularity/main.py` loads data from the current working directory. Run it from the directory containing the `.pt` and `.npy` input files.

---

## Development Workflow

1. Make changes on the appropriate feature branch (never commit directly to `master`).
2. Run the relevant `main.py` to validate end-to-end behavior.
3. Keep `modularity/`, `gwmrf/`, and `utilities/` self-contained — avoid cross-module imports between `modularity/` and `gwmrf/`.
4. Shared preprocessing utilities belong in `utilities/`.
5. Do not add a `requirements.txt` or package configuration without discussion — this is a research repo without a formal packaging setup.

---

## References

- Tsitsulin et al., "Graph Clustering with Graph Neural Networks," JMLR 2023 (DMoN)
- Schaefer et al., "Local-Global Parcellation," Cerebral Cortex 2018
- Gordon et al., "Generation and Evaluation of a Cortical Area Parcellation," Cerebral Cortex 2016
