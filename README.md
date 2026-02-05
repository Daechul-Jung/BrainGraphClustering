# BrainGraphClustering

**Deep Modularity Networks for Brain Parcellation on the Cortical Mesh**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

This repository implements two cortical parcellation methods that partition the cerebral cortex into functionally coherent regions using resting-state functional connectivity (RSFC) data from the Human Connectome Project (HCP):

1. **Deep Modularity (DMoN)** -- A GNN-based method that optimizes a differentiable modularity objective on a gradient-weighted cortical mesh graph, with Laplacian smoothness for spatial contiguity.
2. **gwMRF Baseline** -- A PyTorch re-implementation of the gradient-weighted Markov Random Field model (Schaefer et al., 2018) with vMF likelihoods and graph-cut inference.

---

## Method Overview

### Deep Modularity Parcellation

The proposed method builds a gradient-weighted functional graph from RSFC gradients on the cortical surface mesh, encodes per-vertex signals (fMRI time series + spatial coordinates) via GCN encoders, and produces soft cluster assignments through DMoN heads. The objective combines:

- **Functional modularity**: maximizes within-parcel connectivity relative to a degree-preserving null model
- **Laplacian smoothness**: penalizes jagged boundaries on the cortical mesh
- **Collapse regularization**: prevents degenerate solutions where clusters go unused

$$\mathcal{L} = \underbrace{-\frac{1}{2m}\text{Tr}(C_f^\top B C_f)}_{\text{modularity}} + \underbrace{\mu\,\text{Tr}(C_s^\top L_s C_s)}_{\text{smoothness}} + \lambda_f \mathcal{R}_{\text{col}}(C_f) + \lambda_s \mathcal{R}_{\text{col}}(C_s)$$

### gwMRF Baseline

The gwMRF model combines a von Mises-Fisher mixture likelihood for RSFC time series, a gradient-weighted pairwise MRF prior, and a spatial vMF term for connectivity encouragement. Inference uses a MAP3/MAP1/MAP2 schedule with alpha-expansion graph cuts.

---

## Repository Structure

```
BrainGraphClustering/
├── modularity/                          # Deep Modularity (proposed method)
│   ├── main.py                          # Entry point: per-hemisphere pipeline
│   ├── model.py                         # CombinedModel: dual-head GCN + DMoN
│   ├── gnn.py                           # GCN layers with skip connections
│   ├── dmon.py                          # DMoN: soft assignments + modularity loss
│   ├── trainer.py                       # Training loop (Adam, checkpointing)
│   ├── loader.py                        # Graph construction from edge density
│   └── utils.py                         # Visualization and checkpoint utilities
│
├── gwmrf/                               # gwMRF baseline
│   ├── main.py                          # Entry point: gwMRF pipeline
│   ├── model.py                         # vMF likelihoods + graph-cut inference
│   ├── trainer.py                       # MAP3/MAP1/MAP2 optimization
│   ├── network_clustering.py            # Parcel-to-network clustering (7/17)
│   ├── gwMRF_set_params.py              # Parameter configuration
│   └── gwMRF_generate_components.py     # Connected component analysis
│
└── utilities/                           # Shared preprocessing
    ├── prepare_func.py                  # Load & normalize fMRI time series
    ├── prepare_all.py                   # Full preparation pipeline
    ├── read_avg_mesh.py                 # Surface mesh I/O and averaging
    ├── spgrad_rsfc_gradient.py          # RSFC gradient computation
    ├── spgrad_watershed_algorithm.py    # Watershed boundary detection
    ├── spgrad_findminima.py             # Local minima for seed selection
    ├── create_border.py                 # Border/boundary utilities
    └── download_batch.py                # HCP data download helpers
```

---

## Installation

```bash
git clone https://github.com/Daechul-Jung/BrainGraphClustering.git
cd BrainGraphClustering
pip install torch numpy scipy nibabel pygco tqdm matplotlib pyvista
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.9
- NumPy, SciPy, nibabel (neuroimaging I/O)
- pygco (graph cuts, for gwMRF baseline)
- tqdm, matplotlib (training progress and plots)
- pyvista (optional, for 3D surface visualization)

---

## Data

This project uses HCP resting-state fMRI in **fs_LR 32k** space. Each hemisphere has ~32,492 vertices, with medial-wall vertices excluded during processing.

**Required preprocessed files** (per hemisphere):
| File | Shape | Description |
|------|-------|-------------|
| `avg_mesh_{lh,rh}.pt` | dict | Averaged surface mesh (coords, faces, neighbors, cortex indices) |
| `group_mean_{lh,rh}_func.pt` | `(V, T)` | Group-mean unit-normalized fMRI time series |
| `group_mean_{lh,rh}_edge_density.npy` | `(V,)` | RSFC gradient edge-density map (0 = homogeneous, 1 = boundary) |

The preprocessing pipeline in `utilities/` handles: GIFTI loading, time-series normalization (z-score + L2), subsampled RSFC gradient computation, and watershed-based edge-density maps.

---

## Usage

### Deep Modularity

```bash
python modularity/main.py \
    --epochs 200 \
    --clusters 200 \
    --mu_spatial 0.01 \
    --lr 1e-3 \
    --cuda cuda:0
```

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Training epochs |
| `--clusters` | 200 | Number of parcels per hemisphere (K) |
| `--mu_spatial` | 0.01 | Weight for Laplacian spatial smoothness |
| `--lr` | 1e-3 | Adam learning rate |
| `--cuda` | `cuda:7` | Device |

**Model architecture defaults** (set in `main.py`):
- Hidden dimension: 1024
- Activation: SELU
- Dropout: 0.1
- Collapse regularization: 0.1
- Skip connections: enabled
- 2-layer GCN for both functional and spatial encoders

**Outputs** (saved to `results/`):
- `{LH,RH}_{lh,rh}_labels_int.npy` -- Cortex-only cluster labels (0-indexed)
- `{LH,RH}_{lh,rh}_labels_full_int.npy` -- Full-surface labels (medial wall = -1)
- `checkpoints/` -- Model checkpoints
- `training_loss.png` -- Loss curve

### gwMRF Baseline

```bash
python gwmrf/main.py \
    --input_fullpaths /path/to/data.txt \
    --output_path /path/to/results \
    --num_left_cluster 400 \
    --num_right_cluster 400
```

See `gwmrf/gwMRF_set_params.py` for full configuration (smoothcost, gamma schedule, gradient prior type).

---

## Architecture Details

### Pipeline (Deep Modularity)

```
HCP rfMRI (dtseries) ──► Vertex time series X_f ∈ R^{N×T}
                                │
Surface mesh + RSFC gradients ──► Gradient-weighted adjacency A_{ij} = 1 - g_{ij}
                                │       Normalized: A_hat = D^{-1/2} A D^{-1/2}
                                │       Mesh Laplacian: L_s = D_s - A_s
                                ▼
                         GCN Encoders
                    ┌─── GCN_f(X_f, A_hat) → Z_f ∈ R^{N×K}
                    │
                    └─── GCN_s(X_s, A_hat) → Z_s ∈ R^{N×K}
                                ▼
                         DMoN Heads
                    ┌─── C_f = softmax(W_f · Z_f)  → modularity loss
                    │
                    └─── C_s = softmax(W_s · Z_s)  → Laplacian smoothness
                                ▼
                    Loss = -Q(C_f) + μ·Tr(C_s^T L_s C_s) + λ_f·R(C_f) + λ_s·R(C_s)
                                ▼
                    Labels: l_i = argmax_k (C_f)_{ik}
```

### Key Design Choices

- **Dual-head architecture**: Functional and spatial signals are encoded separately, preventing the spatial head from dominating the modularity objective.
- **Gradient-weighted graph**: Edge weights `A_{ij} = 1 - g_{ij}` preserve RSFC boundary sensitivity from the gradient map while providing a smooth, differentiable graph for GCN message passing.
- **No graph cuts**: The entire pipeline is differentiable and optimized end-to-end with Adam, avoiding the discrete optimization and scaling sensitivity of MRF approaches.

---

## Evaluation Protocol

For comprehensive benchmarking, we recommend:

| Metric | Description |
|--------|-------------|
| **Objective descent** | Loss components over training (modularity, Laplacian, collapse) |
| **Functional homogeneity** | Within-parcel mean RSFC correlation |
| **Spatial contiguity** | Connected components per parcel; boundary smoothness |
| **Stability** | Adjusted Rand Index / Variation of Information across seeds |
| **Runtime** | Wall-clock convergence time per hemisphere |
| **Network validity** | Overlap with Yeo 7/17 networks (optional) |

---

## References

- Schaefer, A. et al. "Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI." *Cerebral Cortex* (2018).
- Tsitsulin, A. et al. "Graph Clustering with Graph Neural Networks." *JMLR* 24 (2023).
- Gordon, E.M. et al. "Generation and Evaluation of a Cortical Area Parcellation from Resting-State Correlations." *Cerebral Cortex* (2016).

## Citation

```bibtex
@software{jung2025braingraphclustering,
  title   = {BrainGraphClustering: Deep Modularity Networks for Brain Parcellation},
  author  = {Jung, Daechul},
  year    = {2025},
  url     = {https://github.com/Daechul-Jung/BrainGraphClustering}
}
```

## Contact

- Daechul Jung -- [daechul.jung@vanderbilt.edu](mailto:daechul.jung@vanderbilt.edu)
