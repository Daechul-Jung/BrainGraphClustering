# BrainGraphClustering

Brain Parcellation from gwMRF to Deep Modularity: A Python Implementation for Functional Brain Network Analysis

## Overview

This repository contains two distinct approaches for brain parcellation and network clustering:

1. **gwMRF (Gradient Weighted Markov Random Field)**: A Python reimplementation of the CBIG gwMRF pipeline, originally written in MATLAB
2. **Modularity-based Deep Learning**: A novel approach using Graph Neural Networks (GNNs) with modularity optimization for improved brain parcellation utilizing the same data

## Background

### gwMRF Implementation
The gwMRF code is a complete Python reimplementation of the existing CBIG gwMRF pipeline for brain parcellation. The original MATLAB implementation has been converted to Python using PyTorch, maintaining the same mathematical framework while providing better performance and easier integration with modern deep learning workflows.

### Modularity-based Approach
The modularity code represents an innovative improvement over the traditional gwMRF model, incorporating:
- Graph Neural Networks for feature learning
- Contrastive learning for robust representations
- Modularity optimization for better network structure identification
- Deep clustering with DMoN (Deep Modularity Networks)

## Repository Structure

```
BrainGraphClustering/
├── gwmrf/                    # gwMRF implementation
│   ├── main.py              # Main execution script
│   ├── model.py             # gwMRF model implementation
│   ├── trainer.py           # Training and clustering logic
│   ├── network_clustering.py # Network-level clustering utilities
│   ├── gwMRF_set_params.py  # Parameter configuration
│   └── gwMRF_generate_components.py # Component generation
├── modularity/               # Deep modularity approach
│   ├── main.py              # Main execution script
│   ├── model.py             # Combined GNN + DMoN model
│   ├── gnn.py               # Graph Neural Network implementation
│   ├── dmon.py              # Deep Modularity Networks
│   ├── trainer.py           # Training loop
│   ├── loader.py            # Data loading utilities
│   └── utils.py             # Utility functions
└── utilities/                # Shared utilities
    ├── prepare_func.py       # Data preparation functions
    ├── spgrad_rsfc_gradient.py # RSFC gradient computation
    ├── spgrad_watershed_algorithm.py # Watershed algorithm
    └── ...                   # Other utility modules
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy
- Nibabel (for neuroimaging data)
- PyGCO (for graph cuts)

### Install Dependencies
```bash
pip install torch numpy scipy nibabel pygco
```

## Usage

### gwMRF Implementation

The gwMRF implementation follows the original CBIG pipeline:

```bash
python gwmrf/main.py \
    --input_fullpaths /path/to/subject/data.txt \
    --output_path /path/to/output \
    --num_left_cluster 400 \
    --num_right_cluster 400 \
    --smoothcost 5000 \
    --num_iterations 7 \
    --num_runs 2
```

**Key Parameters:**
- `--num_left_cluster/--num_right_cluster`: Number of parcels per hemisphere
- `--smoothcost`: Spatial smoothness parameter for MRF
- `--num_iterations`: Iterations per random initialization
- `--num_runs`: Number of random initializations

**Output:**
- Parcel labels for each hemisphere
- Network assignments (7 and 17 networks)
- Visualization-ready .label.gii files

### Modularity-based Approach

The modularity approach uses deep learning for improved parcellation:

```bash
python modularity/main.py
```

**Key Features:**
- **GCN Encoder**: Graph Convolutional Network for feature learning
- **Contrastive Learning**: NT-Xent loss for robust representations
- **DMoN Clustering**: Deep Modularity Networks for optimal clustering
- **Modularity Optimization**: Direct optimization of network modularity

**Training Process:**
1. Feature extraction using GCN
2. Contrastive learning with data augmentation
3. Modularity-based clustering
4. Combined loss optimization

## Technical Details

### gwMRF Model

The gwMRF model implements:

- **von Mises-Fisher (vMF) distributions** for functional connectivity
- **Spatial regularization** using geodesic distances
- **Graph cut optimization** for parcel boundaries
- **Multi-resolution clustering** with gamma annealing

**Mathematical Framework:**
```
U_total = U_global + λ * U_spatial
```
Where:
- `U_global`: Functional connectivity likelihood
- `U_spatial`: Spatial smoothness constraint
- `λ`: Smoothness weight parameter

### Modularity Model

The modularity approach combines:

- **Graph Neural Networks**: `GCN(features, adjacency) → embeddings`
- **Deep Clustering**: `DMoN(embeddings, adjacency) → clusters`
- **Modularity Loss**: Direct optimization of network modularity
- **Contrastive Learning**: Robust representation learning

**Key Innovations:**
1. **End-to-end learning** of both features and clustering
2. **Modularity-driven optimization** instead of heuristic approaches
3. **Contrastive learning** for better generalization
4. **Graph-aware architecture** preserving spatial relationships

## Performance Comparison

| Aspect | gwMRF | Modularity |
|--------|-------|------------|
| **Approach** | Traditional MRF | Deep Learning |
| **Optimization** | Graph cuts | Gradient descent |
| **Feature Learning** | Manual (RSFC gradients) | Learned (GNN) |
| **Spatial Constraints** | Explicit (geodesic) | Implicit (graph structure) |
| **Scalability** | Moderate | High |
| **Interpretability** | High | Moderate |

## Data Requirements

### Input Format
- **Functional Data**: .func.gii files (HCP format)
- **Surface Data**: .surf.gii files (FreeSurfer format)
- **Gradient Data**: Pre-computed RSFC gradients (.npy)

### Data Structure
```
subject_data/
├── functional/
│   ├── lh.func.gii
│   └── rh.func.gii
├── surfaces/
│   ├── lh.inflated.surf.gii
│   └── rh.inflated.surf.gii
└── gradients/
    ├── lh_grad.npy
    └── rh_grad.npy
```

## Advanced Configuration

### gwMRF Parameters
```python
# In gwMRF_set_params.py
params = {
    'smoothcost': 5000,        # Spatial smoothness
    'start_gamma': 50000,      # Initial gamma value
    'exponential': 15.0,       # Gamma decay rate
    'iter_reduce_gamma': 300   # Gamma reduction frequency
}
```

### Modularity Parameters
```python
# In modularity/main.py
opt = {
    'hidden_dim': 64,              # GNN hidden dimension
    'collapse_regularization': 0.1, # Clustering regularization
    'dropout_rate': 0.1,           # Dropout for regularization
    'activation': 'selu',          # Activation function
    'skip_connection': True        # Skip connections in GNN
}
```

## Output and Visualization

### Parcel Maps
- **Vertex-level labels**: Direct mapping to surface vertices
- **Network assignments**: 7 and 17 network parcellations
- **Format**: .label.gii files for FreeSurfer visualization

### Analysis Results
- **Connectivity matrices**: Parcel-to-parcel connectivity
- **Network properties**: Modularity, clustering coefficient
- **Spatial statistics**: Parcel size, shape, and distribution

## Contributing

This repository welcomes contributions! Areas for improvement include:

1. **Performance optimization** for large-scale datasets
2. **Additional clustering algorithms** for network identification
3. **Visualization tools** for parcellation results
4. **Validation metrics** for parcellation quality

## Citation

If you use this code in your research, please cite:

```bibtex
@software{brain_graph_clustering,
  title={BrainGraphClustering: From gwMRF to Deep Modularity},
  author={Daechul-Jung},
  year={2025},
  url={https://github.com/Daechul-Jung/BrainGraphClustering}
}
```

## Contact

For questions and support, please open an issue on GitHub or contact [daechul.jung@vanderbilt.edu] or [jungdaechul@berkeley.edu]

---

# Deep Modularity with Contrastive Learning

## Overview

**Goal**: Parcellate a triangle mesh (e.g., cortex) whose vertices carry **time-series** into coherent clusters ("parcels").

**Key Ingredients**:
- A **gradient-weighted graph** from the mesh: neighbors with small local gradient are strongly connected
- A **GNN encoder** that turns each vertex's features into an embedding
- **Contrastive learning** to make embeddings robust/invariant
- **DMoN (deep modularity)** to turn embeddings into clusters by maximizing graph modularity

## Mathematical Framework

### Inputs and Symbols

- **Mesh**: Triangle mesh with vertices V={1,…,N} and edges E from mesh topology (typically ~6 neighbors)
- **Per-vertex data**:
  - **Time-series**: $x_i \in \R^T$ for vertex i
  - **Coordinates**: $p_i \in \R^3$ (optional)
  - **Local gradient** between neighbors i and j: $g_{ij} \in [0,1]$ (0 = very similar, 1 = very dissimilar)

- **Graph**:
  - **Weighted adjacency** $A \in \R^{N \times N}$: $A_{ij} = 1 - g_{ij}$ if $(i,j) \in E$, else 0
  - **Degree vector** $d \in \R^N$: $d_i = \sum_j A_{ij}$
  - **Normalized adjacency** $\hat{A} = D^{-1/2}AD^{-1/2}$ with $D = \text{diag}(d)$
  - **Modularity matrix** $B = A - \frac{dd^\top}{2m}$, where $m = \frac{1}{2}\sum_{ij} A_{ij}$

- **Features** per vertex: Possibly **PCA(time-series)** + **positional encodings** of $p_i$, concatenated into $f_i \in \R^F$

- **Encoder** $f_\theta$ (GNN): Produces embeddings $z_i = f_\theta(f_i, \hat{A}) \in \R^d$

- **Soft assignments** $S \in \R^{N \times K}$ (rows sum to 1): $S = \text{softmax}(Wz + b)$

### End-to-End Workflow

#### 1. Data + Graph Construction (`loader.py`)

1. Build $A$ from **local gradient**: $A_{ij} = \begin{cases} 1 - \frac{g_i + g_j}{2}, & j \in N(i) \\ 0, & \text{otherwise} \end{cases}$
2. Compute $D$, $\hat{A} = D^{-1/2}AD^{-1/2}$
3. Build features $F$ (PCA(time-series) + positional encodings)
4. Return: `features (N×F)`, `adj (A)`, `norm_adj (ĤA)`, `deg (d)` — all as PyTorch tensors

#### 2. Encoder (`gnn.py`)

Graph convolution layer:
$H^{(1)} = \sigma(\hat{A}FW_0 + b_0)$, $Z = \sigma(\hat{A}H^{(1)}W_1 + b_1)$

#### 3. Clustering Head (`dmon.py`)

- Soft assignments $S = \text{softmax}(WZ)$
- Pooled graph $G_p = S^\top AS$
- **Modularity objective**: $Q = \frac{1}{2m}\text{Tr}(S^\top BS)$, $L_{mod} = -Q + \lambda_c \underbrace{\left(\frac{\|1^\top S\|_2}{N\sqrt{K}} - 1\right)}_{\text{collapse regularizer}}$

#### 4. Contrastive Learning (in `main.py`)

- Make **two augmentations** of features/graph: feature masking, gaussian noise, (optionally) edge dropout
- Pass both views through the same encoder: $Z^{(1)} = f_\theta(F^{(1)}, \hat{A})$, $Z^{(2)} = f_\theta(F^{(2)}, \hat{A})$
- **NT-Xent / InfoNCE** (node-wise positives across views):
  $L_{ctr} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\text{sim}(z_i^{(1)}, z_i^{(2)})/\tau)}{\sum_{j=1}^N \exp(\text{sim}(z_i^{(1)}, z_j^{(2)})/\tau)}$
  with $\text{sim}(u,v) = \frac{u^\top v}{\|u\| \|v\|}$
- **Total loss** (train end-to-end): $L = L_{ctr} + \lambda_{mod} L_{mod}$

### Training Loop (High-Level)

```python
repeat:
  # augment two views
  F1 = mask/noise(F);  A1 = maybe edge-drop(A)  [optional]
  F2 = mask/noise(F);  A2 = maybe edge-drop(A)  [optional]

  # encode
  Z1 = GNN(F1, ĤA)
  Z2 = GNN(F2, ĤA)

  # contrastive loss
  Lctr = NT_Xent(Z1, Z2)

  # modularity loss on one view
  Hp, S, Lmod = DMoN(Z1, A)

  # total
  L = Lctr + λ_mod * Lmod
  backprop & update θ, W
```

### Inference (No Augmentations, No Contrastive Loss)

```python
with no_grad:
  Z = GNN(F, ĤA)
  _, S, _ = DMoN(Z, A)
  labels = S.argmax(dim=1)  # hard clusters
```

## Why Deep Modularity Works on Gradient-Weighted Meshes

### Modularity with Local-Gradient Graph

- Your **adjacency** encodes **functional similarity** (low gradient ⇒ strong edge): $A_{ij} = 1 - g_{ij}$
- The **modularity matrix** $B = A - \frac{dd^\top}{2m}$ subtracts a **null-model** term (what would be expected "by chance" given degrees)
- This prevents overvaluing hubs or dense neighborhoods simply due to degree, and rewards partitions that have **more within-cluster weight than expected**

### DMoN Objective (Spectral Intuition)

- Classic spectral community detection maximizes $\text{Tr}(S^\top BS)$ in a relaxed space
- DMoN **learns** $S$ jointly with **embeddings** $Z$, letting the network discover feature-aware communities that also **align with the graph's gradient structure**
- Because $A$ comes from **gradient similarity**, maximizing modularity prefers clusters where **local gradients are low within clusters** and **high across clusters**, exactly what you want for parcellation

## Advantages Over vMF / Gradient-Weighted MRFs

### What vMF-MRF Does

- **Unary (likelihood)**: Aligns each vertex's time-series direction to a parcel mean direction via **von Mises–Fisher** on the unit sphere
- **Pairwise (MRF)**: Penalizes label discontinuities across edges, often weighting edges by **gradient magnitude** (higher gradient ⇒ higher penalty to "same label")
- **Energy** (schematic): $E(L, \{\mu_k\}) = \sum_i \underbrace{-\kappa_{L_i} \mu_{L_i}^\top y_i}_{\text{vMF unary}} + \beta \sum_{(i,j) \in E} w_{ij} \mathbf{1}[L_i \neq L_j]$
- Optimized by iterative, hand-crafted procedures

### Limitations of vMF-MRF

- **Fixed similarity**: vMF assumes angular similarity in raw/normalized time-series; cannot learn **nonlinear** invariances or integrate rich spatial/temporal cues
- **Local pairwise modeling**: Pairwise MRF does not incorporate a **global null model**; high-degree regions can bias solutions
- **No end-to-end features**: Features aren't learned from data; they're imposed

### Advantages of Contrastive + DMoN

1. **Learned representations** $Z = f_\theta(F, \hat{A})$: The encoder **learns** nonlinear mappings that make clusters **linearly separable** in $Z$ while respecting mesh structure via $\hat{A}$
2. **Global optimality proxy** via modularity: Maximizing $\text{Tr}(S^\top BS)$ explicitly targets **excess within-cluster connectivity over chance**, i.e., a **degree-corrected** notion of community
3. **End-to-end training**: $\theta$ and $W$ are optimized jointly with the clustering objective; no hand engineering of potentials
4. **Robustness via contrastive learning**:
   - **Invariance** to noise/missing data/edge changes from augmentations
   - **Regularizes** embeddings to avoid degenerate or brittle partitions
5. **Soft assignments + collapse regularization**: Prevents trivial solutions, unlike some MRF settings that can stick in poor local minima

**Net effect**: You retain the **good inductive bias** of the gradient-weighted graph while adding **learned invariances** and a **global, degree-aware** clustering principle (modularity)

## Why Contrastive Learning Matters

- **Not strictly required**, but it **consistently helps**:
  - **Discriminative embeddings**: Pushes different nodes apart and same-node views together ⇒ clearer cluster structure in $Z$
  - **Invariance**: Feature masking/noise/edge-drop teach robustness to measurement noise and mesh idiosyncrasies
  - **Stability / generalization**: Better cross-subject/session behavior
  - **Anti-collapse pressure**: Complements DMoN's collapse regularizer by structuring the embedding space

Mathematically, NT-Xent maximizes a **lower bound on mutual information** between views of the same node. You acquire embeddings that preserve node-specific signal and discard nuisance variation — a perfect pretext for clustering.

## Minimal Code Sketch

```python
# loader: build A from local gradients, ĤA, features
features, A, ĤA, deg = load_hcp_subject(...)

# model
Z1 = GNN(augment(features), ĤA)   # view 1
Z2 = GNN(augment(features), ĤA)   # view 2
L_ctr = NT_Xent(Z1, Z2)

Hp, S, L_mod = DMoN(Z1, A)        # modularity loss on one view

L_total = L_ctr + λ_mod * L_mod
L_total.backward(); optimizer.step()

# inference (no aug, no contrastive)
Z = GNN(features, ĤA)
_, S, _ = DMoN(Z, A)
labels = S.argmax(1)
```

## Glossary

- **$A$**: Weighted adjacency from local gradient (data structure, not learned)
- **$\hat{A}$**: Symmetrically normalized adjacency used in GNN message passing
- **$D$**: Degree matrix; $d$: degree vector
- **$B$**: Modularity matrix $A - \frac{dd^\top}{2m}$
- **$F$**: Input features per vertex (PCA(time-series) + positional encodings)
- **$Z$**: Learned node embeddings from the GNN
- **$W$**: Learnable linear map from embeddings to clusters
- **$S$**: Soft $N \times K$ cluster assignment matrix (rows sum to 1)

---

**Note**: This implementation represents a significant advancement in brain parcellation methodology, combining the mathematical rigor of traditional MRF approaches with the representational power of modern deep learning techniques.
