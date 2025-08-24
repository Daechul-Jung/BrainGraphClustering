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

**Note**: This implementation represents a significant advancement in brain parcellation methodology, combining the mathematical rigor of traditional MRF approaches with the representational power of modern deep learning techniques.
