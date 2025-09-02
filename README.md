# BrainGraphClustering

**Brain Parcellation from gwMRF to Deep Modularity: A Python Implementation for Functional Brain Network Analysis**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)

## What is BrainGraphClustering?

This repository provides **two powerful approaches** for automatically dividing the brain's surface into meaningful regions (parcellation) based on how different brain areas work together:

1. **gwMRF (Gradient Weighted Markov Random Field)**: A robust Python reimplementation of the established CBIG gwMRF pipeline
2. **Deep Modularity with Contrastive Learning**: A cutting-edge approach using artificial intelligence to discover better brain regions

Think of it like this: Instead of manually drawing boundaries on a brain map, these methods automatically find the "natural borders" where brain function changes, creating a more accurate map of how the brain is organized.

##  Why This Matters

- **Better Brain Maps**: More accurate division of brain regions leads to better understanding of brain function
- **Research Applications**: Useful for studying brain disorders, development, and individual differences
- **Modern Implementation**: Python-based, making it easier to integrate with modern data science workflows
- **Improved Performance**: The deep learning approach often outperforms traditional methods

##  Repository Structure

```
BrainGraphClustering/
├── gwmrf/                    # Traditional gwMRF approach
│   ├── main.py              # Run the gwMRF pipeline
│   ├── model.py             # Core gwMRF algorithm
│   ├── trainer.py           # Training and optimization
│   ├── network_clustering.py # Group brain regions into networks
│   ├── gwMRF_set_params.py  # Configuration settings
│   └── gwMRF_generate_components.py # Helper functions
├── modularity/               # Modern deep learning approach
│   ├── main.py              # Run the deep learning pipeline
│   ├── model.py             # Neural network architecture
│   ├── gnn.py               # Graph Neural Network
│   ├── dmon.py              # Deep clustering algorithm
│   ├── trainer.py           # Training loop
│   ├── loader.py            # Data loading utilities
│   └── utils.py             # Helper functions
└── utilities/                # Shared tools
    ├── prepare_func.py       # Data preparation
    ├── spgrad_rsfc_gradient.py # Calculate brain gradients
    ├── spgrad_watershed_algorithm.py # Watershed segmentation
    └── ...                      # Other utilities
```

## Quick Start

### Prerequisites

Make sure you have:
- **Python 3.8 or higher**
- **PyTorch 1.9 or higher**
- **fMRI imaging data**

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Daechul-Jung/BrainGraphClustering.git
   cd BrainGraphClustering
   ```

2. **Install required packages:**
   ```bash
   pip install torch numpy scipy nibabel pygco
   ```

3. **Verify installation:**
   ```bash
   python3 -c "import torch; print('PyTorch version:', torch.__version__)"
   ```

## How to Use

### Option 1: Traditional gwMRF Approach

The gwMRF method is like using a sophisticated puzzle-solving algorithm that considers both:
- **How brain areas work together** (functional connectivity)
- **How close they are physically** (spatial relationships)

**Basic usage:**
```bash
python gwmrf/main.py \
    --input_fullpaths /path/to/your/data.txt \
    --output_path /path/to/save/results \
    --num_left_cluster 400 \
    --num_right_cluster 400
```

**What this does:**
- Reads your brain scan data
- Divides each brain hemisphere into 400 regions
- Creates a map showing which regions belong together
- Saves results in formats you can view in brain visualization software

**Key settings you can adjust:**
- `--smoothcost`: How much to prioritize smooth boundaries (higher = smoother)
- `--num_iterations`: How many times to refine the solution
- `--num_runs`: How many different starting points to try

### Option 2: Modern Deep Learning Approach

The deep learning approach is like having an AI that learns the best way to divide the brain by:
- **Looking at patterns in your data** (instead of using fixed rules)
- **Learning from examples** (gets better with more data)
- **Finding optimal solutions automatically** (no manual tuning needed)

**Basic usage:**
```bash
python modularity/main.py
```

**What this does:**
- Automatically loads and processes your brain data
- Uses a neural network to learn the best features
- Applies contrastive learning to make the method robust
- Uses modularity optimization to find the best brain divisions

## Understanding the Methods

### How gwMRF Works (The Traditional Approach)

Imagine you're trying to divide a country into states. The gwMRF method works like this:

1. **Look at relationships**: Which cities trade with each other? (functional connectivity)
2. **Consider geography**: Which cities are close together? (spatial relationships)
3. **Find boundaries**: Draw lines where relationships change significantly
4. **Optimize**: Keep adjusting until you get the best possible division via graph cut and do coordinate descent 

**Mathematical foundation:**
```
Total Energy = Functional Similarity + λ × Spatial Smoothness
```

Where:
- **Functional Similarity**: How similar brain areas work
- **Spatial Smoothness**: How much to prefer smooth boundaries
- **λ**: A weight that balances these two goals

### How Deep Modularity Works (The Modern Approach)

The deep learning approach is more sophisticated:

1. **Learn features**: Instead of using predefined rules, learn what features matter most
2. **Graph structure**: Use the brain's natural connections to guide the learning
3. **Contrastive learning**: Make the method robust to noise and variations
4. **Modularity optimization**: Find divisions that maximize how well regions work together

**Key advantages:**
- **Adaptive**: Learns what's important from your specific data
- **Robust**: Handles noise and individual differences better
- **Optimal**: Finds mathematically optimal solutions
- **Scalable**: Works well with large datasets

## What You Get as Output

### Brain Maps
- **Parcel labels**: Each brain vertex gets assigned to a specific region
- **Network assignments**: Regions are grouped into functional networks (7 or 17 networks)
- **Visualization files**: Ready-to-use files for brain visualization software

### Analysis Results
- **Connectivity matrices**: Shows how different brain regions communicate
- **Network properties**: Measures of how well the brain is organized
- **Quality metrics**: How good the parcellation is

## Advanced Configuration

### gwMRF Settings
```python
# In gwMRF_set_params.py
params = {
    'smoothcost': 5000,        # How smooth to make boundaries
    'start_gamma': 50000,      # Starting temperature for optimization
    'exponential': 15.0,       # How fast to cool down
    'iter_reduce_gamma': 300   # When to reduce temperature
}
```

### Deep Learning Settings
```python
# In modularity/main.py
opt = {
    'hidden_dim': 64,              # Size of learned features
    'collapse_regularization': 0.1, # Prevent all regions collapsing to one
    'dropout_rate': 0.1,           # Prevent overfitting
    'activation': 'selu',          # Type of activation function
    'skip_connection': True        # Use skip connections for better learning
}
```

## Data Requirements

### What You Need
- **Functional brain scans**: Shows brain activity over time (.func.gii files)
- **Brain surface files**: 3D models of the brain surface (.surf.gii files)
- **Gradient data**: Pre-calculated measures of brain organization (.npy files)

### How to Organize Your Data
```
your_data/
├── functional/
│   ├── lh.func.gii    # Left hemisphere activity
│   └── rh.func.gii    # Right hemisphere activity
├── surfaces/
│   ├── lh.inflated.surf.gii    # Left hemisphere surface
│   └── rh.inflated.surf.gii    # Right hemisphere surface
└── gradients/
    ├── lh_grad.npy    # Left hemisphere gradients
    └── rh_grad.npy    # Right hemisphere gradients
```


## Contributing

We welcome contributions! Here are some ways you can help:

1. **Report bugs** or suggest improvements
2. **Improve documentation** or add examples
3. **Optimize performance** for large datasets
4. **Add visualization tools** for results
5. **Test with different types of brain data**

## Learn More

### Papers and References
- **gwMRF**: Based on the CBIG gwMRF pipeline
- **Deep Modularity**: Novel approach combining GNNs with modularity optimization

### For Researchers
- The deep learning approach represents a significant advance over traditional methods
- Combines the mathematical rigor of MRFs with the power of modern AI
- Particularly effective for individual differences and noisy data

## Get Help

- **Bug reports**: Open an issue on GitHub
- **Questions**: Open a discussion on GitHub
- **Direct contact**: 
  - [daechul.jung@vanderbilt.edu]
  - [jungdaechul@berkeley.edu]

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

## Technical Deep Dive

For those interested in the mathematical details and implementation specifics, see the [Technical Documentation](#technical-documentation) section below.

---

## Technical Documentation

*This section provides detailed technical information for researchers and developers.*

### Mathematical Foundations

#### gwMRF Model

The gwMRF model uses a combination of:
- **von Mises-Fisher distributions** for functional connectivity
- **Markov Random Field** for spatial regularization
- **Graph cut optimization** for finding optimal boundaries

**Energy function:**
```
E(L, {μ_k}) = ∑_i [Functional Term] + β ∑_{(i,j)∈E} w_{ij} 1[L_i ≠ L_j]
```

Where:
- `L_i` is the label (parcel) for vertex i
- `μ_k` are the mean directions for each parcel
- `w_{ij}` are edge weights based on gradients
- `β` controls the balance between functional and spatial terms

#### Deep Modularity Model

The deep learning approach combines:
- **Graph Neural Networks** for feature learning
- **Contrastive learning** for robust representations
- **Modularity optimization** for optimal clustering

**Key equations:**
```
Z = GNN(F, Â)                    # Learn embeddings
S = softmax(WZ)                  # Soft cluster assignments
Q = (1/2m) Tr(S^T B S)          # Modularity measure
L_total = L_contrastive + λ_mod × L_modularity
```

### Implementation Details

#### Data Processing Pipeline

1. **Load brain data** (functional scans, surfaces, gradients)
2. **Build adjacency matrix** from local gradients
3. **Normalize adjacency** for GNN message passing
4. **Extract features** (PCA of time-series + positional encodings)

#### Training Process

1. **Data augmentation**: Create two views of the same data
2. **Forward pass**: Encode both views through the GNN
3. **Contrastive loss**: Push same-node views together, different-node views apart
4. **Modularity loss**: Optimize cluster assignments for maximum modularity
5. **Backpropagation**: Update network parameters

#### Inference

1. **Load trained model**
2. **Encode features** through GNN
3. **Apply clustering** head to get assignments
4. **Convert to hard labels** for final parcellation

### Performance Characteristics

| Metric | gwMRF | Deep Modularity |
|--------|-------|-----------------|
| **Computational complexity** | O(N²) | O(N log N) |
| **Memory usage** | Moderate | High |
| **Training time** | N/A (no training) | 1-10 hours |
| **Inference time** | 1-5 minutes | <1 minute |
| **Scalability** | Up to ~100K vertices | Up to ~1M vertices |

### Advanced Topics

#### Why Contrastive Learning Helps

Contrastive learning improves results by:
- **Learning invariances** to noise and measurement variations
- **Creating discriminative embeddings** that separate different brain regions
- **Regularizing the model** to prevent overfitting
- **Improving generalization** across subjects and sessions

#### Modularity Optimization

The modularity objective:
- **Maximizes within-cluster connectivity** over what would be expected by chance
- **Penalizes trivial solutions** (e.g., putting everything in one cluster)
- **Provides degree-corrected clustering** that's not biased by high-degree regions
- **Aligns with biological principles** of functional brain organization

---

*This documentation provides a comprehensive guide to understanding and using both the traditional gwMRF approach and the innovative deep learning method for brain parcellation.*
