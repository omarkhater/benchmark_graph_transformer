# GraphTransformer Benchmark

This repository provides a reproducible benchmarking suite for the
[GraphTransformer](https://github.com/omarkhater/pytorch_geometric/tree/add_graph_transformer) model
using MLflow for experiment tracking and Hydra for configuration.

## Currnet Successfull Benchmarked Datasets

- Proteins
- MUTAG
- CITESEER
- PUBMED
- CORA

## Current Benchmarked Model Variants

- LEGACY: GIN, GCN, GAT, SAGE
- Graph Transformer: Bias Providers, Positional Encoders, GNN integration, Full

## Benchmark summary

### Node classification

| Dataset  | Size (nodes) | Classes | Model                                   | Train Acc | Test Acc | Train Macro F₁ | Test Macro F₁ |
|:---------|-------------:|--------:|:----------------------------------------|----------:|---------:|---------------:|--------------:|
| CORA     |        2 708 |       7 | SAGE-Basic                              |    0.8280 |   0.7714 |         0.8233 |       0.7685 |
| CORA     |        2 708 |       7 | GAT-Basic                               |    0.8728 |   0.8057 |         0.8712 |       0.8002 |
| CORA     |        2 708 |       7 | GCN-Basic                               |    0.8743 |   0.8083 |         0.8731 |       0.8056 |
| CORA     |        2 708 |       7 | GIN-Basic                               |    0.8967 |   0.8310 |         0.8959 |       0.8284 |
| **CORA**     |    2 708 |       7 | **GraphTransformer-Full-GCN-Parallel** | **0.9941** | **0.9356** |     **0.9939** |   **0.9353** |
| CITSEER  |        3 327 |       6 | SAGE-Basic                              |    0.9359 |   0.7905 |         0.9294 |       0.7367 |
| CITSEER  |        3 327 |       6 | GAT-Basic                               |    0.8768 |   0.7613 |         0.8527 |       0.6908 |
| CITSEER  |        3 327 |       6 | GCN-Basic                               |    0.8571 |   0.7635 |         0.8114 |       0.6729 |
| CITSEER  |        3 327 |       6 | GIN-Basic                               |    0.8827 |   0.7713 |         0.8569 |       0.7056 |
| **CITSEER**  | 3 327 |       6 | **GraphTransformer-Bias**               | **1.0000** | **0.8546** |     **1.0000** |   **0.8139** |
| PUBMED   |       19 717 |       3 | SAGE-Basic                              |    0.5654 |   0.7452 |         0.5673 |       0.7425 |
| PUBMED   |       19 717 |       3 | GAT-Basic                               |    0.6076 |   0.7334 |         0.6081 |       0.7318 |
| PUBMED   |       19 717 |       3 | GCN-Basic                               |    0.6076 |   0.7334 |         0.6081 |       0.7318 |
| PUBMED   |       19 717 |       3 | GIN-Basic                               |    0.6190 |   0.7649 |         0.6191 |       0.7618 |
| **PUBMED**   | 19 717 |       3 | **GraphTransformer-Basic**              | **0.6555** | **0.8125** |     **0.6555** |   **0.8024** |

### Graph classification

| Dataset   | Size (graphs) | Classes | Model                                   | Train Acc | Test Acc | Train AUPRC | Test AUPRC | Train F₁ | Test F₁ |
|:----------|--------------:|--------:|:----------------------------------------|----------:|---------:|-----------:|-----------:|---------:|--------:|
| MUTAG     |           188 |       2 | SAGE-Basic                              |    0.6118 |   0.5556 |     0.6031 |     0.5158 |   0.6257 | 0.5556 |
| MUTAG     |           188 |       2 | GAT-Basic                               |    0.6908 |   0.6111 |     0.8533 |     0.8453 |   0.5645 | 0.4636 |
| MUTAG     |           188 |       2 | GCN-Basic                               |    0.6908 |   0.6111 |     0.5755 |     0.5055 |   0.5645 | 0.4636 |
| MUTAG     |           188 |       2 | GIN-Basic                               |    0.6908 |   0.6111 |     0.5548 |     0.4927 |   0.5645 | 0.4636 |
| **MUTAG**     |     188 |       2 | **GraphTransformer-Full-GCN-Parallel** | **0.8355** | **0.8333** | **0.9577** | **0.9606** | **0.8409** | **0.8349** |
| PROTEINS  |         1 113 |       2 | SAGE-Basic                              |    0.5892 |   0.6847 |     0.4117 |     0.4014 |   0.4064 | 0.4893 |
| PROTEINS  |         1 113 |       2 | GAT-Basic                               |    0.6420 |   0.6486 |     0.4014 |     0.4014 |   0.4893 | 0.4893 |
| PROTEINS  |         1 113 |       2 | GCN-Basic                               |    0.5881 |   0.6847 |     0.4087 |     0.4087 |   0.4064 | 0.4893 |
| PROTEINS  |         1 113 |       2 | GIN-Basic                               |    0.6869 |   0.7297 |     0.6209 |     0.6209 |   0.6707 | 0.6707 |
| **PROTEINS** |   1 113 |       2 | **GraphTransformer-Full-GCN-Parallel** | **0.7755** | **0.8108** | **0.6562** | **0.6562** | **0.7792** | **0.7792** |


## 📋 Prerequisites

- Python 3.12+
- Pytorch 2.6+
- Poetry for dependency management
- CUDA-capable GPU (recommended)

## 🏗️ Project Layout

```
graph-transformer-benchmark/
├── configs/          # Hydra YAML configs
│   └── variants/     # Model variant configurations
├── data/            # Dataset artifacts or download scripts
├── src/             # Source code
│   └── graph_transformer_benchmark/
│       ├── data/          # Data Package
│       ├── graph_models/  # Model implementations
│       ├── training/      # Training Package
│       ├── evaluation/    # Evaluation Package
│       ├── utils/         # Helper functions
│       └── cli.py         # Command-line interface
│       └── train.py       # training entry point
├── scripts/         # Automation scripts
│   └── run_models.py      #
├── tests/          # Unit & smoke tests
├── notebooks/      # Exploratory analysis & result plots
├── .gitignore
├── poetry.toml     # Poetry configuration
├── pyproject.toml  # Project dependencies
└── README.md
```

## 🔧 Installation

1. Install Poetry:
```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Unix/MacOS
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/graph-transformer-benchmark.git
cd graph-transformer-benchmark
```

3. Ensure correct Python version (3.12+):
```bash
python --version
```

4. Create and activate virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate
```

5. Install dependencies:
```bash
poetry install
```

## ⚙️ Configuration

Modify configs in `configs/` to customize:
- Model variants:
  - Basic GraphTransformer
  - GraphTransformer with bias
  - GraphTransformer with positional encoding
  - Full GraphTransformer
  - GraphSAGE (baseline)
- Training parameters (batch size, learning rate, etc.)
- Dataset selection (MUTAG, OGBN, Cora, etc.)
- Hardware utilization (GPU/CPU)

## 🚀 Quickstart

1. Benchmarking on MUTAG dataset:

```bash
python scripts/run_models.py `
  --models "variants/basic,variants/bias,variants/pos,variants/gcn_pre,variants/gcn_post,variants/gcn_parallel,variants/full_gcn_pre,variants/full_gcn_post,variants/full_gcn_parallel" `
  --dataset MUTAG `
  --device cpu `
  --exp-name InitialGraphTransformerBenchmarking-MUTAG
```

2. Benchmarking on CORA dataset

```bash
python scripts/run_models.py `
--dataset CORA `
--models "variants/basic,variants/bias,variants/pos,sage,gat,gcn,gin" `
--device cpu `
--exp-name CORA_Initial_Benchmark
```

3. Control Batch size, data loader settings

```bash

python scripts/run_models.py \
  --models "variants/full_gcn_pre" \
  --dataset proteins \
  --device cuda \
  --exp-name test_improved_training-CUDA \
  data.use_subgraph_sampler=true \
  data.sampler.type=neighbor \
  data.batch_size=128 \
  data.sampler.num_neighbors=[5,5]
```

4. Enable Downloading OGB datasets:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python scripts/run_models.py \
  --models "variants/full_gcn_pre" \
  --dataset ogbn-arxiv \
  --device cuda \
  --exp-name test-ogbn \
  data.use_subgraph_sampler=true \
  data.sampler.type=neighbor \
  data.batch_size=16 \
  data.sampler.num_neighbors=[1,1]
```


## 📊 MLFlow Tracking

1. Configure MLflow tracking (optional):
```bash
# Create .env file with MLflow settings
echo "MLFLOW_TRACKING_URI=file:./mlruns" > .env
# Or use a remote tracking server:
# echo "MLFLOW_TRACKING_URI=http://mlflow.example.com:5000" > .env
```

2. Start MLflow UI:
```bash
# Using local file storage (default)
poetry run mlflow ui --backend-store-uri file:./mlruns

# Or using configured tracking URI from .env
poetry run mlflow ui
```

3. View dashboard:
- Local server: http://localhost:5000
- Remote server: Use the URL provided by your MLflow tracking server
  - Example: http://mlflow.example.com:5000
  - Ensure proper network/VPN access if required
  - Authentication may be required depending on server configuration

## Know Bugs (Good First Issues)

- **Regression datasets:**
  Graph-level and node-level regression runs crash with a dtype mismatch (long vs float) inside all models. The culprit is that integer feature tensors coming from certain regression datasets are fed directly into the model, which expects floating-point inputs. Converting the input `data.x` (and any extra node attributes) to `float32` before the first convolution layer might resolve the error.

- **OGB Datasets:**
  * Loading pre-processed OGB files fails under **PyTorch ≥ 2.6** because `torch.load` now defaults to `weights_only=True`, triggering an `UnpicklingError` for `torch_geometric.data.data.DataEdgeAttr`.
  * Even with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, classification-style datasets (e.g. `ogbg-code2`) can survive the initial load but later blow up in the model factory with
    ```
    ValueError: num_class must be a positive int (got -1)
    ```
    because regression tasks set `num_classes = -1`.
  * Node-level datasets like **ogbn-proteins** may have `data.x = None`, causing
    ```
    AttributeError: 'NoneType' object has no attribute 'size'
    ```
    in `enrich_batch` when it does `batch.x.size(0)`.

## 📝 License

MIT License
