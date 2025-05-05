# GraphTransformer Benchmark

This repository provides a reproducible benchmarking suite for the
[GraphTransformer](https://github.com/omarkhater/pytorch_geometric/tree/add_graph_transformer) model
using MLflow for experiment tracking and Hydra for configuration.

## 📋 Prerequisites

- Python 3.12+
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
│       ├── data/          # Dataset loaders & preprocessing
│       ├── graph_models/  # Model implementations
│       ├── training/      # Training loops & utils
│       ├── utils/         # Helper functions
│       └── cli.py         # Command-line interface
├── scripts/         # Automation scripts
│   └── run_sweep.ps1
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
--dataset MUTAG `
--models "variants/full,sage,gat,gcn,gin" `
--device cpu `
--exp-name MUTAG_Initial_Benchmark
```

2. Benchmarking on CORA dataset

```bash
python scripts/run_models.py `
--dataset CORA `
--models "variants/full,sage,gat,gcn,gin" `
--device cpu `
--exp-name CORA_Initial_Benchmark
```

## 🔍 Hyperparameter Optimization

The repository includes Optuna-based sweeping functionality:
- Automatic task detection (graph/node classification)
- Parallel trial execution
- MLflow experiment tracking
- Supported datasets:
  - Graph classification: TU datasets, OGB graph datasets
  - Node classification: Cora, CiteSeer, PubMed, OGB node datasets

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

## 📝 License

MIT License
