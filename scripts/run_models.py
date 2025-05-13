#!/usr/bin/env python3
"""
Run one training job per model variant on a single dataset.

Usage:
    python scripts/run_models.py \\
      --dataset PubMed \\
      --models variants/basic,variants/full,sage \\
      --seed 2025 \\
      --device cuda \\
      --exp-name my_first_bench \\
      -- \\
      data.use_subgraph_sampler=true \\
      data.sampler.type=neighbor \\
      data.batch_size=128 \\
      data.sampler.num_neighbors=[5,5]
"""
import argparse
import subprocess
from typing import List


def get_task_kind(name: str) -> str:
    """Determine whether the dataset is graph- or node-level."""
    key = name.lower()
    if key.startswith("ogbn-") or key in {"cora", "citeseer", "pubmed"}:
        return "node"
    return "graph"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-shot training for multiple model variants"
    )
    parser.add_argument(
        "--dataset", "-d", default="MUTAG",
        help="Dataset name (e.g. MUTAG, Cora, ogbn-arxiv)"
    )
    parser.add_argument(
        "--models", "-m",
        default="variants/basic,variants/bias,variants/pos,"
                "variants/full,variants/gnn,sage",
        help="Comma-separated list of model configs"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", "-D", default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on (cpu or cuda)"
    )
    parser.add_argument(
        "--exp-name", "-e", default="benchmark_simple",
        help="MLflow experiment name"
    )

    # parse_known_args lets us collect any trailing Hydra overrides (key=val)
    args, overrides = parser.parse_known_args()

    task = get_task_kind(args.dataset)
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    print(
        f"\n▶ Single-shot runs on '{args.dataset}' (task={task}) "
        f"under experiment '{args.exp_name}'\n"
    )

    for model in model_list:
        print(f"─➤ Model: {model}")
        # Base Hydra overrides
        cmd: List[str] = [
            "poetry", "run", "python", "-m",
            "graph_transformer_benchmark.cli",
            f"data.dataset={args.dataset}",
            f"model={model}",
            f"model.task={task}",
            f"training.device={args.device}",
            f"training.seed={args.seed}",
            f"training.mlflow.experiment_name={args.exp_name}",
        ]
        # Append any extra overrides after the "--"
        if overrides:
            cmd.extend(overrides)

        print(">", " ".join(cmd))
        # capture both stdout and stderr for reporting
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✔ Completed {model}\n")
        else:
            print(f"✘ Failed   {model}")
            print("STDOUT>\n", result.stdout.strip())
            print("STDERR>\n", result.stderr.strip())
            print()  # blank line

    print("✅ All models processed.")


if __name__ == "__main__":
    main()
