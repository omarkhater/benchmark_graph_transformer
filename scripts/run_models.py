#!/usr/bin/env python3
"""
Run one training job per model variant on a single dataset.

Usage:
    python scripts/run_models.py \
        --dataset PubMed \
        --models variants/basic,variants/full,sage \
        --seed 2025 \
        --device cuda \
        --exp-name my_first_bench
"""
import argparse
import subprocess


def get_task_kind(name: str) -> str:
    """Determine whether the dataset is graph- or node-level."""
    key = name.lower()
    if key.startswith("ogbn-") or key in {"cora", "citeseer", "pubmed"}:
        return "node"
    return "graph"


def main():
    parser = argparse.ArgumentParser(
        description="Run single-shot training for multiple model variants"
    )
    parser.add_argument(
        "--dataset", "-d", default="MUTAG",
        help="Dataset name (e.g. MUTAG, Cora, ogbn-arxiv)"
    )
    parser.add_argument(
        "--models",
        "-m",
        default="variants/basic,variants/bias,variants/pos,"
        "variants/full,variants/gnn,sage",
        help="Comma-separated list of model configs"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on (cpu or cuda)"
    )
    parser.add_argument(
        "--exp-name", "-e", default="benchmark_simple",
        help="MLflow experiment name"
    )
    args = parser.parse_args()

    task = get_task_kind(args.dataset)
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    print(
        f"\n▶ Single-shot runs on '{args.dataset}' (task={task}) "
        f"under experiment '{args.exp_name}'\n"
    )

    for model in model_list:
        print(f"─➤ Model: {model}")
        cmd = [
            "poetry", "run", "python", "-m", "graph_transformer_benchmark.cli",
            f"data.dataset={args.dataset}",
            f"model={model}",
            f"model.task={task}",
            f"training.device={args.device}",
            f"training.mlflow.experiment_name={args.exp_name}",
        ]
        print(">", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"✔ Completed {model}\n")
        else:
            print(f"✘ Failed   {model}")
            print("STDOUT>", result.stdout.strip())
            print("STDERR>", result.stderr.strip())
            print()  # blank line

    print("✅ All models processed.")


if __name__ == "__main__":
    main()
