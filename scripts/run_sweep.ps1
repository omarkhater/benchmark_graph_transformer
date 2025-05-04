<#
.SYNOPSIS
    Run an Optuna sweep over the six Graph‑Transformer flavours
    (plus a pure GNN) on a single dataset and seed.

.DESCRIPTION
    • Automatically sets `model.task` to "graph" or "node"
      depending on the chosen dataset.
    • Accepts overrides for Optuna trials and GPU/CPU device.
    • Emits coloured status messages and propagates PowerShell
      errors if the CLI fails.
#>

[CmdletBinding()]
param(
  [string] $Dataset = "MUTAG",
  [int]    $Seed = 42,
  [int]    $Trials = 20, # override training.n_trials
  [string] $Device = "cuda"    # or "cpu"
)

function Get-TaskKind {
  param([string] $Name)
  $k = $Name.ToLower()
  if ($k -like "ogbn-*") { return "node" }
  if ($k -in @("cora", "citeseer", "pubmed")) { return "node" }
  return "graph"   # TU, ogbg-, etc.
}

# -------------------------------------------------------------------------
$task = Get-TaskKind $Dataset
$variants = @(
  "variants/basic",
  "variants/bias",
  "variants/pos",
  "variants/full",
  "sage"
) -join ','

Write-Host "`n▶  Running Optuna sweep on '$Dataset' (task=$task) …" -ForegroundColor Cyan

# Build command array
$cmd = @(
  "poetry", "run", "python", "-m", "graph_transformer_benchmark.cli",
  "--multirun",
  "data.dataset=$Dataset",
  "training.n_trials=$Trials",
  "training.device=$Device",
  "training.mlflow.experiment_name=sweep_$Dataset",
  "hydra/sweeper=optuna",
  "model=$variants",
  "++model.task=$task"
)

# Split into executable + args
$exe = $cmd[0]
$args = $cmd[1..($cmd.Length - 1)]

Write-Host "▶ Executing: $exe $($args -join ' ')" -ForegroundColor DarkYellow

# Invoke
& $exe @args

if ($LASTEXITCODE -ne 0) {
  throw "Sweep failed with exit code $LASTEXITCODE"
}

Write-Host "✅  Completed sweep on '$Dataset' (seed=$Seed)" -ForegroundColor Green
