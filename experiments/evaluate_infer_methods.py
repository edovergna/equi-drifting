#!/usr/bin/env python3
"""
Benchmark atom-type inference methods on real QM9 molecules and log to W&B.

Given only the 3-D positions of real QM9 molecules, each method tries to
recover the known ground-truth atom types (H / C / N / O / F).

Computes:
  accuracy          -- fraction of atoms with correctly inferred type
  accuracy_<elem>   -- per-element accuracy for each of H, C, N, O, F
  sec_per_mol       -- wall-clock seconds per molecule
  confusion_matrix  -- wandb Table: rows = true type, cols = predicted type

Usage:
    python experiments/evaluate_infer_methods.py --infer_method heuristic
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import torch
import wandb
from torch_geometric.loader import DataLoader

from model.datamodule import QM9DataModule
from model.mol_utils import infer_types_from_pos_batch

_BATCH_SIZE = 256
_ELEM_NAMES = ["H", "C", "N", "O", "F"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate atom-type inference methods on QM9 test molecules."
    )
    parser.add_argument(
        "--infer_method",
        type=str,
        default="heuristic",
        choices=["degree", "heuristic"],
        help="Atom-type inference method to benchmark.",
    )
    parser.add_argument(
        "--n_molecules",
        type=int,
        default=5000,
        help="Number of QM9 test-split molecules to evaluate.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/QM9",
        help="Root directory for the QM9 dataset.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Log to W&B in offline mode.",
    )
    return parser.parse_args()


def _trim_batch(data, n_remaining: int):
    """Drop molecules beyond n_remaining from a PyG batch."""
    keep = data.batch < n_remaining
    data.pos = data.pos[keep]
    data.real_atom_types = data.real_atom_types[keep]
    data.batch = data.batch[keep]
    return data


def main() -> None:
    args = parse_args()

    run = wandb.init(
        entity="equivariant-drifting",
        project="infer-method-evaluation",
        name=args.infer_method,
        config=vars(args),
        mode="offline" if args.offline else "online",
    )

    print(f"Loading QM9 from '{args.root}' ...")
    dm = QM9DataModule(root=args.root, n_real_molecules=_BATCH_SIZE, num_workers=0)
    dm.setup()

    test_loader = DataLoader(
        dm.test_set,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    n_classes = len(_ELEM_NAMES)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    total_time = 0.0
    n_evaluated = 0

    print(f"Evaluating '{args.infer_method}' on up to {args.n_molecules} molecules ...")
    for data in test_loader:
        if n_evaluated >= args.n_molecules:
            break

        n_in_batch = int(data.batch.max().item()) + 1
        n_remaining = args.n_molecules - n_evaluated

        if n_in_batch > n_remaining:
            data = _trim_batch(data, n_remaining)
            n_in_batch = n_remaining

        true_idx = data.real_atom_types.argmax(dim=1)

        t0 = time.perf_counter()
        pred_one_hot = infer_types_from_pos_batch(
            data.pos, data.batch, device=torch.device("cpu"), method=args.infer_method
        )
        total_time += time.perf_counter() - t0

        pred_idx = pred_one_hot.argmax(dim=1)

        for t, p in zip(true_idx.tolist(), pred_idx.tolist()):
            confusion[t, p] += 1

        n_evaluated += n_in_batch
        print(f"  {n_evaluated}/{args.n_molecules} evaluated ...", end="\r")

    print()

    n_atoms = confusion.sum()
    n_correct = confusion.diagonal().sum()
    n_mols = n_evaluated

    metrics: dict[str, float | int] = {
        "accuracy": n_correct / n_atoms if n_atoms else 0.0,
        "sec_per_mol": total_time / n_mols if n_mols else 0.0,
        "n_evaluated_mols": n_mols,
        "n_evaluated_atoms": int(n_atoms),
    }
    for i, elem in enumerate(_ELEM_NAMES):
        true_count = confusion[i].sum()
        metrics[f"accuracy_{elem}"] = (
            confusion[i, i] / true_count if true_count else 0.0
        )

    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    wandb.log(metrics)

    # Confusion matrix as a wandb Table (rows = true, cols = predicted)
    header = ["true \\ pred"] + _ELEM_NAMES
    conf_table = wandb.Table(columns=header)
    for i, elem in enumerate(_ELEM_NAMES):
        conf_table.add_data(elem, *confusion[i].tolist())
    wandb.log({"confusion_matrix": conf_table})

    run.finish()


if __name__ == "__main__":
    main()
