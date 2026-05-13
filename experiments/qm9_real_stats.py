#!/usr/bin/env python3
"""
Compute chemical validity / stability stats on real QM9 molecules and log to W&B.

Mirrors the metrics logged by ChemicalValidityCallback but runs on ground-truth
molecules instead of generated ones, giving an upper-bound reference.

Metrics logged:
  chem/validity        — fraction of structurally valid molecules
  chem/uniqueness      — fraction of unique valid identifiers among valid mols
  chem/heavy_atom_mean — mean number of non-H atoms per molecule
  chem/atom_stability  — fraction of atoms with correct bond count
  chem/mol_stability   — fraction of molecules where every atom is stable
  chem/valid_smiles    — W&B table of top-50 identifiers by count

Usage:
    python experiments/qm9_real_stats.py [--split test] [--n_molecules 5000]
"""

import sys
from collections import Counter
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
from model.mol_utils import batch_to_stability, batch_to_validity, heavy_atom_counts

_BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ChemicalValidityCallback metrics on real QM9 molecules."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which QM9 split to evaluate.",
    )
    parser.add_argument(
        "--n_molecules",
        type=int,
        default=5000,
        help="Maximum number of molecules to evaluate (0 = all).",
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
    n_limit = args.n_molecules if args.n_molecules > 0 else int(1e9)

    run = wandb.init(
        entity="equivariant-drifting",
        project="qm9-real-stats",
        name=f"real-{args.split}",
        config=vars(args),
        mode="offline" if args.offline else "online",
    )

    print(f"Loading QM9 from '{args.root}' ...")
    dm = QM9DataModule(root=args.root, n_real_molecules=_BATCH_SIZE, num_workers=0)
    dm.setup()

    split_map = {"train": dm.train_set, "val": dm.val_set, "test": dm.test_set}
    loader = DataLoader(
        split_map[args.split],
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Accumulate tensors across batches (mirrors the callback's collect-then-compute pattern)
    pos_list: list[torch.Tensor] = []
    atype_list: list[torch.Tensor] = []
    bvec_list: list[torch.Tensor] = []
    offset = 0
    n_collected = 0

    print(f"Collecting up to {n_limit} molecules from the '{args.split}' split ...")
    for data in loader:
        if n_collected >= n_limit:
            break

        n_in_batch = int(data.batch.max().item()) + 1
        n_remaining = n_limit - n_collected
        if n_in_batch > n_remaining:
            data = _trim_batch(data, n_remaining)
            n_in_batch = n_remaining

        # Convert one-hot → integer indices expected by mol_utils
        atom_idx = data.real_atom_types.argmax(dim=1)

        pos_list.append(data.pos.cpu())
        atype_list.append(atom_idx.cpu())
        bvec_list.append(data.batch.cpu() + offset)
        offset += n_in_batch
        n_collected += n_in_batch
        print(f"  {n_collected}/{n_limit} collected ...", end="\r")

    print()

    if not pos_list:
        print("No molecules collected; exiting.")
        run.finish()
        return

    pos = torch.cat(pos_list, dim=0)
    a_hard = torch.cat(atype_list, dim=0)
    bvec = torch.cat(bvec_list, dim=0)

    print("Computing validity ...")
    results = batch_to_validity(pos, a_hard, bvec)

    print("Computing heavy-atom counts ...")
    heavy = heavy_atom_counts(a_hard, bvec)

    print("Computing stability ...")
    atom_stable_frac, mol_stable_frac = batch_to_stability(pos, a_hard, bvec)

    n_total = len(results)
    n_valid = sum(1 for ok, _ in results if ok)
    valid_ids = [ident for ok, ident in results if ok and ident is not None]

    validity = n_valid / n_total if n_total > 0 else 0.0
    uniqueness = len(set(valid_ids)) / len(valid_ids) if valid_ids else 0.0
    heavy_mean = float(np.mean(heavy)) if heavy else 0.0

    metrics = {
        "chem/validity": validity,
        "chem/uniqueness": uniqueness,
        "chem/heavy_atom_mean": heavy_mean,
        "chem/atom_stability": atom_stable_frac,
        "chem/mol_stability": mol_stable_frac,
        "n_molecules": n_total,
    }

    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    wandb.log(metrics)

    try:
        table = wandb.Table(columns=["identifier", "count"])
        for ident, cnt in Counter(valid_ids).most_common(50):
            table.add_data(ident, cnt)
        wandb.log({"chem/valid_smiles": table})
    except Exception:
        pass

    run.finish()


if __name__ == "__main__":
    main()
