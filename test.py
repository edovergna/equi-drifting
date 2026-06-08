#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import MoleculeGenerator, QM9DataModule
from model.geometry import center_positions_per_graph
from model.mol_utils import batch_to_stability, batch_to_validity, heavy_atom_counts
from model.wandb_utils import load_config, load_pretrained_generator

# TO BE UPDATED FOR NEW STRUCTURE

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated molecule metrics per molecule size."
    )
    parser.add_argument("--wandb_run_id", required=True)
    parser.add_argument("--wandb_variant", choices=["best", "final"], default="best")
    parser.add_argument("--n_per_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--root", default="data/QM9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--size_split",
        choices=["train", "val", "test", "all"],
        default="val",
        help="Used only when W&B config has no min_num_atoms/max_num_atoms filter.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def optional_int(value):
    if value is None or value == "None" or value == "":
        return None
    return int(value)


def build_model_from_wandb_config(config: dict) -> MoleculeGenerator:
    generator_cfg = {
        "hidden_nf": config.get("hidden_dim", 256),
        "n_layers": config.get("num_layers", 9),
        "aggr_type": config.get("aggr_type", "sum"),
        "num_atom_types": config.get("num_atom_types", 5),
        "tanh_coord_updates": config.get("tanh_coord_updates", True),
        "attention": config.get("attention", True),
    }

    drift_cfg = {
        "n_gen_molecules": config.get("n_gen_molecules", 64),
        "num_atom_types": config.get("num_atom_types", 5),
        "eps": config.get("epsilon", 1e-8),
    }

    return MoleculeGenerator(generator_cfg=generator_cfg, drift_cfg=drift_cfg)


def load_sizes_from_qm9(args: argparse.Namespace, config: dict) -> list[int]:
    dm = QM9DataModule(
        root=args.root,
        n_real_molecules=1,
        num_workers=0,
        sample_frac=float(config.get("sample_frac", 1.0)),
        min_num_atoms=None,
        max_num_atoms=None,
    )
    dm.setup("fit")

    if args.size_split == "train":
        return list(dm.available_train_num_atoms)
    if args.size_split == "val":
        return list(dm.available_val_num_atoms)
    if args.size_split == "test":
        return list(dm.available_test_num_atoms)

    return sorted(
        set(dm.available_train_num_atoms)
        | set(dm.available_val_num_atoms)
        | set(dm.available_test_num_atoms)
    )


def load_possible_sizes(args: argparse.Namespace, config: dict) -> list[int]:
    min_num_atoms = optional_int(config.get("min_num_atoms"))
    max_num_atoms = optional_int(config.get("max_num_atoms"))

    qm9_sizes = load_sizes_from_qm9(args, config)

    if min_num_atoms is not None or max_num_atoms is not None:
        if min_num_atoms is not None:
            qm9_sizes = [s for s in qm9_sizes if s >= min_num_atoms]
        if max_num_atoms is not None:
            qm9_sizes = [s for s in qm9_sizes if s <= max_num_atoms]

        if not qm9_sizes:
            raise RuntimeError(
                f"No QM9 sizes remain after W&B config filter: "
                f"min_num_atoms={min_num_atoms}, max_num_atoms={max_num_atoms}"
            )

    return qm9_sizes


@torch.no_grad()
def generate_fixed_size(
    model: MoleculeGenerator,
    n_molecules: int,
    num_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_prior, pos_prior, batch_vec, edge_index = model._sample_prior_batch(
        n_molecules=n_molecules,
        num_atoms=num_atoms,
    )

    pos_prior = center_positions_per_graph(pos_prior, batch_vec)

    gen_pos, gen_type_logits = model.generator(pos_prior, x_prior, edge_index)
    gen_pos = center_positions_per_graph(gen_pos, batch_vec)
    gen_atom_types = gen_type_logits.softmax(dim=-1).argmax(dim=-1)

    return gen_pos.cpu(), gen_atom_types.cpu(), batch_vec.cpu()


def compute_callback_metrics(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
) -> dict:
    results = batch_to_validity(pos, atom_types, batch_vec)
    atom_stability, mol_stability = batch_to_stability(pos, atom_types, batch_vec)
    heavy = heavy_atom_counts(atom_types, batch_vec)

    n_total = len(results)
    n_valid = sum(1 for ok, _ in results if ok)
    valid_ids = [ident for ok, ident in results if ok and ident is not None]

    validity = n_valid / n_total if n_total else 0.0
    uniqueness = len(set(valid_ids)) / len(valid_ids) if valid_ids else 0.0

    return {
        "n": n_total,
        "validity": validity,
        "uniqueness": uniqueness,
        "atom_stability": float(atom_stability),
        "mol_stability": float(mol_stability),
        "heavy_atom_mean": float(np.mean(heavy)) if heavy else 0.0,
        "valid_mols": n_valid,
        "unique_valid": len(set(valid_ids)),
        "stable_mols": int(round(mol_stability * n_total)),
        "stable_atoms": int(round(atom_stability * int(pos.shape[0]))),
        "total_atoms": int(pos.shape[0]),
    }


def evaluate_size(
    model: MoleculeGenerator,
    size: int,
    n_per_size: int,
    batch_size: int,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_parts = []
    type_parts = []
    batch_parts = []

    offset = 0
    remaining = n_per_size

    while remaining > 0:
        n_chunk = min(batch_size, remaining)
        pos, atom_types, batch_vec = generate_fixed_size(model, n_chunk, size)

        pos_parts.append(pos)
        type_parts.append(atom_types)
        batch_parts.append(batch_vec + offset)

        offset += n_chunk
        remaining -= n_chunk

    pos = torch.cat(pos_parts, dim=0)
    atom_types = torch.cat(type_parts, dim=0)
    batch_vec = torch.cat(batch_parts, dim=0)

    metrics = compute_callback_metrics(pos, atom_types, batch_vec)
    metrics["size"] = size
    return metrics, pos, atom_types, batch_vec


def print_table(rows: list[dict]) -> None:
    print()
    print(
        f"{'size':>6} {'n':>7} {'valid':>10} {'unique':>10} "
        f"{'atom_stab':>11} {'mol_stab':>10} {'valid_mols':>11} "
        f"{'uniq_valid':>11} {'stable_mols':>12} {'heavy_mean':>11}"
    )
    print("-" * 112)

    for row in rows:
        print(
            f"{str(row['size']):>6} "
            f"{row['n']:>7d} "
            f"{row['validity']:>10.4f} "
            f"{row['uniqueness']:>10.4f} "
            f"{row['atom_stability']:>11.4f} "
            f"{row['mol_stability']:>10.4f} "
            f"{row['valid_mols']:>11d} "
            f"{row['unique_valid']:>11d} "
            f"{row['stable_mols']:>12d} "
            f"{row['heavy_atom_mean']:>11.2f}"
        )


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)

    print(f"Loading W&B config for run {args.wandb_run_id} ...")
    config = load_config(args.wandb_run_id)

    model = build_model_from_wandb_config(config)
    model.to(device)
    model.eval()

    print(f"Loading generator_{args.wandb_variant}.pth from W&B ...")
    load_pretrained_generator(args.wandb_run_id, model, variant=args.wandb_variant)

    sizes = load_possible_sizes(args, config)
    if not sizes:
        raise RuntimeError("No molecule sizes found to evaluate.")

    batch_size = min(args.batch_size, args.n_per_size)
    if batch_size <= 0:
        raise ValueError("--batch_size and --n_per_size must be positive.")

    print(f"Evaluating sizes: {sizes}")
    print(f"Generating {args.n_per_size} molecules per size.")

    rows = []
    all_pos = []
    all_types = []
    all_batches = []
    global_offset = 0

    for size in sizes:
        print(f"Evaluating size {size} ...")
        metrics, pos, atom_types, batch_vec = evaluate_size(
            model=model,
            size=size,
            n_per_size=args.n_per_size,
            batch_size=batch_size,
        )

        rows.append(metrics)

        all_pos.append(pos)
        all_types.append(atom_types)
        all_batches.append(batch_vec + global_offset)
        global_offset += metrics["n"]

    overall = compute_callback_metrics(
        torch.cat(all_pos, dim=0),
        torch.cat(all_types, dim=0),
        torch.cat(all_batches, dim=0),
    )
    overall["size"] = "ALL"
    rows.append(overall)

    print_table(rows)


if __name__ == "__main__":
    main()