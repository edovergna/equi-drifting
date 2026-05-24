"""Overfit a single EGNN to multiple QM9 molecule geometries from a fixed prior batch."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch_geometric.datasets import QM9

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.egnn import EGNN
from model.geometry import center_positions_per_graph
from model.sample_prior import get_dense_edge_index, sample_atom_dirichlet_noise

from train_single import load_qm9, seed_everything, write_xyz


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the multi-molecule geometry overfit experiment.

    Returns:
        argparse.Namespace with experiment settings.
    """
    parser = argparse.ArgumentParser(
        description="Overfit one fixed prior batch to multiple QM9 geometries."
    )
    parser.add_argument("--root", type=str, default="data/QM9")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/many_molecule_geometry"
    )
    parser.add_argument("--n_molecules", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_num_atoms", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--prior_pos_clamp", type=float, default=3.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_xyz_count", type=int, default=12)
    parser.add_argument("--force_reload", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    """Resolve a device name string to a torch.device.

    Args:
        name: One of "auto", "cpu", "cuda", or "mps".

    Returns:
        The selected torch.device.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def select_molecules(
    dataset: QM9, start_index: int, n_molecules: int, max_num_atoms: int | None
) -> tuple[list[int], list]:
    """Select a contiguous slice of QM9 molecules from the atom-count-filtered set.

    Args:
        dataset: The full QM9 dataset.
        start_index: Starting offset within the filtered molecule list.
        n_molecules: Number of molecules to select.
        max_num_atoms: Maximum number of atoms per molecule; None means no filter.

    Returns:
        A tuple of (list of dataset indices, list of molecule data objects).
    """
    candidates = [
        i
        for i, data in enumerate(dataset)
        if max_num_atoms is None or data.num_nodes <= max_num_atoms
    ]
    selected = candidates[start_index : start_index + n_molecules]
    if len(selected) < n_molecules:
        raise ValueError(
            f"Requested {n_molecules} molecules from start_index={start_index}, "
            f"but only {len(selected)} are available after filtering."
        )
    return selected, [dataset[i] for i in selected]


def build_fixed_batch(
    molecules: list,
    device: torch.device,
    prior_pos_clamp: float,
) -> dict[str, torch.Tensor]:
    """Build a batched tensor dict of prior and target positions for all molecules.

    Args:
        molecules: List of QM9 molecule data objects.
        device: Device to place all tensors on.
        prior_pos_clamp: Maximum absolute value for clamping prior positions.

    Returns:
        Dict with keys x_prior, pos_prior, target_pos, batch_vec, z, edge_index, atom_counts.
    """
    xs = []
    prior_positions = []
    target_positions = []
    batch_parts = []
    z_parts = []
    edge_parts = []
    atom_counts = []
    offset = 0

    for mol_idx, mol in enumerate(molecules):
        n = int(mol.num_nodes)
        atom_counts.append(n)

        x_prior = sample_atom_dirichlet_noise(n, num_atom_types=5, device=device)[0]
        pos_prior = torch.randn(n, 3, device=device).clamp(
            -prior_pos_clamp, prior_pos_clamp
        )
        pos_prior = pos_prior - pos_prior.mean(dim=0, keepdim=True)

        target_pos = mol.pos.to(device=device, dtype=torch.float32)
        target_pos = target_pos - target_pos.mean(dim=0, keepdim=True)

        xs.append(x_prior)
        prior_positions.append(pos_prior)
        target_positions.append(target_pos)
        batch_parts.append(torch.full((n,), mol_idx, dtype=torch.long, device=device))
        z_parts.append(mol.z.to(device=device, dtype=torch.long))
        edge_parts.append(get_dense_edge_index(n, device) + offset)
        offset += n

    return {
        "x_prior": torch.cat(xs, dim=0),
        "pos_prior": torch.cat(prior_positions, dim=0),
        "target_pos": torch.cat(target_positions, dim=0),
        "batch_vec": torch.cat(batch_parts, dim=0),
        "z": torch.cat(z_parts, dim=0),
        "edge_index": torch.cat(edge_parts, dim=1),
        "atom_counts": torch.tensor(atom_counts, dtype=torch.long),
    }


def per_molecule_rmsd(
    pos_gen: torch.Tensor, target_pos: torch.Tensor, batch_vec: torch.Tensor
) -> torch.Tensor:
    """Compute per-molecule RMSD between generated and target positions.

    Args:
        pos_gen: Generated positions [total_nodes, 3].
        target_pos: Target positions [total_nodes, 3].
        batch_vec: Batch indices [total_nodes].

    Returns:
        RMSD values tensor of shape [n_molecules].
    """
    err2 = (pos_gen - target_pos).pow(2).sum(dim=-1)
    rmsds = []
    for mol_idx in range(int(batch_vec.max().item()) + 1):
        mask = batch_vec == mol_idx
        rmsds.append(err2[mask].mean().sqrt())
    return torch.stack(rmsds)


def per_molecule_loss(
    pos_gen: torch.Tensor, target_pos: torch.Tensor, batch_vec: torch.Tensor
) -> torch.Tensor:
    """Compute the mean per-molecule MSE loss between generated and target positions.

    Args:
        pos_gen: Generated positions [total_nodes, 3].
        target_pos: Target positions [total_nodes, 3].
        batch_vec: Batch indices [total_nodes].

    Returns:
        Scalar mean loss over all molecules.
    """
    err2 = (pos_gen - target_pos).pow(2).sum(dim=-1)
    losses = []
    for mol_idx in range(int(batch_vec.max().item()) + 1):
        mask = batch_vec == mol_idx
        losses.append(err2[mask].mean())
    return torch.stack(losses).mean()


def save_example_xyzs(
    out_dir: Path,
    z: torch.Tensor,
    target_pos: torch.Tensor,
    prior_pos: torch.Tensor,
    final_pos: torch.Tensor,
    batch_vec: torch.Tensor,
    count: int,
) -> None:
    """Write prior, target, and final XYZ files for the first `count` molecules.

    Args:
        out_dir: Output directory; an xyz_examples/ subdirectory is created inside it.
        z: Atomic numbers [total_nodes].
        target_pos: Target positions [total_nodes, 3].
        prior_pos: Prior positions [total_nodes, 3].
        final_pos: Final generated positions [total_nodes, 3].
        batch_vec: Batch indices [total_nodes].
        count: Maximum number of molecules to write.
    """
    examples_dir = out_dir / "xyz_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    n_graphs = min(count, int(batch_vec.max().item()) + 1)
    for mol_idx in range(n_graphs):
        mask = batch_vec == mol_idx
        write_xyz(
            examples_dir / f"mol_{mol_idx:03d}_prior.xyz",
            z[mask],
            prior_pos[mask],
            f"fixed prior molecule {mol_idx}",
        )
        write_xyz(
            examples_dir / f"mol_{mol_idx:03d}_target.xyz",
            z[mask],
            target_pos[mask],
            f"target molecule {mol_idx}",
        )
        write_xyz(
            examples_dir / f"mol_{mol_idx:03d}_final.xyz",
            z[mask],
            final_pos[mask],
            f"best generated molecule {mol_idx}",
        )


def main() -> None:
    """Run the multi-molecule geometry overfit experiment."""
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    dataset = load_qm9(args.root, args.force_reload)
    dataset_indices, molecules = select_molecules(
        dataset, args.start_index, args.n_molecules, args.max_num_atoms
    )
    batch = build_fixed_batch(molecules, device, args.prior_pos_clamp)

    torch.save(
        {
            "x_prior": batch["x_prior"].detach().cpu(),
            "pos_prior": batch["pos_prior"].detach().cpu(),
            "edge_index": batch["edge_index"].detach().cpu(),
            "batch_vec": batch["batch_vec"].detach().cpu(),
            "atom_counts": batch["atom_counts"],
            "dataset_indices": dataset_indices,
            "seed": args.seed,
        },
        out_dir / "fixed_prior.pt",
    )
    torch.save(
        {
            "pos": batch["target_pos"].detach().cpu(),
            "z": batch["z"].detach().cpu(),
            "batch_vec": batch["batch_vec"].detach().cpu(),
            "atom_counts": batch["atom_counts"],
            "dataset_indices": dataset_indices,
        },
        out_dir / "targets.pt",
    )

    model = EGNN(
        num_atom_types=5,
        num_blocks=args.num_layers,
        hidden_nf=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    metrics_path = out_dir / "metrics.csv"
    best_mean_rmsd = float("inf")
    best_pos = None

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "mean_rmsd",
                "median_rmsd",
                "max_rmsd",
                "max_abs_err",
            ],
        )
        writer.writeheader()

        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            pos_gen, _ = model(
                batch["pos_prior"], batch["x_prior"], batch["edge_index"]
            )
            pos_gen = center_positions_per_graph(pos_gen, batch["batch_vec"])

            loss = per_molecule_loss(pos_gen, batch["target_pos"], batch["batch_vec"])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rmsds = per_molecule_rmsd(
                    pos_gen, batch["target_pos"], batch["batch_vec"]
                )
                mean_rmsd = rmsds.mean()
                median_rmsd = rmsds.median()
                max_rmsd = rmsds.max()
                max_abs_err = (pos_gen - batch["target_pos"]).abs().max()

            if mean_rmsd.item() < best_mean_rmsd:
                best_mean_rmsd = mean_rmsd.item()
                best_pos = pos_gen.detach().cpu()
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "dataset_indices": dataset_indices,
                        "best_mean_rmsd": best_mean_rmsd,
                        "per_molecule_rmsd": rmsds.detach().cpu(),
                    },
                    out_dir / "best_model.pt",
                )

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = {
                    "step": step,
                    "loss": float(loss.item()),
                    "mean_rmsd": float(mean_rmsd.item()),
                    "median_rmsd": float(median_rmsd.item()),
                    "max_rmsd": float(max_rmsd.item()),
                    "max_abs_err": float(max_abs_err.item()),
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"step={step:06d} loss={row['loss']:.8f} "
                    f"mean_rmsd={row['mean_rmsd']:.6f} "
                    f"median_rmsd={row['median_rmsd']:.6f} "
                    f"max_rmsd={row['max_rmsd']:.6f}"
                )

    final_pos = best_pos if best_pos is not None else pos_gen.detach().cpu()
    torch.save(
        {
            "pos": final_pos,
            "z": batch["z"].detach().cpu(),
            "batch_vec": batch["batch_vec"].detach().cpu(),
            "atom_counts": batch["atom_counts"],
            "dataset_indices": dataset_indices,
        },
        out_dir / "final.pt",
    )
    save_example_xyzs(
        out_dir,
        batch["z"].detach().cpu(),
        batch["target_pos"].detach().cpu(),
        batch["pos_prior"].detach().cpu(),
        final_pos,
        batch["batch_vec"].detach().cpu(),
        args.save_xyz_count,
    )

    print(f"n_molecules={args.n_molecules}")
    print(f"total_atoms={int(batch['target_pos'].shape[0])}")
    print(f"best_mean_rmsd={best_mean_rmsd:.6f}")
    print(f"outputs={out_dir}")


if __name__ == "__main__":
    main()
