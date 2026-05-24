"""Overfit a single EGNN to one QM9 molecule geometry from a fixed prior sample."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Center, Compose

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.datamodule import EncodeAtomTypesTransform, FullyConnectedTransform
from model.egnn import EGNN
from model.geometry import center_positions_per_graph
from model.sample_prior import get_dense_edge_index, sample_atom_dirichlet_noise

ATOM_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the single-molecule geometry overfit experiment.

    Returns:
        argparse.Namespace with experiment settings.
    """
    parser = argparse.ArgumentParser(
        description="Overfit one fixed prior sample to one QM9 molecule geometry."
    )
    parser.add_argument("--root", type=str, default="data/QM9")
    parser.add_argument("--output_dir", type=str, default="outputs/single_molecule_geometry")
    parser.add_argument("--molecule_index", type=int, default=0)
    parser.add_argument("--max_num_atoms", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--prior_pos_clamp", type=float, default=3.0)
    parser.add_argument("--log_every", type=int, default=100)
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


def seed_everything(seed: int) -> None:
    """Set random seeds for NumPy, PyTorch, and CUDA for reproducibility.

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qm9(root: str, force_reload: bool) -> QM9:
    """Load the QM9 dataset with fully-connected graph and atom-type transforms.

    Temporarily disables RDKit to avoid dependency issues during graph construction.

    Args:
        root: Root directory for the QM9 dataset.
        force_reload: Whether to ignore cached processed data and reprocess.

    Returns:
        QM9 dataset with Center, FullyConnectedTransform, and EncodeAtomTypesTransform applied.
    """
    rdkit_saved = {
        k: v
        for k, v in sys.modules.items()
        if k == "rdkit" or k.startswith("rdkit.")
    }
    for k in list(rdkit_saved):
        sys.modules[k] = None  # type: ignore[assignment]
    sys.modules.setdefault("rdkit", None)  # type: ignore[assignment]
    try:
        return QM9(
            root,
            pre_transform=Compose(
                [Center(), FullyConnectedTransform(), EncodeAtomTypesTransform()]
            ),
            force_reload=force_reload,
        )
    finally:
        for k in list(sys.modules):
            if sys.modules[k] is None and (k == "rdkit" or k.startswith("rdkit.")):
                del sys.modules[k]
        sys.modules.update(rdkit_saved)


def select_molecule(dataset: QM9, molecule_index: int, max_num_atoms: int | None):
    """Select a single QM9 molecule by filtered index.

    Args:
        dataset: The full QM9 dataset.
        molecule_index: Index into the filtered set of molecules passing the atom-count filter.
        max_num_atoms: Maximum number of atoms per molecule; None means no filter.

    Returns:
        A tuple of (dataset_index, molecule_data).
    """
    candidates = [
        i for i, data in enumerate(dataset)
        if max_num_atoms is None or data.num_nodes <= max_num_atoms
    ]
    if not candidates:
        raise ValueError(
            f"No QM9 molecule found with max_num_atoms={max_num_atoms}."
        )
    if molecule_index < 0 or molecule_index >= len(candidates):
        raise IndexError(
            f"molecule_index={molecule_index} is outside the filtered set "
            f"of {len(candidates)} molecules."
        )
    dataset_index = candidates[molecule_index]
    return dataset_index, dataset[dataset_index]


def write_xyz(path: Path, z: torch.Tensor, pos: torch.Tensor, comment: str) -> None:
    """Write atomic positions to an XYZ file.

    Args:
        path: Output file path.
        z: Atomic numbers tensor [n_atoms].
        pos: Atomic positions tensor [n_atoms, 3].
        comment: Comment line written as the second line of the file.
    """
    z_cpu = z.detach().cpu().long()
    pos_cpu = pos.detach().cpu()
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{pos_cpu.shape[0]}\n")
        f.write(f"{comment}\n")
        for atomic_number, xyz in zip(z_cpu.tolist(), pos_cpu.tolist()):
            symbol = ATOM_SYMBOLS.get(int(atomic_number), "X")
            f.write(f"{symbol} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}\n")


def main() -> None:
    """Run the single-molecule geometry overfit experiment."""
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    dataset = load_qm9(args.root, args.force_reload)
    dataset_index, molecule = select_molecule(
        dataset, args.molecule_index, args.max_num_atoms
    )

    target_pos = molecule.pos.to(device=device, dtype=torch.float32)
    target_pos = target_pos - target_pos.mean(dim=0, keepdim=True)
    z = molecule.z.to(device)
    n_nodes = int(molecule.num_nodes)

    x_prior = sample_atom_dirichlet_noise(n_nodes, num_atom_types=5, device=device)[0]
    pos_prior = torch.randn(n_nodes, 3, device=device).clamp(
        -args.prior_pos_clamp, args.prior_pos_clamp
    )
    pos_prior = pos_prior - pos_prior.mean(dim=0, keepdim=True)
    edge_index = get_dense_edge_index(n_nodes, device)
    batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=device)

    torch.save(
        {
            "x_prior": x_prior.detach().cpu(),
            "pos_prior": pos_prior.detach().cpu(),
            "edge_index": edge_index.detach().cpu(),
            "dataset_index": dataset_index,
            "seed": args.seed,
        },
        out_dir / "fixed_prior.pt",
    )
    torch.save(
        {
            "pos": target_pos.detach().cpu(),
            "z": z.detach().cpu(),
            "dataset_index": dataset_index,
        },
        out_dir / "target.pt",
    )
    write_xyz(out_dir / "prior.xyz", z, pos_prior, "fixed prior geometry")
    write_xyz(out_dir / "target.xyz", z, target_pos, "target QM9 geometry")

    model = EGNN(
        num_atom_types=5,
        num_blocks=args.num_layers,
        hidden_nf=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    metrics_path = out_dir / "metrics.csv"
    best_rmsd = float("inf")
    best_pos = None

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "rmsd", "max_abs_err"])
        writer.writeheader()

        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            pos_gen, _ = model(pos_prior, x_prior, edge_index)
            pos_gen = center_positions_per_graph(pos_gen, batch_vec)

            loss = F.mse_loss(pos_gen, target_pos)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                err = pos_gen - target_pos
                rmsd = err.pow(2).sum(dim=-1).mean().sqrt()
                max_abs_err = err.abs().max()

            if rmsd.item() < best_rmsd:
                best_rmsd = rmsd.item()
                best_pos = pos_gen.detach().cpu()
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "dataset_index": dataset_index,
                        "best_rmsd": best_rmsd,
                    },
                    out_dir / "best_model.pt",
                )

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = {
                    "step": step,
                    "loss": float(loss.item()),
                    "rmsd": float(rmsd.item()),
                    "max_abs_err": float(max_abs_err.item()),
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"step={step:06d} loss={row['loss']:.8f} "
                    f"rmsd={row['rmsd']:.6f} max_abs_err={row['max_abs_err']:.6f}"
                )

    final_pos = best_pos if best_pos is not None else pos_gen.detach().cpu()
    write_xyz(out_dir / "final.xyz", z.detach().cpu(), final_pos, "best generated geometry")
    print(f"dataset_index={dataset_index}")
    print(f"n_atoms={n_nodes}")
    print(f"best_rmsd={best_rmsd:.6f}")
    print(f"outputs={out_dir}")


if __name__ == "__main__":
    main()

