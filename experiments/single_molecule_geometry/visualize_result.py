"""Visualize single-molecule geometry overfit results from train_single.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ATOM_COLORS = {
    1: "#d9d9d9",  # H
    6: "#222222",  # C
    7: "#3050f8",  # N
    8: "#ff0d0d",  # O
    9: "#90e050",  # F
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the single-molecule geometry visualization.

    Returns:
        argparse.Namespace with visualization settings.
    """
    parser = argparse.ArgumentParser(
        description="Visualize single-molecule geometry overfit outputs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/single_molecule_geometry",
        help="Directory produced by train_single.py.",
    )
    parser.add_argument("--prefix", type=str, default="geometry")
    return parser.parse_args()


def load_outputs(output_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load target and final geometry tensors from a train_single.py output directory.

    Args:
        output_dir: Directory containing target.pt and final.xyz.

    Returns:
        A tuple of (target_pos, final_pos, z) tensors.
    """
    target = torch.load(output_dir / "target.pt", map_location="cpu")
    final_xyz = output_dir / "final.xyz"
    if not final_xyz.exists():
        raise FileNotFoundError(f"Missing {final_xyz}. Run train_single.py first.")

    target_pos = target["pos"].float()
    z = target["z"].long()
    final_pos = read_xyz_positions(final_xyz).float()

    if final_pos.shape != target_pos.shape:
        raise ValueError(
            f"Shape mismatch: final={tuple(final_pos.shape)} "
            f"target={tuple(target_pos.shape)}"
        )
    return target_pos, final_pos, z


def read_xyz_positions(path: Path) -> torch.Tensor:
    """Parse atomic positions from an XYZ file.

    Args:
        path: Path to an XYZ-format file.

    Returns:
        Tensor of shape [n_atoms, 3] with xyz coordinates.
    """
    rows = path.read_text(encoding="utf-8").splitlines()[2:]
    coords = []
    for row in rows:
        parts = row.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return torch.tensor(coords)


def set_equal_axes(ax, *positions: torch.Tensor) -> None:
    """Set equal aspect ratio 3D axes bounds to encompass all given position tensors.

    Args:
        ax: Matplotlib 3D axes object.
        *positions: Variable number of position tensors [n, 3].
    """
    stacked = torch.cat([p.detach().cpu() for p in positions], dim=0)
    mins = stacked.min(dim=0).values
    maxs = stacked.max(dim=0).values
    center = (mins + maxs) / 2
    radius = float((maxs - mins).max().item() / 2)
    radius = max(radius, 1e-3)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def atom_colors(z: torch.Tensor) -> list[str]:
    """Map atomic numbers to hex color strings.

    Args:
        z: Tensor of atomic numbers [n_atoms].

    Returns:
        List of hex color strings, one per atom.
    """
    return [ATOM_COLORS.get(int(v), "#aa00ff") for v in z.tolist()]


def plot_geometry(output_dir: Path, prefix: str) -> None:
    """Render and save side-by-side and overlay 3D geometry plots for a single molecule.

    Args:
        output_dir: Directory containing train_single.py outputs.
        prefix: Filename prefix for saved PNG files.
    """
    import matplotlib.pyplot as plt

    target_pos, final_pos, z = load_outputs(output_dir)
    colors = atom_colors(z)
    err = final_pos - target_pos
    rmsd = err.pow(2).sum(dim=-1).mean().sqrt().item()
    max_abs_err = err.abs().max().item()

    fig = plt.figure(figsize=(12, 5))
    for idx, (title, pos) in enumerate(
        [("Target", target_pos), ("Final", final_pos)], start=1
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=90, edgecolor="k")
        ax.set_title(title)
        set_equal_axes(ax, target_pos, final_pos)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle(f"Single molecule geometry overfit | RMSD={rmsd:.4f}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_side_by_side.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(
        target_pos[:, 0],
        target_pos[:, 1],
        target_pos[:, 2],
        c=colors,
        s=95,
        edgecolor="k",
        label="target",
    )
    ax.scatter(
        final_pos[:, 0],
        final_pos[:, 1],
        final_pos[:, 2],
        c=colors,
        s=35,
        marker="x",
        label="final",
    )
    for a, b in zip(target_pos, final_pos):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color="#666666",
            linewidth=0.8,
            alpha=0.5,
        )
    set_equal_axes(ax, target_pos, final_pos)
    ax.set_title(f"Overlay | RMSD={rmsd:.4f}, max_abs_err={max_abs_err:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_overlay.png", dpi=200)
    plt.close(fig)

    print(output_dir / f"{prefix}_side_by_side.png")
    print(output_dir / f"{prefix}_overlay.png")


def main() -> None:
    """Run the single-molecule geometry visualization."""
    args = parse_args()
    plot_geometry(Path(args.output_dir), args.prefix)


if __name__ == "__main__":
    main()

