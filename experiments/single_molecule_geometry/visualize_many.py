from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from visualize_result import atom_colors, set_equal_axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize multi-molecule geometry overfit outputs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/many_molecule_geometry",
        help="Directory produced by train_many.py.",
    )
    parser.add_argument("--n_show", type=int, default=12)
    parser.add_argument("--prefix", type=str, default="many_geometry")
    return parser.parse_args()


def load_outputs(output_dir: Path) -> tuple[dict, dict]:
    targets_path = output_dir / "targets.pt"
    final_path = output_dir / "final.pt"
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing {targets_path}. Run train_many.py first.")
    if not final_path.exists():
        raise FileNotFoundError(f"Missing {final_path}. Run train_many.py first.")
    return (
        torch.load(targets_path, map_location="cpu"),
        torch.load(final_path, map_location="cpu"),
    )


def per_molecule_rows(targets: dict, final: dict) -> list[dict[str, float | int]]:
    target_pos = targets["pos"].float()
    final_pos = final["pos"].float()
    batch_vec = targets["batch_vec"].long()
    rows = []
    n_molecules = int(batch_vec.max().item()) + 1
    for mol_idx in range(n_molecules):
        mask = batch_vec == mol_idx
        err = final_pos[mask] - target_pos[mask]
        rmsd = err.pow(2).sum(dim=-1).mean().sqrt().item()
        rows.append(
            {
                "molecule": mol_idx,
                "dataset_index": int(targets["dataset_indices"][mol_idx]),
                "n_atoms": int(mask.sum().item()),
                "rmsd": rmsd,
                "max_abs_err": err.abs().max().item(),
            }
        )
    return rows


def write_rmsd_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["molecule", "dataset_index", "n_atoms", "rmsd", "max_abs_err"]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_grid(output_dir: Path, prefix: str, n_show: int) -> None:
    import matplotlib.pyplot as plt

    targets, final = load_outputs(output_dir)
    rows = per_molecule_rows(targets, final)
    write_rmsd_csv(output_dir / f"{prefix}_per_molecule_rmsd.csv", rows)

    target_pos = targets["pos"].float()
    final_pos = final["pos"].float()
    batch_vec = targets["batch_vec"].long()
    z = targets["z"].long()

    n_plots = min(n_show, len(rows))
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for plot_idx in range(n_plots):
        mask = batch_vec == plot_idx
        t_pos = target_pos[mask]
        f_pos = final_pos[mask]
        colors = atom_colors(z[mask])

        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection="3d")
        ax.scatter(
            t_pos[:, 0],
            t_pos[:, 1],
            t_pos[:, 2],
            c=colors,
            s=55,
            edgecolor="k",
            label="target",
        )
        ax.scatter(
            f_pos[:, 0],
            f_pos[:, 1],
            f_pos[:, 2],
            c=colors,
            s=24,
            marker="x",
            label="final",
        )
        for a, b in zip(t_pos, f_pos):
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                color="#666666",
                linewidth=0.6,
                alpha=0.45,
            )
        set_equal_axes(ax, t_pos, f_pos)
        ax.set_title(f"mol {plot_idx} | RMSD {rows[plot_idx]['rmsd']:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if n_plots:
        fig.axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_overlay_grid.png", dpi=200)
    plt.close(fig)

    print(output_dir / f"{prefix}_overlay_grid.png")
    print(output_dir / f"{prefix}_per_molecule_rmsd.csv")


def main() -> None:
    args = parse_args()
    plot_grid(Path(args.output_dir), args.prefix, args.n_show)


if __name__ == "__main__":
    main()

