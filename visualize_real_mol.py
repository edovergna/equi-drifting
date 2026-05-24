from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Center, Compose

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.datamodule import EncodeAtomTypesTransform, FullyConnectedTransform
from model.mol_utils.constants import _COV_RADII

ATOM_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
ATOM_COLORS = {
    1: "lightgray",
    6: "dimgray",
    7: "steelblue",
    8: "tomato",
    9: "limegreen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write XYZ and PNG for QM9 molecule index 2, expected H2O."
    )
    parser.add_argument("--root", default="data/QM9")
    parser.add_argument("--output_dir", default="outputs/qm9_h2o_index_2")
    parser.add_argument("--dataset_index", type=int, default=2)
    parser.add_argument("--force_reload", action="store_true")
    parser.add_argument("--bond_tolerance", type=float, default=0.45)
    parser.add_argument("--min_bond_distance", type=float, default=0.32)
    return parser.parse_args()


def load_qm9(root: str, force_reload: bool) -> QM9:
    return QM9(
        root,
        pre_transform=Compose(
            [Center(), FullyConnectedTransform(), EncodeAtomTypesTransform()]
        ),
        force_reload=force_reload,
    )


def write_xyz(path: Path, z: torch.Tensor, pos: torch.Tensor, comment: str) -> None:
    z = z.detach().cpu().long()
    pos = pos.detach().cpu()

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{pos.shape[0]}\n")
        f.write(f"{comment}\n")
        for atomic_number, xyz in zip(z.tolist(), pos.tolist()):
            symbol = ATOM_SYMBOLS.get(int(atomic_number), "X")
            f.write(f"{symbol} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}\n")


def should_draw_bond(
    zi: int,
    zj: int,
    distance: float,
    tolerance: float,
    min_distance: float,
) -> bool:
    if distance < min_distance:
        return False

    radius_i = _COV_RADII.get(zi)
    radius_j = _COV_RADII.get(zj)
    if radius_i is None or radius_j is None:
        return False

    return distance <= radius_i + radius_j + tolerance


def write_molecule_png(
    path: Path,
    z: torch.Tensor,
    pos: torch.Tensor,
    bond_tolerance: float,
    min_bond_distance: float,
) -> None:
    z_np = z.detach().cpu().long().numpy()
    pos_np = pos.detach().cpu().numpy()

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    for atomic_number in sorted(set(z_np.tolist())):
        mask = z_np == atomic_number
        ax.scatter(
            pos_np[mask, 0],
            pos_np[mask, 1],
            pos_np[mask, 2],
            c=ATOM_COLORS.get(int(atomic_number), "steelblue"),
            s=120,
            label=ATOM_SYMBOLS.get(int(atomic_number), "X"),
            depthshade=True,
            edgecolors="k",
            linewidths=0.3,
        )

    for i in range(len(pos_np)):
        for j in range(i + 1, len(pos_np)):
            distance = float(np.linalg.norm(pos_np[i] - pos_np[j]))
            if should_draw_bond(
                int(z_np[i]),
                int(z_np[j]),
                distance,
                bond_tolerance,
                min_bond_distance,
            ):
                ax.plot(
                    [pos_np[i, 0], pos_np[j, 0]],
                    [pos_np[i, 1], pos_np[j, 1]],
                    [pos_np[i, 2], pos_np[j, 2]],
                    "k-",
                    alpha=0.35,
                    linewidth=1.0,
                )

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=3))

    ax.tick_params(labelsize=7, pad=1)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=6, markerscale=0.7)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    xyz_dir = output_dir / "xyz"
    png_dir = output_dir / "png"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_qm9(args.root, args.force_reload)
    molecule = dataset[args.dataset_index]

    z = molecule.z
    pos = molecule.pos.float()

    write_xyz(
        xyz_dir / "h2o_index_2.xyz",
        z,
        pos,
        f"QM9 dataset index {args.dataset_index}",
    )

    write_molecule_png(
        png_dir / "h2o_index_2.png",
        z,
        pos,
        bond_tolerance=args.bond_tolerance,
        min_bond_distance=args.min_bond_distance,
    )

    formula_counts = {ATOM_SYMBOLS.get(int(v), "X"): z.tolist().count(int(v)) for v in z}
    print(f"Wrote XYZ to {xyz_dir / 'h2o_index_2.xyz'}")
    print(f"Wrote PNG to {png_dir / 'h2o_index_2.png'}")
    print(f"Atomic numbers: {z.tolist()}")
    print(f"Counts: {formula_counts}")


if __name__ == "__main__":
    main()