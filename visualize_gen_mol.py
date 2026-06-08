from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import MoleculeGenerator, QM9DataModule
from model.geometry import center_positions_per_graph
from model.mol_utils import batch_to_stability
from model.mol_utils.constants import _ATOM_NAMES
from model.sample_prior import compute_size_distribution
from model.wandb_utils import load_config, load_pretrained_generator
from matplotlib.ticker import MaxNLocator

_ATOM_COLORS = ["lightgray", "dimgray", "steelblue", "tomato", "limegreen"]

# TO BE UPDATED FOR NEW STRUCTURE

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load generator weights from W&B, generate molecules, "
            "filter unstable molecules, and export stable ones as XYZ + PNG files."
        )
    )
    parser.add_argument("--wandb_run_id", required=True)
    parser.add_argument("--wandb_variant", choices=["best", "final"], default="best")
    parser.add_argument("--output_dir", default="outputs/generated_molecules")
    parser.add_argument("--n_molecules", type=int, default=100)
    parser.add_argument(
        "--num_atoms",
        type=int,
        default=3,
        help=(
            "If set, generate all molecules with this atom count. "
            "Set to None in code if you want to sample from QM9 size distribution."
        ),
    )
    parser.add_argument("--root", default="data/QM9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--bond_threshold",
        type=float,
        default=1.42,
        help="Distance threshold used only for drawing bonds in PNG previews.",
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


def write_xyz(
    path: Path,
    atom_type_idx: torch.Tensor,
    pos: torch.Tensor,
    comment: str,
) -> None:
    atom_type_idx = atom_type_idx.detach().cpu().long()
    pos = pos.detach().cpu()

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{pos.shape[0]}\n")
        f.write(f"{comment}\n")
        for type_idx, xyz in zip(atom_type_idx.tolist(), pos.tolist()):
            symbol = _ATOM_NAMES[int(type_idx)]
            f.write(f"{symbol} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}\n")


def write_molecule_png(
    path: Path,
    atom_type_idx: torch.Tensor,
    pos: torch.Tensor,
    title: str,
    bond_threshold: float = 2.0,
) -> None:
    atom_type_idx = atom_type_idx.detach().cpu().long().numpy()
    pos = pos.detach().cpu().numpy()

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    for type_idx, (color, name) in enumerate(zip(_ATOM_COLORS, _ATOM_NAMES)):
        mask = atom_type_idx == type_idx
        if mask.any():
            ax.scatter(
                pos[mask, 0],
                pos[mask, 1],
                pos[mask, 2],
                c=color,
                s=120,
                label=name,
                depthshade=True,
                edgecolors="k",
                linewidths=0.3,
            )

    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if np.linalg.norm(pos[i] - pos[j]) < bond_threshold:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    [pos[i, 2], pos[j, 2]],
                    "k-",
                    alpha=0.25,
                    linewidth=0.8,
                )

    ax.set_box_aspect([1, 1, 1])

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=6, markerscale=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=3))

    ax.tick_params(labelsize=7, pad=1)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def generate(model: MoleculeGenerator, n_molecules: int, num_atoms: int | None):
    old_n_gen = model.n_gen_molecules
    model.n_gen_molecules = n_molecules

    x_prior, pos_prior, batch_vec, edge_index = model._sample_prior_batch(
        n_molecules=n_molecules,
        num_atoms=num_atoms,
    )

    pos_prior = center_positions_per_graph(pos_prior, batch_vec)
    gen_pos, gen_type_logits = model.generator(pos_prior, x_prior, edge_index)
    gen_pos = center_positions_per_graph(gen_pos, batch_vec)
    gen_atom_types = gen_type_logits.softmax(dim=-1).argmax(dim=-1)

    model.n_gen_molecules = old_n_gen
    return gen_pos, gen_atom_types, batch_vec


def is_stable_molecule(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
) -> tuple[bool, float, float]:
    pos_cpu = pos.detach().cpu()
    atom_types_cpu = atom_types.detach().cpu()

    single_batch_vec = torch.zeros(
        pos_cpu.shape[0],
        dtype=torch.long,
    )

    atom_stable_frac, mol_stable_frac = batch_to_stability(
        pos_cpu,
        atom_types_cpu,
        single_batch_vec,
    )

    return mol_stable_frac == 1.0, atom_stable_frac, mol_stable_frac


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    xyz_dir = output_dir / "xyz"
    png_dir = output_dir / "png"

    xyz_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)

    config = load_config(args.wandb_run_id)
    model = build_model_from_wandb_config(config)
    model.to(device)
    model.eval()

    load_pretrained_generator(
        args.wandb_run_id,
        model,
        variant=args.wandb_variant,
    )

    if args.num_atoms is None:
        dm = QM9DataModule(
            root=args.root,
            n_real_molecules=1,
            num_workers=0,
            min_num_atoms=config.get("min_num_atoms"),
            max_num_atoms=config.get("max_num_atoms"),
        )
        dm.setup("fit")
        sizes, probs = compute_size_distribution(dm.train_set)
        model.set_size_distribution(sizes, probs)

    gen_pos, gen_atom_types, batch_vec = generate(
        model,
        n_molecules=args.n_molecules,
        num_atoms=args.num_atoms,
    )

    n_written = 0

    for mol_idx in range(args.n_molecules):
        mask = batch_vec == mol_idx

        mol_pos = gen_pos[mask]
        mol_atom_types = gen_atom_types[mask]

        is_stable, atom_stable_frac, mol_stable_frac = is_stable_molecule(
            mol_pos,
            mol_atom_types,
        )

        if not is_stable:
            continue

        filename_stem = f"generated_stable_{n_written:03d}"

        write_xyz(
            xyz_dir / f"{filename_stem}.xyz",
            mol_atom_types,
            mol_pos,
            (
                f"wandb_run={args.wandb_run_id}, "
                f"variant={args.wandb_variant}, "
                f"source_molecule={mol_idx}, "
                f"atom_stable_frac={atom_stable_frac:.4f}, "
                f"mol_stable_frac={mol_stable_frac:.4f}"
            ),
        )

        write_molecule_png(
            png_dir / f"{filename_stem}.png",
            mol_atom_types,
            mol_pos,
            title=f"Stable generated molecule {n_written}",
            bond_threshold=args.bond_threshold,
        )

        n_written += 1

    print(
        f"Wrote {n_written} stable molecules out of {args.n_molecules} generated."
    )
    print(f"XYZ files: {xyz_dir}")
    print(f"PNG files: {png_dir}")


if __name__ == "__main__":
    main()