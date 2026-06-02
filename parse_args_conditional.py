"""Command line argument parser for conditional position generation training."""

import argparse


def parse_args_conditional():
    """Create and parse arguments for train_conditional.py.

    Returns:
        Populated argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a conditional conformer generator on QM9. "
        "Atom types are given as input; the model learns to predict 3D positions."
    )

    # --------------------------
    # Data Args
    # --------------------------
    parser.add_argument("--root", type=str, default="data/QM9")
    parser.add_argument("--force_reload", action="store_true")
    parser.add_argument(
        "--sample_frac", type=float, default=1.0,
        help="Fraction of QM9 to use (0 < x <= 1). Useful for quick iteration.",
    )
    parser.add_argument("--min_num_atoms", type=int, default=None)
    parser.add_argument("--max_num_atoms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_real_molecules", type=int, default=64,
        help="Batch size (number of real molecules per step).",
    )
    parser.add_argument(
        "--n_gen_molecules", type=int, default=64,
        help="Number of conditional molecules generated per training step. "
        "Should be >= n_real_molecules for good drift statistics.",
    )
    parser.add_argument("--num_workers", type=int, default=2)

    # --------------------------
    # Overfitting Args
    # --------------------------
    parser.add_argument(
        "--overfit", action="store_true",
        help="Enable overfitting mode: use a tiny fixed subset of QM9 to verify "
        "the model can memorise a handful of molecules. Validation checks less "
        "frequently and num_workers is forced to 0.",
    )
    parser.add_argument(
        "--overfit_atom_count", type=int, default=9,
        help="Filter QM9 to molecules with exactly this many atoms for overfitting.",
    )
    parser.add_argument(
        "--overfit_sample_frac", type=float, default=0.001,
        help="Fraction of the atom-count-filtered split to use for overfitting. "
        "~0.001 of 9-atom molecules gives roughly 10–15 training molecules.",
    )

    # --------------------------
    # Optimization Args
    # --------------------------
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument(
        "--pct_start", type=float, default=0.1,
        help="Fraction of steps for LR warm-up in OneCycleLR.",
    )
    parser.add_argument("--div_factor", type=float, default=25.0)
    parser.add_argument("--final_div_factor", type=float, default=1e4)

    # --------------------------
    # ConditionalEGNN Args
    # --------------------------
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--num_layers", type=int, default=4,
        help="Number of PosGCN layers (position-only blocks).",
    )
    parser.add_argument(
        "--aggr_type", type=str, default="sum", choices=["sum", "mean"],
    )
    parser.add_argument(
        "--tanh_coord_updates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num_atom_types", type=int, default=5)

    # --------------------------
    # Drift Loss Args
    # --------------------------
    parser.add_argument(
        "--position_sigma", type=float, default=1.0,
        help="Kernel bandwidth for position drift (ignored when --dynamic_sigma is set).",
    )
    parser.add_argument(
        "--dynamic_sigma", action="store_true",
        help="Set sigma = num_atoms - 1 per batch, tuning the kernel bandwidth "
        "to molecule size automatically.",
    )
    parser.add_argument(
        "--end_sigma", type=float, default=None,
        help="If set, cosine-anneal position_sigma towards this value.",
    )
    parser.add_argument(
        "--position_eta", type=float, default=1.0,
        help="Step size for position drift field.",
    )
    parser.add_argument(
        "--scale_euclidean", type=float, default=1.0,
        help="Scale multiplier on the Euclidean (position) loss term.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--chem_refinement", action="store_true",
        help="Enable chemical penalty loss in the final training epochs.",
    )

    # --------------------------
    # Chemical Refinement Args
    # --------------------------
    parser.add_argument("--start_frac_epoch", type=float, default=0.8)
    parser.add_argument("--lambda_clash", type=float, default=0.1)
    parser.add_argument("--lambda_valence_excess", type=float, default=0.1)
    parser.add_argument("--lambda_hydrogen_valence", type=float, default=0.1)
    parser.add_argument("--clash_threshold", type=float, default=0.7)
    parser.add_argument("--bond_temperature", type=float, default=0.1)

    # --------------------------
    # Alignment Args
    # --------------------------
    parser.add_argument(
        "--max_iter", type=int, default=10,
        help="Max Kabsch+Hungarian iterations per alignment call.",
    )
    parser.add_argument("--position_tol", type=float, default=1e-4)
    parser.add_argument(
        "--position_weight", type=float, default=1.0,
        help="Weight of position cost in Hungarian cost matrix.",
    )
    parser.add_argument(
        "--types_weight", type=float, default=1.0,
        help="Weight of type cost in Hungarian cost matrix. "
        "Types are fixed so this acts as a hard type-matching signal.",
    )

    # --------------------------
    # WandB Args
    # --------------------------
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--group_tag", type=str, default="conditional")

    # --------------------------
    # Lightning Args
    # --------------------------
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument(
        "--check_val_every_n_epoch", type=int, default=1,
        help="Validate every N epochs. Increased automatically in --overfit mode.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--precision", type=str, default="auto",
        choices=["auto", "32-true", "16-mixed", "bf16-mixed"],
    )

    return parser.parse_args()
