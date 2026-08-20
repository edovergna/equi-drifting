"""Command line argument parser for QM9 drift model training."""

import argparse


def parse_args():
    """Create and parse the command line arguments used by training scripts.

    Returns:
        A populated argparse.Namespace with training, dataset, and logging options.
    """
    parser = argparse.ArgumentParser(description="Train a flow matching model on QM9.")

    # --------------------------
    # Data Args
    # --------------------------
    parser.add_argument(
        "--root",
        type=str,
        default="data/QM9",
        help="Root directory for the QM9 dataset.",
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Whether to force reload the QM9 dataset (required after modifying pre_transform).",
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of each split to use (0 < sample_frac <= 1.0). Useful for quick iteration runs.",
    )
    parser.add_argument(
        "--min_num_atoms",
        type=int,
        default=None,
        help="Keep only molecules with at least this many atoms (inclusive). None means no filter."
    )
    parser.add_argument(
        "--max_num_atoms",
        type=int,
        default=None,
        help="Keep only molecules with at most this many atoms (inclusive). None means no filter.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--n_gen_molecules",
        type=int,
        default=64,
        help="Number of molecules to generate per forward pass during training and validation.",
    )
    parser.add_argument(
        "--n_real_molecules",
        type=int,
        default=128,
        help="Number of real molecules per batch.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for data loading."
    )

    # --------------------
    # Optimization Args
    # --------------------
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-5,
        help="Weight decay for the optimizer.",
    )

    # ------------------
    # EGNN Args
    # ------------------
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for the EGNN model.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=9,
        help="Number of layers for the EGNN model.",
    )
    parser.add_argument(
        "--aggr_type",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help="Aggregation method for EGNN coordinate updates.",
    )
    parser.add_argument(
        "--attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether EGNN model uses attention."
    )
    parser.add_argument(
        "--tanh_coord_updates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use tanh coordinate updates in EGNN."
    )
    parser.add_argument(
        "--num_atom_types",
        type=int,
        default=5,
        help="How many atom types are possible to predict."
    )

    # -----------------------
    # Drift Loss Args
    # -----------------------
    parser.add_argument(
        "--position_sigma",
        type=float,
        default=1.0,
        help="Sigma value for the drifting field of positions.",
    )
    parser.add_argument(
        "--types_sigma",
        type=float,
        default=1.0,
        help="Sigma value for the drifting field of types."
    )
    parser.add_argument(
        "--end_sigma",
        type=float,
        default=None,
        help="If set, then sigma values will be annealed to the specified end value using cosine annealing."
    )
    parser.add_argument(
        "--position_eta",
        type=float,
        default=1.0,
        help="Step-size for the drifting field of positions."
    )
    parser.add_argument(
        "--types_eta",
        type=float,
        default=1.0,
        help="Step-size for the drifting field of types."
    )
    parser.add_argument(
        "--scale_euclidean",
        type=float,
        default=1.0,
        help="Scale Euclidean loss within combined loss."
    )
    parser.add_argument(
        "--scale_spherical",
        type=float,
        default=1.0,
        help="Scale atom-type loss (spherical or Euclidean) within combined loss."
    )
    parser.add_argument(
        "--spherical_space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use spherical geometry for atom types. Disable with "
            "--no-spherical_space to use probability-space Euclidean geometry."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--chem_refinement",
        action="store_true",
        help="When activated, towards end of training, chemical losses will be used for refinement."
    )

    # -------------------------
    # Chemical Refinement Args
    # -------------------------
    parser.add_argument(
        "--start_frac_epoch",
        type=float,
        default=0.8,
        help="From which fraction of epochs onwards, chemical refinement will be used."
    )
    parser.add_argument(
        "--lambda_clash",
        type=float,
        default=0.1,
        help="Scale for loss of clash loss."
    )
    parser.add_argument(
        "--lambda_valence_excess",
        type=float,
        default=0.1,
        help="Scale for loss of excess valence."
    )
    parser.add_argument(
        "--lambda_hydrogen_valence",
        type=float,
        default=0.1,
        help="Scale for loss of hydrogen valence"
    )
    parser.add_argument(
        "--clash_threshold",
        type=float,
        default=0.7,
        help="Threshold to be used in clash loss."
    )
    parser.add_argument(
        "--bond_temperature",
        type=float,
        default=0.1,
        help="Temperature used in bond loss."
    )

    # ------------------------------------------------------------------
    # Legacy alignment args (accepted for old scripts/configs, but unused)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--max_iter",
        type=int,
        default=10,
        help=(
            "Deprecated compatibility option; the drift loss does not align "
            "molecules."
        ),
    )
    parser.add_argument(
        "--position_tol",
        type=float,
        default=1e-4,
        help=(
            "Deprecated compatibility option; the drift loss does not align "
            "molecules."
        ),
    )
    parser.add_argument(
        "--position_weight",
        type=float,
        default=1.0,
        help=(
            "Deprecated compatibility option; the drift loss does not permute "
            "atoms."
        ),
    )
    parser.add_argument(
        "--types_weight",
        type=float,
        default=1.0,
        help=(
            "Deprecated compatibility option; the drift loss does not permute "
            "atoms."
        ),
    )

    # --------------------
    # Wandb Args
    # --------------------
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Wandb run ID to load pretrained generator weights from before training.",
    )
    parser.add_argument(
        "--wandb_variant",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Which saved checkpoint to load from the given wandb run (best or final).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Whether to log to Weights & Biases in offline mode.",
    )
    parser.add_argument(
        "--group_tag",
        type=str,
        default="default_group",
        help="Group tag for Weights & Biases logging.",
    )

    # -------------------
    # Lightning args
    # -------------------
    parser.add_argument(
        "--log_every_n_steps", type=int, default=10, help="Log every n steps."
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="Check validation every n epochs.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=40,
        help="Number of validation checks with no improvement before stopping.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum absolute improvement in val_loss to reset patience.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory where model checkpoints are saved.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic kernels (slower, more reproducible).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "32-true", "16-mixed", "bf16-mixed"],
        help="Trainer precision mode. Use auto to pick fast safe defaults.",
    )
    return parser.parse_args()
