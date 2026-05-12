import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a flow matching model on QM9.")
    # Data
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
        "--n_gen_molecules",
        type=int,
        default=64,
        help="Number of molecules to generate per forward pass during training and validation.",
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Fraction of each split to use (0 < sample_frac <= 1.0). Useful for quick iteration runs.",
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
        "--n_real_molecules",
        type=int,
        default=128,
        help="Number of real molecules per batch.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for data loading."
    )
    # Optimization
    parser.add_argument(
        "--max_epochs", type=int, default=120, help="Maximum number of training epochs."
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
    # Model Args
    parser.add_argument(
        "--predict_bond_types",
        action="store_true",
        help="Whether to predict bond types.",
    )
    parser.add_argument(
        "--predict_atom_types",
        action="store_true",
        help="Whether to predict atom types.",
    )
    parser.add_argument(
        "--use_feature_extractor",
        dest="use_feature_extractor",
        action="store_true",
        help="Use EPT feature extraction before computing drift loss.",
    )
    parser.add_argument(
        "--infer_types_from_pos",
        action="store_true",
        help=(
            "Infer atom types from generated positions, replacing EGNN-predicted atom "
            "types in the EPT feature extractor call. Method is set by --infer_method."
        ),
    )
    parser.add_argument(
        "--infer_method",
        type=str,
        default="heuristic",
        choices=["degree", "heuristic"],
        help=(
            "Atom-type inference method used when --infer_types_from_pos is set "
            "(and for molecule visualisation when atom types are unavailable). "
            "'degree': simple connectivity-degree mapping. "
            "'heuristic': QM9-specific rules using bond lengths + neighbourhood chemistry."
        ),
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.2],
        help="Temperature values for the drifting field (space-separated, e.g. --temperatures 0.02 0.05 0.2).",
    )
    parser.add_argument(
        "--loss_variant",
        type=str,
        default="norm_based",
        choices=["original", "inverse_attn", "norm_based"],
        help=(
            "Drift loss variant: 'original' (coupled attention weighting, single tau), "
            "'inverse_attn' (normalized attention weighting, multi-tau), "
            "'norm_based' (norm-difference kernel, multi-tau)."
        ),
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for the EGNN model.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of layers for the EGNN model.",
    )
    parser.add_argument(
        "--atom_type_temp",
        type=float,
        default=1.0,
        help="Temperature (tau) for Gumbel-softmax atom type sampling during generation.",
    )
    parser.add_argument(
        "--pos_clamp",
        type=float,
        default=100.0,
        help="Clamp generated atom positions to [-pos_clamp, pos_clamp] after centering (Angstroms).",
    )
    parser.add_argument(
        "--pos_clamp_type",
        type=str,
        default="hard",
        choices=["hard", "tanh", "geom"],
        help=(
            "Position clamping strategy: 'hard' (hard clamp to ±pos_clamp), "
            "'tanh' (tanh rescaling with norm_pos_clamp scale), "
            "'geom' (geometric rescale with c_pos_clamp and p_pos_clamp)."
        ),
    )
    parser.add_argument(
        "--c_pos_clamp",
        type=float,
        default=5.0,
        help="Scale parameter (Angstroms) for geom pos clamp: rescale = 1 / (1 + (|pos| / c)^p).",
    )
    parser.add_argument(
        "--p_pos_clamp",
        type=float,
        default=4.0,
        help="Power parameter for geom pos clamp: rescale = 1 / (1 + (|pos| / c)^p).",
    )
    parser.add_argument(
        "--norm_pos_clamp",
        type=float,
        default=10.0,
        help="Normalization scale (Angstroms) for tanh pos clamp: pos_out = pos_clamp * tanh(|pos| / norm_pos_clamp).",
    )
    parser.add_argument(
        "--prior_pos_clamp",
        type=float,
        default=4.0,
        help="Clamp prior position samples to [-prior_pos_clamp, prior_pos_clamp] standard deviations.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="euclidean",
        choices=["euclidean", "riemannian", "aligned"],
        help="Generator variant to train: 'euclidean' (EPT-based) or 'riemannian' (spherical atom types) or 'aligned'.",
    )
    # Wandb args
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
    # Lightning args
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
