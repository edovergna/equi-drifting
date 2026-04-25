import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a flow matching model on QM9.")
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
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for the EGNN model.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of layers for the EGNN model.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--type_loss_weight", type=float, default=0.1, help="Weight for the type loss."
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
        default=10,
        help="Number of validation checks with no improvement before stopping.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
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
