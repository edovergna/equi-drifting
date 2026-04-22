import argparse
import random

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from model.datamodule import QM9DataModule
from model.lit_modules import DriftingMoleculeGenerator
from parse_args import parse_args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available torch.device (CUDA, then MPS, else CPU)."""

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        device = torch.device("mps")
    else:
        print("GPU is not available. Using CPU.")
        device = torch.device("cpu")

    return device


def main(args: argparse.Namespace):
    set_seed(args.seed)
    torch.set_float32_matmul_precision("medium")
    device = get_device()

    run = wandb.init(
        entity="equivariant-drifting",
        project="tests",
        group=args.group_tag,
        mode="offline" if args.offline else "online",
        config=vars(args),
    )

    datamodule = QM9DataModule(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force_reload=args.force_reload,
    )
    model = DriftingMoleculeGenerator(None, None).to(device)
    callbacks = []

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=1,
        deterministic=True,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=WandbLogger(experiment=run, save_dir="."),
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
