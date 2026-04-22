import argparse

import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from model.datamodule import QM9DataModule
from model.lit_modules import DriftingMoleculeGenerator
from parse_args import parse_args
from utils import get_device, set_seed


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
        force_download=args.force_download,
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
