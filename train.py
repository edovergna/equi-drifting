import sys
from pathlib import Path

# Make sure the project root is in the Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from model import (
    ChemicalValidityCallback,
    DriftingMoleculeGenerator,
    EmbeddingMonitorCallback,
    GradientMonitorCallback,
    MoleculeVisualizationCallback,
    QM9DataModule,
    initialize_training_config,
)
from parse_args import parse_args


def main(args: argparse.Namespace):

    device, precision, deterministic, benchmark = initialize_training_config(args)

    run = wandb.init(
        entity="equivariant-drifting",
        project="tests-kristian",
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

    generator_cfg = {
        "in_node_nf": 7,
        "hidden_nf": args.hidden_dim,
        "n_layers": args.num_layers,
        "num_atom_types": 5,
        "num_bond_types": 5,
        "predict_bond_types": args.predict_bond_types,
    }

    drift_cfg = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperatures": args.temperatures,
    }

    model = DriftingMoleculeGenerator(generator_cfg, drift_cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        checkpoint_callback,
        GradientMonitorCallback(),
        EmbeddingMonitorCallback(),
        MoleculeVisualizationCallback(
            n_molecules=4, bond_threshold=2.0, every_n_epochs=1
        ),
        ChemicalValidityCallback(),
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=args.max_epochs,
        devices=1,
        deterministic=deterministic,
        benchmark=benchmark,
        precision=precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=WandbLogger(experiment=run, save_dir="."),
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=datamodule)

    if checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best val_loss: {checkpoint_callback.best_model_score}")
    else:
        print("No best checkpoint found; testing with current model weights.")

    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
