"""Training entrypoint and experiment orchestration for QM9 drifting models.

This module configures the dataset, model, callbacks, and PyTorch Lightning trainer
for end-to-end model training, validation, and optional testing.
"""

import sys
import traceback
from pathlib import Path

# Make sure the project root is in the Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import wandb
from model import (
    TypesGenerator,
    GeneratorCheckpointCallback,
    GradientMonitorCallback,
    QM9DataModule,
    SizeDistributionCallback,
    initialize_training_config,
)
from model.wandb_utils import load_pretrained_generator
from parse_args import parse_args


def main(args: argparse.Namespace):
    """Run training for the QM9 drift model using the provided CLI args.

    Args:
        args: Parsed command line arguments describing data paths, model
            hyperparameters, trainer settings, and logging options.
    """

    device, precision, deterministic, benchmark = initialize_training_config(args)

    run = wandb.init(
        entity="equivariant-drifting",
        project="drifting-for-types",
        group=args.group_tag,
        mode="offline" if args.offline else "online",
        config=vars(args),
    )

    try:
        datamodule = QM9DataModule(
            root=args.root,
            n_real_molecules=args.n_real_molecules,
            num_workers=min(args.num_workers, args.n_real_molecules),
            force_reload=args.force_reload,
            sample_frac=args.sample_frac,
            max_num_atoms=args.max_num_atoms,
            min_num_atoms=args.min_num_atoms,
        )
        generator_cfg = {
            "input_dim": args.num_atom_types,
            "max_num_atoms": args.max_num_atoms,
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        }

        drift_cfg = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "sigma": args.sigma,
            "eta": args.eta,
            "eps": args.epsilon,
            "n_gen_molecules": args.n_gen_molecules,
            "num_atom_types": args.num_atom_types,
            "max_epochs": args.max_epochs,
            "end_sigma": args.end_sigma,
            "max_num_atoms": args.max_num_atoms,
        }

        model = TypesGenerator(generator_cfg, drift_cfg)

        if args.wandb_run_id:
            print(f"Loading pretrained generator from wandb run: {args.wandb_run_id}")
            load_pretrained_generator(
                args.wandb_run_id, model, variant=args.wandb_variant
            )

        gen_ckpt = GeneratorCheckpointCallback(monitor="val_loss", mode="min")

        callbacks = [
            GradientMonitorCallback(),
            gen_ckpt,
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
        gen_ckpt.load_best_weights(model)
        if len(datamodule.test_set) > 0:
            trainer.test(model, datamodule=datamodule)
        else:
            print("Skipping test: test set is empty after atom-count filtering.")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        wandb.finish()
    except Exception:
        print("\nAn error occurred during training:")
        traceback.print_exc()
        wandb.finish(exit_code=1)
        raise
    else:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)