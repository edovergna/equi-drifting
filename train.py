import sys
from pathlib import Path

# Make sure the project root is in the Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import wandb
from model import (AtomTypeDistributionCallback, ChemicalValidityCallback,
                   DriftingMoleculeGenerator, EmbeddingMonitorCallback,
                   GeneratorCheckpointCallback, GradientMonitorCallback,
                   MoleculeVisualizationCallback, QM9DataModule,
                   SizeDistributionCallback, initialize_training_config, RiemannianDriftingMoleculeGenerator)
from model.wandb_utils import load_pretrained_generator
from parse_args import parse_args

def main(args: argparse.Namespace):

    device, precision, deterministic, benchmark = initialize_training_config(args)

    run = wandb.init(
        entity="equivariant-drifting",
        project="riemannian",
        group=args.group_tag,
        mode="offline" if args.offline else "online",
        config=vars(args),
    )

    try:
        datamodule = QM9DataModule(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            force_reload=args.force_reload,
            sample_frac=args.sample_frac,
        )

        generator_cfg = {
            "hidden_nf": args.hidden_dim,
            "n_layers": args.num_layers,
            "num_atom_types": 5,
            "num_bond_types": 5,
            "predict_bond_types": args.predict_bond_types,
            "pos_clamp": args.pos_clamp,
            "pos_clamp_type": args.pos_clamp_type,
            "c_pos_clamp": args.c_pos_clamp,
            "p_pos_clamp": args.p_pos_clamp,
            "norm_pos_clamp": args.norm_pos_clamp,
            "prior_pos_clamp": args.prior_pos_clamp,
        }

        drift_cfg = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "temperatures": args.temperatures,
            "loss_variant": args.loss_variant,
            "atom_type_temp": args.atom_type_temp,
            "sigma_r": args.sigma_r,
            "sigma_a": args.sigma_a,
            "eta_pos": args.eta_pos,
            "eta_type": args.eta_type,
            "weight_pos": args.weight_pos,
            "weight_type": args.weight_type,
        }

        model = RiemannianDriftingMoleculeGenerator(generator_cfg, drift_cfg)

        if args.wandb_run_id:
            print(f"Loading pretrained generator from wandb run: {args.wandb_run_id}")
            load_pretrained_generator(
                args.wandb_run_id, model, variant=args.wandb_variant
            )

        gen_ckpt = GeneratorCheckpointCallback(monitor="val_loss", mode="min")

        callbacks = [
            GradientMonitorCallback(),
            EmbeddingMonitorCallback(),
            MoleculeVisualizationCallback(
                n_molecules=4, bond_threshold=2.0, every_n_epochs=1
            ),
            ChemicalValidityCallback(),
            # SizeDistributionCallback(),
            # AtomTypeDistributionCallback(),
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
        trainer.test(model, datamodule=datamodule)

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        wandb.finish()
    except Exception:
        wandb.finish(exit_code=1)
        raise
    else:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
