import sys
import traceback
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
from model import (
    AtomTypeDistributionCallback,
    ChemicalValidityCallback,
    MoleculeGenerator,
    GeneratorCheckpointCallback,
    GradientMonitorCallback,
    MoleculeVisualizationCallback,
    QM9DataModule,
    SizeDistributionCallback,
    initialize_training_config,
)
from model.wandb_utils import load_pretrained_generator
from parse_args import parse_args

def main(args: argparse.Namespace):

    device, precision, deterministic, benchmark = initialize_training_config(args)

    run = wandb.init(
        entity="equivariant-drifting",
        project="aligned-drifting",
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
            "hidden_nf": args.hidden_dim,
            "n_layers": args.num_layers,
            "aggr_type": args.aggr_type,
            "num_atom_types": args.num_atom_types,
            "tanh_coord_updates": args.tanh_coord_updates,
            "attention": args.attention,
        }

        drift_cfg = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "p_sigma": args.position_sigma,
            "t_sigma": args.types_sigma,
            "p_eta": args.position_eta,
            "t_eta": args.types_eta,
            "scale_eucl": args.scale_euclidean,
            "scale_spher": args.scale_spherical,
            "eps": args.epsilon,
            "max_iter": args.max_iter,
            "p_tol": args.position_tol,
            "p_weight": args.position_weight,
            "t_weight": args.types_weight,
            "n_gen_molecules": args.n_gen_molecules,
            "num_atom_types": args.num_atom_types,
        }

        model = MoleculeGenerator(generator_cfg, drift_cfg)

        if args.wandb_run_id:
            print(f"Loading pretrained generator from wandb run: {args.wandb_run_id}")
            load_pretrained_generator(
                args.wandb_run_id, model, variant=args.wandb_variant
            )

        gen_ckpt = GeneratorCheckpointCallback(monitor="val_loss", mode="min")

        callbacks = [
            GradientMonitorCallback(),
            MoleculeVisualizationCallback(
                n_molecules=min(4, args.n_real_molecules),
                bond_threshold=2.0,
                every_n_epochs=1,
            ),
            ChemicalValidityCallback(),
            # SizeDistributionCallback(),
            # AtomTypeDistributionCallback(),
            gen_ckpt,
        ]

        trainer = pl.Trainer(
            accelerator="cpu",
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
