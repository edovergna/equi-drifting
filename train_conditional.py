"""Training entrypoint for conditional 3D conformer generation.

This script trains a ConditionalMoleculeGenerator on QM9: given fixed
atom types, the model learns to predict 3D atomic coordinates using a
position-only drift loss with Kabsch + Hungarian alignment.

Two modes:
  - Full training (default): full or sub-sampled QM9, standard split.
  - Overfitting (--overfit): tiny fixed subset to verify the model can
    memorise a handful of molecules before scaling up.

Usage examples
--------------
# Overfit on ~10 molecules of size 9 to confirm the model works:
python train_conditional.py --overfit --max_epochs 2000 --offline

# Scale training with default settings:
python train_conditional.py --max_epochs 500 --hidden_dim 256 --num_layers 9

# Train on small molecules only:
python train_conditional.py --min_num_atoms 3 --max_num_atoms 9

WandB project: "conditional-drifting" (separate from the joint model's project).
"""

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import wandb
from model import (
    ChemicalValidityCallback,
    GeneratorCheckpointCallback,
    GradientMonitorCallback,
    QM9DataModule,
    initialize_training_config,
)
from model.callbacks.chemical_validity_per_size import AtomSizeValidityCallback
from model.lit_module_conditional import ConditionalMoleculeGenerator
from parse_args_conditional import parse_args_conditional


def _build_datamodule(args) -> QM9DataModule:
    """Construct the QM9 DataModule for either overfit or full training.

    Overfit mode forces:
      - Single atom-count group (overfit_atom_count).
      - Tiny sample fraction (overfit_sample_frac).
      - Batch size = n_real_molecules (so one step = the full overfit set).
      - num_workers = 0 to avoid multiprocessing overhead on tiny data.
    """
    if args.overfit:
        return QM9DataModule(
            root=args.root,
            n_real_molecules=args.n_real_molecules,
            num_workers=0,
            force_reload=args.force_reload,
            sample_frac=args.overfit_sample_frac,
            min_num_atoms=args.overfit_atom_count,
            max_num_atoms=args.overfit_atom_count,
        )

    return QM9DataModule(
        root=args.root,
        n_real_molecules=args.n_real_molecules,
        num_workers=min(args.num_workers, args.n_real_molecules),
        force_reload=args.force_reload,
        sample_frac=args.sample_frac,
        max_num_atoms=args.max_num_atoms,
        min_num_atoms=args.min_num_atoms,
    )


def main(args):
    """Run conditional conformer generation training.

    Args:
        args: Parsed command-line arguments from parse_args_conditional().
    """
    device, precision, deterministic, benchmark = initialize_training_config(args)

    tags = ["conditional"]
    if args.overfit:
        tags.append("overfit")

    run = wandb.init(
        entity="equivariant-drifting",
        project="pos_dutch_daniel",
        group=args.group_tag,
        mode="offline" if args.offline else "online",
        config=vars(args),
        tags=tags,
    )

    try:
        datamodule = _build_datamodule(args)

        generator_cfg = {
            "hidden_nf": args.hidden_dim,
            "n_layers": args.num_layers,
            "aggr_type": args.aggr_type,
            "num_atom_types": args.num_atom_types,
            "tanh_coord_updates": args.tanh_coord_updates,
        }

        drift_cfg = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "p_sigma": args.position_sigma,
            "p_eta": args.position_eta,
            "scale_eucl": args.scale_euclidean,
            "eps": args.epsilon,
            "max_iter": args.max_iter,
            "p_tol": args.position_tol,
            "p_weight": args.position_weight,
            "t_weight": args.types_weight,
            "n_gen_molecules": args.n_gen_molecules,
            "num_atom_types": args.num_atom_types,
            "pct_start": args.pct_start,
            "div_factor": args.div_factor,
            "final_div_factor": args.final_div_factor,
            "chem_refinement": args.chem_refinement,
            "max_epochs": args.max_epochs,
            "start_frac_epoch": args.start_frac_epoch,
            "lambda_clash": args.lambda_clash,
            "lambda_valence_excess": args.lambda_valence_excess,
            "lambda_hydrogen_valence": args.lambda_hydrogen_valence,
            "clash_threshold": args.clash_threshold,
            "bond_temperature": args.bond_temperature,
            "end_sigma": args.end_sigma,
            "dynamic_sigma": args.dynamic_sigma,
        }

        model = ConditionalMoleculeGenerator(generator_cfg, drift_cfg)

        gen_ckpt = GeneratorCheckpointCallback(monitor="val_loss", mode="min")

        callbacks = [
            GradientMonitorCallback(),
            ChemicalValidityCallback(),
            AtomSizeValidityCallback(),
            gen_ckpt,
        ]

        # In overfit mode we care about seeing the training loss curve converge,
        # not about running a val epoch every step — slow it down.
        check_val_every_n = args.check_val_every_n_epoch
        if args.overfit:
            check_val_every_n = max(check_val_every_n, args.max_epochs // 20)

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=args.max_epochs,
            devices=1,
            deterministic=deterministic,
            benchmark=benchmark,
            precision=precision,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            check_val_every_n_epoch=check_val_every_n,
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
    args = parse_args_conditional()
    main(args)
