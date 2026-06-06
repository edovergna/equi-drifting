"""PyTorch Lightning module for molecular generation using drift-based training.

This module implements MoleculeGenerator, a LightningModule that combines a Transformer
architecture with drift loss for generative modeling types.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

from ..losses.types_drift_loss import (
    TrainingDivergedException,
    compute_types_drift_loss,
)
from ..transformer import Transformer, make_noise_mask
from ..sample_prior import compute_size_distribution

from ..spherical_utils import (probs_to_sphere, sphere_to_probs)


class TypesGenerator(LightningModule):
    """PyTorch Lightning module for training and inference of molecule generators.

    Uses a Transformer backbone with drift loss to learn the distribution over molecular
    atom types.
    """
    _SAVE_COMPONENTS = ["generator"]
    _LOAD_COMPONENTS = ["generator"]

    def __init__(self, generator_cfg=None, drift_cfg=None):
        """Initialize the molecule generator with configuration dictionaries.

        Args:
            generator_cfg: Dict with Transformer hyperparameters
            drift_cfg: Dict with training hyperparameters (lr, weight_decay, sigma, eta, etc.)
        """
        super().__init__()

        default_generator_cfg = {
            "input_dim": 5,
            "max_num_atoms": 29,
            "embedding_dim": 128,
            "num_heads": 8,
            "num_layers": 6,
            "dropout": 0.1,
        }
        default_drift_cfg = {
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "sigma": 1.0,
            "eta": 1.0,
            "eps": 1e-8,
            "n_gen_molecules": 64,
            "num_atom_types": 5,
            "pct_start": 0.1,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
        }

        self.generator_cfg = {**default_generator_cfg, **(generator_cfg or {})}
        self.drift_cfg = {**default_drift_cfg, **(drift_cfg or {})}
        self.save_hyperparameters(
            {"generator_cfg": self.generator_cfg, "drift_cfg": self.drift_cfg}
        )

        if self.drift_cfg.get("end_sigma", None) is not None:
            self.end_sigma = self.drift_cfg["end_sigma"]
            self.sigma_start = self.drift_cfg.get("sigma", 1.0)
            self.anneal_sigma = True
        else:
            self.anneal_sigma = False

        self.generator = self._init_generator(self.generator_cfg)

        self.n_gen_molecules = self.drift_cfg.get("n_gen_molecules", 64)
        self.max_num_atoms = self.drift_cfg.get("max_num_atoms", 29)
        self.num_types = self.drift_cfg.get("num_atom_types", 5)

        self.max_epochs = self.drift_cfg.get("max_epochs", 100)

        self.eps = self.drift_cfg.get("eps", 1e-8)

        self._norm_rescale_grad: float | None = None

    def _init_generator(self, cfg) -> Transformer:
        """Instantiate the Transformer model from generator configuration.

        Args:
            cfg: Generator configuration dictionary.

        Returns:
            An initialized Transformer model.
        """
        return Transformer(
            input_dim=cfg["input_dim"],
            max_num_atoms=cfg["max_num_atoms"],
            embedding_dim=cfg["embedding_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )

    def _init_size_distribution(self) -> None:
        """Compute molecule size distribution from the training dataset.

        Raises:
            RuntimeError: If size distribution cannot be computed from trainer/datamodule.
        """
        if self.trainer is not None and self.trainer.datamodule is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, "train_set") and dm.train_set is not None:
                sizes, probs = compute_size_distribution(dm.train_set)
                self._size_values = sizes
                self._size_probs = probs
                return
        raise RuntimeError(
            "Atom size distribution not set. Call set_size_distribution() before sampling, "
            "or ensure the model is bound to a trainer with a QM9DataModule."
        )
    
    def _sample_prior_batch(
        self, num_atoms
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of molecules

        Args:
            num_atoms: Number of atoms per molecule.

        Returns:
            Tuple of (noise_types, noise_mask).
        """
        noise = torch.randn(self.n_gen_molecules, self.max_num_atoms, self.num_types, device=self.device)
        noise_mask = make_noise_mask(self.max_num_atoms, num_atoms, batch_size=self.n_gen_molecules)
        noise_mask = noise_mask.to(self.device)

        return noise, noise_mask

    def _forward(self, batch, num_atoms):
        """Shared forward pass: sample prior → Transformer → sphere embeddings.

        Args:
            batch: Unused; kept for a uniform training-step signature.
            num_atoms: Number of atoms per molecule.

        Returns:
            gen_types_sphere.
        """
        # Sample from the prior distribution
        noise, noise_mask = self._sample_prior_batch(num_atoms)

        # Generate molecule with Transformer
        gen_types_logits = self.generator(noise, noise_mask)

        # Get relevant logits
        relevant_logits = gen_types_logits[noise_mask].reshape(self.n_gen_molecules, num_atoms, self.num_types)

        # Turn x_logits into probabilites
        gen_types_prob = F.softmax(relevant_logits, dim=-1)

        # And project to the sphere
        gen_types_sphere = probs_to_sphere(gen_types_prob, self.eps)

        return gen_types_sphere

    def on_after_backward(self):
        """Log gradient-related metrics after backpropagation."""
        if self._norm_rescale_grad is not None:
            self.log(
                "geom/norm_rescale_grad",
                self._norm_rescale_grad,
                on_step=True,
                on_epoch=False,
            )

    def _batch_num_atoms(self, batch) -> int:
        """Extract the number of atoms in the first molecule of a batch.

        Args:
            batch: PyTorch Geometric data batch with ptr attribute.

        Returns:
            Number of atoms in the first molecule.
        """
        return int((batch.ptr[1] - batch.ptr[0]).item())
    
    def on_train_epoch_start(self):
        """Apply cosine annealing to sigma values at the start of each training epoch."""
        if self.anneal_sigma:
            progress = self.current_epoch / max(1, self.trainer.max_epochs - 1)

            self.drift_cfg["sigma"] = (
                self.end_sigma
                + 0.5
                * (self.sigma_start - self.end_sigma)
                * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            )

            self.log("drift/sigma", self.drift_cfg["sigma"])


    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: PyTorch Geometric data batch from the dataloader.
            batch_idx: Index of the batch.

        Returns:
            Scalar loss tensor for backpropagation.
        """
        num_atoms = self._batch_num_atoms(batch)
        gen_types_sphere = self._forward(batch, num_atoms)

        real_types = batch.real_atom_types

        try:
            loss, stats = compute_types_drift_loss(
                gen_types_sphere, real_types, num_atoms, cfg=self.drift_cfg
            )
        except TrainingDivergedException as e:
            self.print(f"\n[Step {self.global_step}] {e}\nStopping training.")
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        bs = self.n_gen_molecules
        self.log("train_loss", loss, batch_size=bs, on_step=True, on_epoch=True)

        hist_stats = {k: v for k, v in stats.items() if isinstance(v, wandb.Histogram)}
        for key, val in stats.items():
            if key in hist_stats:
                continue
            self.log(
                f"drift_train/{key}", val, batch_size=bs, on_step=True, on_epoch=False
            )
        if hist_stats and hasattr(self.logger, "experiment"):
            try:
                self.logger.experiment.log(
                    {f"drift_train/{k}": v for k, v in hist_stats.items()},
                    commit=False,
                )
            except Exception:
                pass

        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.

        Args:
            batch: PyTorch Geometric data batch from the dataloader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary of generated and real positions/types for downstream callbacks.
        """
        num_atoms = self._batch_num_atoms(batch)
        gen_types_sphere = self._forward(batch, num_atoms)

        real_types = batch.real_atom_types

        val_loss, stats = compute_types_drift_loss(
            gen_types_sphere, real_types, num_atoms, cfg=self.drift_cfg
        )

        bs = self.n_gen_molecules
        self.log(
            "val_loss",
            val_loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        hist_stats = {k: v for k, v in stats.items() if isinstance(v, wandb.Histogram)}
        for key, val in stats.items():
            if key in hist_stats:
                continue
            self.log(
                f"drift_val/{key}",
                val,
                batch_size=bs,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if hist_stats:
            self._val_hist_stats = {f"drift_val/{k}": v for k, v in hist_stats.items()}

        # Project sphere embeddings back to probabilities
        with torch.no_grad():
            gen_types_prob = sphere_to_probs(gen_types_sphere, self.eps)

        return {
            "gen_atom_types": gen_types_prob.detach().cpu().argmax(dim=-1),
            "real_atom_types": batch.real_atom_types.detach().cpu(),
        }

    def on_validation_epoch_end(self):
        """Log histogram statistics to wandb at the end of validation."""
        hist_stats = getattr(self, "_val_hist_stats", {})
        if (
            hist_stats
            and hasattr(self, "logger")
            and hasattr(self.logger, "experiment")
        ):
            try:
                self.logger.experiment.log(hist_stats, commit=False)
            except Exception:
                pass
        self._val_hist_stats = {}

    def test_step(self, batch, batch_idx):
        """Perform a single test step.

        Args:
            batch: PyTorch Geometric data batch from the dataloader.
            batch_idx: Index of the batch.

        Returns:
            Scalar test loss tensor.
        """
        num_atoms = self._batch_num_atoms(batch)
        gen_types_sphere = self._forward(batch, num_atoms)
        real_types = batch.real_atom_types

        test_loss, _ = compute_types_drift_loss(
            gen_types_sphere, real_types, num_atoms, cfg=self.drift_cfg
        )

        bs = self.n_gen_molecules
        self.log(
            "test_loss",
            test_loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return test_loss

    def save_individual_components(self, save_path: str) -> None:
        """Save generator model weights to disk.

        Args:
            save_path: Directory path where the generator.pth file will be saved.
        """
        torch.save(self.generator.state_dict(), f"{save_path}/generator.pth")

    def load_individual_components(self, folder_path) -> None:
        """Load generator model weights from disk.

        Args:
            folder_path: Directory path containing the generator.pth file.
        """
        folder_path = Path(folder_path)
        self.generator.load_state_dict(
            torch.load(folder_path / "generator.pth", map_location=self.device)
        )

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and lr_scheduler configuration for PyTorch Lightning.
        """
        optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.drift_cfg["lr"],
            weight_decay=self.drift_cfg["weight_decay"],
            eps=self.eps,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.drift_cfg["lr"],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.drift_cfg["pct_start"],
            anneal_strategy="cos",
            div_factor=self.drift_cfg["div_factor"],
            final_div_factor=self.drift_cfg["final_div_factor"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }