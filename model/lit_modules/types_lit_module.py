"""PyTorch Lightning module for molecular generation using drift-based training.

This module implements MoleculeGenerator, a LightningModule that combines an EGNN
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
    compute_types_drift_loss
)
from ..egnn import EGNN
from ..geometry import center_positions_per_graph, per_graph_center_norms
from ..sample_prior import compute_size_distribution, sample_prior_batch

from ..spherical_utils import (probs_to_sphere, sphere_to_probs)


class TypesGenerator(LightningModule):
    """PyTorch Lightning module for training and inference of molecule generators.

    Uses an EGNN backbone with drift loss to learn the distribution over molecular
    geometries and atom types.
    """
    _SAVE_COMPONENTS = ["generator"]
    _LOAD_COMPONENTS = ["generator"]

    def __init__(self, generator_cfg=None, drift_cfg=None):
        """Initialize the molecule generator with configuration dictionaries.

        Args:
            generator_cfg: Dict with EGNN hyperparameters (hidden_nf, n_layers, etc.)
            drift_cfg: Dict with training hyperparameters (lr, weight_decay, sigma, eta, etc.)
        """
        super().__init__()

        default_generator_cfg = {
            "hidden_nf": 128,
            "n_layers": 2,
            "num_atom_types": 5,
            "aggr_type": "sum",
            "tanh_coord_updates": True,
            "attention": True
        }
        default_drift_cfg = {
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "p_sigma": 1.0,
            "t_sigma": 1.0,
            "p_eta": 1.0,
            "t_eta": 1.0,
            "scale_eucl": 1.0,
            "scale_spher": 1.0,
            "eps": 1e-8,
            "max_iter": 10,
            "p_tol": 1e-4,
            "p_weight": 1.0,
            "t_weight": 1.0,
            "n_gen_molecules": 64,
            "num_atom_types": 5,
            "pct_start": 0.1,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
            "lambda_clash": 0.1,
            "lambda_valence_excess": 0.1, 
            "lambda_hydrogen_valence": 0.1,
            "clash_threshold": 0.7,
            "bond_temperature": 0.1,
        }

        self.generator_cfg = {**default_generator_cfg, **(generator_cfg or {})}
        self.drift_cfg = {**default_drift_cfg, **(drift_cfg or {})}
        self.save_hyperparameters(
            {"generator_cfg": self.generator_cfg, "drift_cfg": self.drift_cfg}
        )

        if self.drift_cfg.get("end_sigma", None) is not None:
            self.end_sigma = self.drift_cfg["end_sigma"]
            self.p_sigma_start = self.drift_cfg.get("p_sigma", 1.0)
            self.t_sigma_start = self.drift_cfg.get("t_sigma", 1.0)
            self.anneal_sigma = True
        else:
            self.anneal_sigma = False

        self.generator = self._init_generator(self.generator_cfg)

        self.n_gen_molecules = self.drift_cfg.get("n_gen_molecules", 64)
        self.chem_refinement = self.drift_cfg.get("chem_refinement", False)
        self.max_epochs = self.drift_cfg.get("max_epochs", 100)
        self.start_frac_epoch = self.drift_cfg.get("start_frac_epoch", 0.8)

        self.eps = self.drift_cfg.get("eps", 1e-8)

        self._size_values: np.ndarray | None = None
        self._size_probs: np.ndarray | None = None
        self._norm_rescale_grad: float | None = None

    def _init_generator(self, cfg) -> EGNN:
        """Instantiate the EGNN model from generator configuration.

        Args:
            cfg: Generator configuration dictionary.

        Returns:
            An initialized EGNN model.
        """
        return EGNN(
            hidden_nf=cfg["hidden_nf"],
            num_blocks=cfg["n_layers"],
            num_atom_types=cfg["num_atom_types"],
            aggr_type=cfg["aggr_type"],
            tanh_coord_updates=cfg["tanh_coord_updates"],
            attention=cfg["attention"],
        )

    def set_size_distribution(self, sizes: np.ndarray, probs: np.ndarray) -> None:
        """Set the molecule size distribution for prior sampling.

        Args:
            sizes: Array of possible molecule sizes (number of atoms).
            probs: Probability distribution over sizes.
        """
        self._size_values = sizes
        self._size_probs = probs

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
        self, n_molecules: int, num_atoms: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of molecules from the prior distribution.

        Args:
            n_molecules: Number of molecules to sample.
            num_atoms: If set, override the size distribution to sample this many atoms.

        Returns:
            Tuple of (atom_types, positions, batch_indices, dense_edge_index).
        """
        if num_atoms is None:
            if self._size_values is None or self._size_probs is None:
                self._init_size_distribution()
            size_values = self._size_values
            size_probs = self._size_probs
        else:
            size_values = np.array([num_atoms], dtype=int)
            size_probs = np.array([1.0], dtype=float)

        x, pos, batch_vec, dense_edge_index, atom_counts = sample_prior_batch(
            n_molecules,
            size_values,
            size_probs,
            self.generator_cfg["num_atom_types"],
            self.device,
        )
        self._last_sampled_counts = atom_counts
        self._last_sampled_atom_probs = x.detach().cpu().numpy()
        return x, pos, batch_vec, dense_edge_index

    def _forward(self, batch, num_atoms):
        """Shared forward pass: sample prior → EGNN → center positions → sphere embeddings.

        Args:
            batch: Unused; kept for a uniform training-step signature.
            num_atoms: Number of atoms per molecule.

        Returns:
            Tuple of (gen_pos, gen_types_sphere, gen_batch_vec).
        """
        # Sample from the prior distribution
        x_prior, pos_prior, gen_batch_vec, gen_dense_edge_index = (
            self._sample_prior_batch(self.n_gen_molecules, num_atoms)
        )

        pos_prior = center_positions_per_graph(pos_prior, gen_batch_vec)

        # Generate molecule with EGNN
        gen_pos, gen_types = self.generator(pos_prior, x_prior, gen_dense_edge_index)

        # Center positions
        gen_pos = center_positions_per_graph(gen_pos, gen_batch_vec)

        # Turn x_logits into probabilites
        gen_types_prob = F.softmax(gen_types, dim=-1)

        # And project to the sphere
        gen_types_sphere = probs_to_sphere(gen_types_prob, self.eps)

        return (
            gen_pos,
            gen_types_sphere,
            gen_batch_vec,
        )

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

            self.drift_cfg["p_sigma"] = (
                self.end_sigma
                + 0.5
                * (self.p_sigma_start - self.end_sigma)
                * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            )

            self.drift_cfg["t_sigma"] = (
                self.end_sigma
                + 0.5
                * (self.t_sigma_start - self.end_sigma)
                * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            )

            self.log("drift/p_sigma", self.drift_cfg["p_sigma"])
            self.log("drift/t_sigma", self.drift_cfg["t_sigma"])


    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Args:
            batch: PyTorch Geometric data batch from the dataloader.
            batch_idx: Index of the batch.

        Returns:
            Scalar loss tensor for backpropagation.
        """
        num_atoms = self._batch_num_atoms(batch)
        gen_pos, gen_types_sphere, gen_batch_vec = self._forward(batch, num_atoms)

        real_pos, real_types = batch.pos, batch.real_atom_types

        try:
            if self.chem_refinement and (self.current_epoch > self.start_frac_epoch * self.max_epochs):
                loss, stats = compute_drift_loss(
                    gen_pos, real_pos, gen_types_sphere, real_types, num_atoms, chem_refinement=True, cfg=self.drift_cfg
                )
            else:
                loss, stats = compute_drift_loss(
                    gen_pos, real_pos, gen_types_sphere, real_types, num_atoms, chem_refinement=False, cfg=self.drift_cfg
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

        with torch.no_grad():
            pos_norms = gen_pos.norm(dim=-1)
            gen_center_norms = per_graph_center_norms(gen_pos, gen_batch_vec)
            real_center_norms = per_graph_center_norms(batch.pos, batch.batch)
            max_dist = torch.cdist(gen_pos, gen_pos).max()

        self.log("geom/pos_gen_norm_mean", pos_norms.mean(), batch_size=bs)
        self.log("geom/pos_gen_norm_std", pos_norms.std(), batch_size=bs)
        self.log("geom/max_atom_dist", max_dist, batch_size=bs)
        self.log("debug/gen_center_norm_mean", gen_center_norms.mean(), batch_size=bs)
        self.log("debug/gen_center_norm_std", gen_center_norms.std(), batch_size=bs)
        self.log("debug/real_center_norm_mean", real_center_norms.mean(), batch_size=bs)

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
        gen_pos, gen_types_sphere, gen_batch_vec = self._forward(batch, num_atoms)

        real_pos, real_types = batch.pos, batch.real_atom_types

        if self.chem_refinement and (self.current_epoch > self.start_frac_epoch * self.max_epochs):
            val_loss, stats = compute_drift_loss(
                gen_pos, real_pos, gen_types_sphere, real_types, num_atoms, chem_refinement=True, cfg=self.drift_cfg
            )
        else:
            val_loss, stats = compute_drift_loss(
                gen_pos, real_pos, gen_types_sphere, real_types, num_atoms, chem_refinement=False, cfg=self.drift_cfg
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

        with torch.no_grad():
            gen_cn = per_graph_center_norms(gen_pos, gen_batch_vec)
            real_cn = per_graph_center_norms(batch.pos, batch.batch)
        self.log(
            "debug/val_gen_center_norm_mean",
            gen_cn.mean(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "debug/val_real_center_norm_mean",
            real_cn.mean(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )

        # Project sphere embeddings back to probabilities
        with torch.no_grad():
            gen_types_prob = sphere_to_probs(gen_types_sphere, self.eps)

        return {
            "pos_gen": gen_pos.detach().cpu(),
            "gen_atom_types": gen_types_prob.detach().cpu().argmax(dim=-1),
            "pos_real": batch.pos.detach().cpu(),
            "real_atom_types": batch.real_atom_types.detach().cpu(),
            "gen_batch_vec": gen_batch_vec.detach().cpu(),
            "batch_vec": batch.batch.detach().cpu(),
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
        gen_pos, gen_types_sphere, gen_batch_vec = self._forward(batch, num_atoms)
        real_pos, real_types = batch.pos, batch.real_atom_types

        if self.chem_refinement and (self.current_epoch > self.start_frac_epoch * self.max_epochs):
            test_loss, stats = compute_drift_loss(
                gen_pos, real_pos, gen_types_sphere, real_types, num_atoms, chem_refinement=True, cfg=self.drift_cfg
            )
        else:
            test_loss, stats = compute_drift_loss(
                gen_pos, real_pos, gen_types_sphere, real_types, num_atoms, chem_refinement=False, cfg=self.drift_cfg
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