"""PyTorch Lightning module for conditional 3D conformer generation.

Given fixed atom types from QM9, trains a ConditionalEGNN to predict
3D atomic coordinates using position-only drift loss with Kabsch +
Hungarian alignment. Atom types are never noised or predicted — they
are static conditioning signals throughout training and inference.
"""

from pathlib import Path

import torch
from torch.optim.lr_scheduler import OneCycleLR
from lightning.pytorch import LightningModule

from .drift_loss_conditional import (
    TrainingDivergedException,
    compute_conditional_drift_loss,
)
from .egnn_conditional import ConditionalEGNN
from .geometry import center_positions_per_graph, per_graph_center_norms
from .sample_prior import get_dense_edge_index


class ConditionalMoleculeGenerator(LightningModule):
    """Conditional conformer generator: atom types in, 3D positions out.

    Training loop:
      1. For each batch of real QM9 molecules (all same atom count):
         a. Extract real atom types (fixed conditioning).
         b. Sample independent Gaussian position noise.
         c. Center the noise and run ConditionalEGNN → gen_pos.
         d. Center gen_pos.
         e. Compute position-only drift loss vs real QM9 positions.

    The validation step returns a dict compatible with ChemicalValidityCallback:
    generated positions are evaluated together with the real (fixed) atom types
    to measure whether the predicted geometry forms a chemically valid molecule.
    """

    _SAVE_COMPONENTS = ["generator"]
    _LOAD_COMPONENTS = ["generator"]

    def __init__(self, generator_cfg=None, drift_cfg=None):
        """Initialize the conditional conformer generator.

        Args:
            generator_cfg: Dict with ConditionalEGNN hyperparameters.
                Keys: hidden_nf, n_layers, num_atom_types, aggr_type, tanh_coord_updates.
            drift_cfg: Dict with training hyperparameters.
                Keys: lr, weight_decay, p_sigma, p_eta, scale_eucl, eps,
                max_iter, p_tol, p_weight, t_weight, pct_start, div_factor,
                final_div_factor, chem_refinement, max_epochs, start_frac_epoch,
                lambda_clash, lambda_valence_excess, lambda_hydrogen_valence,
                clash_threshold, bond_temperature, end_sigma (optional).
        """
        super().__init__()

        default_generator_cfg = {
            "hidden_nf": 128,
            "n_layers": 4,
            "num_atom_types": 5,
            "aggr_type": "sum",
            "tanh_coord_updates": True,
        }
        default_drift_cfg = {
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "p_sigma": 1.0,
            "p_eta": 1.0,
            "scale_eucl": 1.0,
            "eps": 1e-8,
            "max_iter": 10,
            "p_tol": 1e-4,
            "p_weight": 1.0,
            "t_weight": 1.0,
            "n_gen_molecules": 64,
            "pct_start": 0.1,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
            "chem_refinement": False,
            "max_epochs": 500,
            "start_frac_epoch": 0.8,
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

        self.generator = ConditionalEGNN(
            num_atom_types=self.generator_cfg["num_atom_types"],
            num_blocks=self.generator_cfg["n_layers"],
            hidden_nf=self.generator_cfg["hidden_nf"],
            aggr_type=self.generator_cfg["aggr_type"],
            tanh_coord_updates=self.generator_cfg["tanh_coord_updates"],
        )

        self.eps = self.drift_cfg.get("eps", 1e-8)
        self.n_gen_molecules = self.drift_cfg.get("n_gen_molecules", 64)
        self.chem_refinement = self.drift_cfg.get("chem_refinement", False)
        self.max_epochs = self.drift_cfg.get("max_epochs", 500)
        self.start_frac_epoch = self.drift_cfg.get("start_frac_epoch", 0.8)

        if self.drift_cfg.get("end_sigma") is not None:
            self.end_sigma = self.drift_cfg["end_sigma"]
            self.p_sigma_start = self.drift_cfg.get("p_sigma", 1.0)
            self.anneal_sigma = True
        else:
            self.anneal_sigma = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _batch_num_atoms(self, batch) -> int:
        """Number of atoms in the first molecule of the batch."""
        return int((batch.ptr[1] - batch.ptr[0]).item())

    def _use_chem_refinement(self) -> bool:
        return (
            self.chem_refinement
            and self.current_epoch > self.start_frac_epoch * self.max_epochs
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one conditional forward pass with n_gen_molecules independent samples.

        Randomly draws n_gen_molecules atom-type templates from the batch, samples
        fresh Gaussian position noise for each, and returns centered generated positions.
        Using more gen molecules than the real batch size gives a much better
        drift estimate (same reason the joint model uses n_gen=128 >> n_real).

        Args:
            batch: PyTorch Geometric Data batch with attributes:
                pos [N_total, 3], real_atom_types [N_total, 5],
                dense_edge_index [2, E], batch [N_total].

        Returns:
            Tuple of (gen_pos [N_gen*num_atoms, 3],
                      gen_batch_vec [N_gen*num_atoms],
                      gen_types [N_gen*num_atoms, num_atom_types]).
        """
        num_atoms = self._batch_num_atoms(batch)
        batch_size = int(batch.batch.max().item()) + 1
        real_types_flat = batch.real_atom_types.float()  # [batch_size*num_atoms, 5]

        # Randomly pick which real molecule's types to condition each gen molecule on.
        mol_idx = torch.randint(0, batch_size, (self.n_gen_molecules,), device=self.device)

        # Gather types: reshape to [batch_size, num_atoms, 5], index, then flatten.
        real_types_by_mol = real_types_flat.reshape(batch_size, num_atoms, -1)
        gen_types = real_types_by_mol[mol_idx].reshape(self.n_gen_molecules * num_atoms, -1)

        # Build batch vector and fully-connected edge index for n_gen_molecules graphs.
        gen_batch_vec = torch.repeat_interleave(
            torch.arange(self.n_gen_molecules, device=self.device),
            num_atoms,
        )
        parts, offset = [], 0
        for _ in range(self.n_gen_molecules):
            parts.append(get_dense_edge_index(num_atoms, self.device) + offset)
            offset += num_atoms
        gen_edge_index = torch.cat(parts, dim=1)

        # Sample position noise only; types are given.
        pos_noisy = torch.randn(self.n_gen_molecules * num_atoms, 3, device=self.device)
        pos_noisy = center_positions_per_graph(pos_noisy, gen_batch_vec)

        gen_pos = self.generator(pos_noisy, gen_types, gen_edge_index)
        gen_pos = center_positions_per_graph(gen_pos, gen_batch_vec)

        return gen_pos, gen_batch_vec, gen_types

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        """Cosine-anneal position sigma if end_sigma is configured."""
        if self.anneal_sigma:
            progress = self.current_epoch / max(1, self.trainer.max_epochs - 1)
            self.drift_cfg["p_sigma"] = (
                self.end_sigma
                + 0.5
                * (self.p_sigma_start - self.end_sigma)
                * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            )
            self.log("drift/p_sigma", self.drift_cfg["p_sigma"])

    # ------------------------------------------------------------------
    # Training / validation / test
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """One training step: conditional forward → position drift loss.

        Args:
            batch: PyTorch Geometric data batch.
            batch_idx: Batch index (unused).

        Returns:
            Scalar loss tensor.
        """
        num_atoms = self._batch_num_atoms(batch)
        gen_pos, gen_batch_vec, gen_types = self._forward(batch)
        real_pos = batch.pos
        real_types = batch.real_atom_types

        try:
            loss, stats = compute_conditional_drift_loss(
                gen_pos,
                real_pos,
                gen_types,
                real_types,
                num_atoms,
                chem_refinement=self._use_chem_refinement(),
                cfg=self.drift_cfg,
            )
        except TrainingDivergedException as e:
            self.print(f"\n[Step {self.global_step}] {e}\nStopping training.")
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        bs = int(gen_batch_vec.max().item()) + 1

        self.log("train_loss", loss, batch_size=bs, on_step=True, on_epoch=True)

        for key, val in stats.items():
            self.log(
                f"drift_train/{key}", val, batch_size=bs, on_step=True, on_epoch=False
            )

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
        """One validation step: compute val_loss and prepare chemical validity data.

        Returns a dict compatible with ChemicalValidityCallback. The generated
        positions are paired with the fixed real atom types to evaluate whether
        the predicted conformer is chemically valid.

        Args:
            batch: PyTorch Geometric data batch.
            batch_idx: Batch index (unused).

        Returns:
            Dict with pos_gen, gen_atom_types, gen_batch_vec, etc.
        """
        num_atoms = self._batch_num_atoms(batch)
        gen_pos, gen_batch_vec, gen_types = self._forward(batch)
        real_pos = batch.pos
        real_types = batch.real_atom_types

        val_loss, stats = compute_conditional_drift_loss(
            gen_pos,
            real_pos,
            gen_types,
            real_types,
            num_atoms,
            chem_refinement=self._use_chem_refinement(),
            cfg=self.drift_cfg,
        )

        bs = int(gen_batch_vec.max().item()) + 1

        self.log(
            "val_loss",
            val_loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for key, val in stats.items():
            self.log(
                f"drift_val/{key}",
                val,
                batch_size=bs,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

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

        return {
            "pos_gen": gen_pos.detach().cpu(),
            "gen_atom_types": gen_types.argmax(dim=-1).detach().cpu(),
            "pos_real": real_pos.detach().cpu(),
            "real_atom_types": real_types.detach().cpu(),
            "gen_batch_vec": gen_batch_vec.detach().cpu(),
            "batch_vec": batch.batch.detach().cpu(),
            "num_atoms": num_atoms,
        }

    def test_step(self, batch, batch_idx):
        """One test step.

        Args:
            batch: PyTorch Geometric data batch.
            batch_idx: Batch index (unused).

        Returns:
            Scalar test loss.
        """
        num_atoms = self._batch_num_atoms(batch)
        gen_pos, gen_batch_vec, gen_types = self._forward(batch)
        real_pos = batch.pos
        real_types = batch.real_atom_types

        test_loss, _ = compute_conditional_drift_loss(
            gen_pos,
            real_pos,
            gen_types,
            real_types,
            num_atoms,
            chem_refinement=False,
            cfg=self.drift_cfg,
        )

        bs = int(gen_batch_vec.max().item()) + 1
        self.log(
            "test_loss",
            test_loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return test_loss

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_individual_components(self, save_path: str) -> None:
        """Save generator weights to save_path/generator.pth."""
        torch.save(self.generator.state_dict(), f"{save_path}/generator.pth")

    def load_individual_components(self, folder_path) -> None:
        """Load generator weights from folder_path/generator.pth."""
        folder_path = Path(folder_path)
        self.generator.load_state_dict(
            torch.load(folder_path / "generator.pth", map_location=self.device)
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """AdamW optimizer with OneCycleLR scheduler."""
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
