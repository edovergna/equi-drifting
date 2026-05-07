from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

from ept.ept_loader import load_ept_feature_extractor

from .drift_loss import (
    TrainingDivergedException,
    compute_norm_based_drift_loss,
    compute_inverse_attn_drift_loss,
    original_compute_drift_loss,
)
from .egnn import EGNN
from .geometry import (batch_size_for_logging, center_positions_per_graph,
                       per_graph_center_norms)
from .sample_prior import compute_size_distribution, sample_prior_batch


class DriftingMoleculeGenerator(LightningModule):
    _SAVE_COMPONENTS = ["generator", "feature_extractor"]
    _LOAD_COMPONENTS = ["generator", "feature_extractor"]

    def __init__(self, generator_cfg=None, drift_cfg=None):
        super().__init__()

        default_generator_cfg = {
            "hidden_nf": 128,
            "n_layers": 2,
            "num_atom_types": 5,
            "num_bond_types": 5,
            "coordinate_clamp_range": 3.0,
            "predict_bond_types": False,
            "pos_clamp": 20.0,
            "prior_pos_clamp": 3.0,
        }
        default_drift_cfg = {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "temperatures": [0.02, 0.05, 0.2],
            "loss_variant": "norm_based",
            "pct_start": 0.1,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
        }

        self.generator_cfg = {**default_generator_cfg, **(generator_cfg or {})}
        self.drift_cfg = {**default_drift_cfg, **(drift_cfg or {})}
        self.save_hyperparameters(
            {"generator_cfg": self.generator_cfg, "drift_cfg": self.drift_cfg}
        )

        self.generator = self._init_generator(self.generator_cfg)
        self.feature_extractor = self._init_feature_extractor()
        self._freeze_feature_extractor()

        self.temperatures = self.drift_cfg["temperatures"]
        self.loss_variant = self.drift_cfg["loss_variant"]
        self.pos_clamp = self.generator_cfg["pos_clamp"]
        self.prior_pos_clamp = self.generator_cfg["prior_pos_clamp"]

        self._size_values: np.ndarray | None = None
        self._size_probs: np.ndarray | None = None

    def _init_generator(self, cfg) -> EGNN:
        return EGNN(
            hidden_nf=cfg["hidden_nf"],
            n_layers=cfg["n_layers"],
            num_atom_types=cfg["num_atom_types"],
            num_bond_types=cfg["num_bond_types"],
            predict_bond_types=cfg["predict_bond_types"],
        )

    def _init_feature_extractor(self):
        return load_ept_feature_extractor()

    def _freeze_feature_extractor(self) -> None:
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def set_size_distribution(self, sizes: np.ndarray, probs: np.ndarray) -> None:
        self._size_values = sizes
        self._size_probs = probs

    def _init_size_distribution(self) -> None:
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
        self, n_molecules: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._size_values is None or self._size_probs is None:
            self._init_size_distribution()

        x, pos, batch_vec, dense_edge_index, atom_counts = sample_prior_batch(
            n_molecules,
            self._size_values,
            self._size_probs,
            self.generator_cfg["num_atom_types"],
            self.prior_pos_clamp,
            self.device,
        )
        self._last_sampled_counts = atom_counts  # read by SizeDistributionCallback
        self._last_sampled_atom_probs = x.detach().cpu().numpy()  # read by AtomTypeDistributionCallback
        return x, pos, batch_vec, dense_edge_index

    def _forward(self, batch):
        """Shared forward pass: prior → EGNN → center → hard atoms → EPT embeddings."""
        n_molecules = batch_size_for_logging(batch)
        x_prior, pos_prior, gen_batch_vec, gen_dense_edge_index = (
            self._sample_prior_batch(n_molecules)
        )

        x_gen, _, pos_gen = self.generator(x_prior, pos_prior, gen_dense_edge_index)
        pos_gen = center_positions_per_graph(pos_gen, gen_batch_vec)
        pos_gen = pos_gen.clamp(-self.pos_clamp, self.pos_clamp)
        # gen_atom_types = x_gen.softmax(dim=-1).argmax(dim=-1)
        gen_atom_types = F.gumbel_softmax(x_gen, tau=1.0, hard=True)

        # EPT expects block_id[i] = block index for atom i (each atom is its own block,
        # so block index = atom index), and batch_id[j] = graph index for block j.
        phi_gen = self.feature_extractor(
            pos=pos_gen,
            atom_types=gen_atom_types,
            block_id=torch.arange(pos_gen.shape[0], device=self.device),
            batch_id=gen_batch_vec,
            dense_edge_index=gen_dense_edge_index,
        )
        phi_real = self.feature_extractor(
            pos=batch.pos,
            atom_types=batch.real_atom_types,
            block_id=torch.arange(batch.num_nodes, device=batch.batch.device),
            batch_id=batch.batch,
            dense_edge_index=batch.dense_edge_index,
        )
        return pos_gen, gen_atom_types, phi_gen, phi_real, gen_batch_vec

    def _compute_loss(
        self, phi_gen: torch.Tensor, phi_real: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        if self.loss_variant == "original":
            return original_compute_drift_loss(phi_gen, phi_real, self.temperatures)
        elif self.loss_variant == "inverse_attn":
            return compute_inverse_attn_drift_loss(phi_gen, phi_real, self.temperatures)
        else:
            return compute_norm_based_drift_loss(phi_gen, phi_real, self.temperatures)

    def training_step(self, batch, batch_idx):
        pos_gen, _, phi_gen, phi_real, gen_batch_vec = self._forward(batch)

        if not (torch.isfinite(phi_gen).all() and torch.isfinite(phi_real).all()):
            with torch.no_grad():
                bad_gen_mask = ~torch.isfinite(phi_gen).all(dim=-1)  # [num_graphs]
                bad_real_mask = ~torch.isfinite(phi_real).all(dim=-1)
                bad_mol_mask = bad_gen_mask | bad_real_mask

                n_bad_gen = bad_gen_mask.sum().item()
                n_bad_real = bad_real_mask.sum().item()
                n_total = bad_mol_mask.shape[0]

                atom_mask = bad_mol_mask[gen_batch_vec]
                pos_bad = pos_gen[atom_mask]

                pos_norms_bad = pos_bad.norm(dim=-1)
                max_dist_bad = (
                    torch.cdist(pos_bad, pos_bad).max()
                    if pos_bad.shape[0] > 1
                    else pos_bad.new_tensor(0.0)
                )

                gen_center_norms = per_graph_center_norms(pos_gen, gen_batch_vec)
                real_center_norms = per_graph_center_norms(batch.pos, batch.batch)
                bad_gen_cn = gen_center_norms[bad_mol_mask]
                bad_real_cn = real_center_norms[bad_mol_mask]

                self.print(
                    f"\n[Step {self.global_step}] Non-finite embeddings — "
                    f"{n_bad_gen} gen mol(s), {n_bad_real} real mol(s) out of {n_total}.\n"
                    f"  [bad mols] pos_gen norm   mean={pos_norms_bad.mean():.3f}  std={pos_norms_bad.std():.3f}\n"
                    f"  [bad mols] max pairwise dist={max_dist_bad:.3f}\n"
                    f"  [bad mols] gen center norm  mean={bad_gen_cn.mean():.3f}  std={bad_gen_cn.std():.3f}\n"
                    f"  [bad mols] real center norm mean={bad_real_cn.mean():.3f}  std={bad_real_cn.std():.3f}\n"
                    f"  Stopping."
                )

            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        try:
            loss, stats = self._compute_loss(phi_gen, phi_real)
        except TrainingDivergedException as e:
            self.print(f"\n[Step {self.global_step}] {e}\nStopping training.")
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        bs = batch_size_for_logging(batch)
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
                    step=self.global_step,
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
            pos_norms = pos_gen.norm(dim=-1)
            gen_center_norms = per_graph_center_norms(pos_gen, gen_batch_vec)
            real_center_norms = per_graph_center_norms(batch.pos, batch.batch)
            max_dist = torch.cdist(pos_gen, pos_gen).max()

        self.log("geom/pos_gen_norm_mean", pos_norms.mean(), batch_size=bs)
        self.log("geom/pos_gen_norm_std", pos_norms.std(), batch_size=bs)
        self.log("geom/max_atom_dist", max_dist, batch_size=bs)
        self.log("debug/gen_center_norm_mean", gen_center_norms.mean(), batch_size=bs)
        self.log("debug/gen_center_norm_std", gen_center_norms.std(), batch_size=bs)
        self.log("debug/real_center_norm_mean", real_center_norms.mean(), batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        pos_gen, gen_atom_types, phi_gen, phi_real, gen_batch_vec = self._forward(batch)
        val_loss, stats = self._compute_loss(phi_gen, phi_real)

        bs = batch_size_for_logging(batch)
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
        if hist_stats and hasattr(self.logger, "experiment"):
            try:
                self.logger.experiment.log(
                    {f"drift_val/{k}": v for k, v in hist_stats.items()},
                    step=self.global_step,
                )
            except Exception:
                pass

        with torch.no_grad():
            gen_cn = per_graph_center_norms(pos_gen, gen_batch_vec)
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
            "phi_gen": phi_gen.detach().cpu(),
            "phi_real": phi_real.detach().cpu(),
            "pos_gen": pos_gen.detach().cpu(),
            "gen_atom_types": gen_atom_types.detach().cpu().argmax(dim=-1),
            "pos_real": batch.pos.detach().cpu(),
            "real_atom_types": batch.real_atom_types.detach().cpu(),
            "gen_batch_vec": gen_batch_vec.detach().cpu(),
            "batch_vec": batch.batch.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        _, _, phi_gen, phi_real, _ = self._forward(batch)
        test_loss, _ = self._compute_loss(phi_gen, phi_real)

        bs = batch_size_for_logging(batch)
        self.log(
            "test_loss",
            test_loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return test_loss

    def _is_feature_extractor_trainable(self) -> bool:
        return any(p.requires_grad for p in self.feature_extractor.parameters())

    def save_individual_components(self, save_path: str) -> None:
        torch.save(self.generator.state_dict(), f"{save_path}/generator.pth")
        if self._is_feature_extractor_trainable():
            torch.save(
                self.feature_extractor.ept_model.state_dict(),
                f"{save_path}/feature_extractor.pth",
            )

    def load_individual_components(self, folder_path) -> None:
        folder_path = Path(folder_path)
        self.generator.load_state_dict(
            torch.load(folder_path / "generator.pth", map_location=self.device)
        )
        fe_path = folder_path / "feature_extractor.pth"
        if fe_path.exists():
            self.feature_extractor.ept_model.load_state_dict(
                torch.load(fe_path, map_location=self.device)
            )
            print("Loaded fine-tuned feature extractor weights.")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.drift_cfg["lr"],
            weight_decay=self.drift_cfg["weight_decay"],
            eps=1e-8,
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
