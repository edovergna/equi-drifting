from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

from .drift_loss import (
    TrainingDivergedException,
    compute_aligning_drift_loss
)
from .egnn import EGNN
from .geometry import center_positions_per_graph, per_graph_center_norms
from .sample_prior import compute_size_distribution, sample_prior_batch

from ..spherical_utils import (probs_to_sphere, sphere_to_probs)


class MoleculeGenerator(LightningModule):
    _SAVE_COMPONENTS = ["generator"]
    _LOAD_COMPONENTS = ["generator"]

    def __init__(self, generator_cfg=None, drift_cfg=None):
        super().__init__()

        default_generator_cfg = {
            "hidden_nf": 128,
            "n_layers": 2,
            "num_atom_types": 5,
            "num_bond_types": 5,
            "coordinate_clamp_range": 3.0,
            "predict_bond_types": False,
            "predict_atom_types": True,
            "pos_clamp": 20.0,
            "pos_clamp_type": "geom",
            "c_pos_clamp": 5.0,
            "p_pos_clamp": 4.0,
            "norm_pos_clamp": 10.0,
            "prior_pos_clamp": 3.0,
            "use_feature_extractor": False,
            "infer_types_from_pos": False,
            "infer_method": "heuristic",
        }
        default_drift_cfg = {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "temperatures": [0.02, 0.05, 0.2],
            "loss_variant": "original",
            "atom_type_temp": 1.0,
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

        self.temperatures = self.drift_cfg["temperatures"]
        self.loss_variant = self.drift_cfg["loss_variant"]
        self.atom_type_temp = self.drift_cfg.get("atom_type_temp", 1.0)
        self.n_gen_molecules = self.drift_cfg.get("n_gen_molecules", 64)
        self.pos_clamp = self.generator_cfg["pos_clamp"]
        self.pos_clamp_type = self.generator_cfg["pos_clamp_type"]
        self.c_pos_clamp = self.generator_cfg["c_pos_clamp"]
        self.p_pos_clamp = self.generator_cfg["p_pos_clamp"]
        self.norm_pos_clamp = self.generator_cfg["norm_pos_clamp"]
        self.prior_pos_clamp = self.generator_cfg["prior_pos_clamp"]
        self.infer_types_from_pos = self.generator_cfg.get("infer_types_from_pos", False)
        self.infer_method = self.generator_cfg.get("infer_method", "heuristic")

        # Hard coded epsilon!! TODO: make it in cfg
        self.eps = 1e-8

        self._size_values: np.ndarray | None = None
        self._size_probs: np.ndarray | None = None
        self._norm_rescale_grad: float | None = None

    def _init_generator(self, cfg) -> EGNN:
        return EGNN(
            hidden_nf=cfg["hidden_nf"],
            n_layers=cfg["n_layers"],
            num_atom_types=cfg["num_atom_types"],
            num_bond_types=cfg["num_bond_types"],
            predict_bond_types=cfg["predict_bond_types"],
            predict_atom_types=cfg["predict_atom_types"],
        )

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
        self._last_sampled_atom_probs = (
            x.detach().cpu().numpy()
        )  # read by AtomTypeDistributionCallback
        return x, pos, batch_vec, dense_edge_index

    def _forward(self, batch):
        """Shared forward pass: prior → EGNN → center → hard atoms → EPT embeddings."""
        # Sample from the prior distribution
        x_prior, pos_prior, gen_batch_vec, gen_dense_edge_index = (
            self._sample_prior_batch(self.n_gen_molecules)
        )

        pos_prior = center_positions_per_graph(pos_prior, gen_batch_vec)

        # Generate molecule with EGNN
        x_logits, _, pos_gen = self.generator(x_prior, pos_prior, gen_dense_edge_index)

        # Center positions
        pos_gen = center_positions_per_graph(pos_gen, gen_batch_vec)

        # Clamp generated positions - lets try no clamping of final positions
        # self._norm_rescale_grad = None
        # if self.pos_clamp_type == "hard":
        #     pos_gen = pos_gen.clamp(-self.pos_clamp, self.pos_clamp)
        # elif self.pos_clamp_type == "tanh":
        #     norm = pos_gen.norm(dim=-1, keepdim=True)
        #     rescale = torch.tanh(norm / self.norm_pos_clamp) / (
        #         norm / self.norm_pos_clamp + 1e-8
        #     )
        #     if not self.trainer.sanity_checking and self.trainer.validating is False:
        #         rescale.register_hook(
        #             lambda g: setattr(self, "_norm_rescale_grad", g.abs().mean().item())
        #         )
        #     pos_gen = pos_gen * rescale
        # else:  # geom
        #     norm = pos_gen.norm(dim=-1)
        #     rescale = 1 / (1 + (norm / self.c_pos_clamp) ** self.p_pos_clamp)
        #     if not self.trainer.sanity_checking and self.trainer.validating is False:
        #         rescale.register_hook(
        #             lambda g: setattr(self, "_norm_rescale_grad", g.abs().mean().item())
        #         )
        #     pos_gen = pos_gen * rescale.unsqueeze(-1)

        # Turn x_logits into probabilites
        x_prob = F.softmax(x_logits, dim=-1)

        # And project to the sphere
        x_gen_sphere = probs_to_sphere(x_prob, self.eps)

        return (
            pos_gen,
            x_gen_sphere,
            gen_batch_vec,
        )

    def on_after_backward(self):
        if self._norm_rescale_grad is not None:
            self.log(
                "geom/norm_rescale_grad",
                self._norm_rescale_grad,
                on_step=True,
                on_epoch=False,
            )

    def training_step(self, batch, batch_idx):
        pos_gen, x_gen_sphere, gen_batch_vec = self._forward(batch)

        pos_real, x_real = batch.pos, batch.real_atom_types

        try:
            loss, stats = compute_aligning_drift_loss(
                pos_gen, pos_real, x_gen_sphere, x_real, gen_batch_vec, batch.batch
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
        pos_gen, x_gen_sphere, gen_batch_vec = self._forward(batch)

        pos_real, x_real = batch.pos, batch.real_atom_types

        val_loss, stats = compute_aligning_drift_loss(
                pos_gen, pos_real, x_gen_sphere, x_real, gen_batch_vec, batch.batch
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

        # Project sphere embeddings back to probabilities
        with torch.no_grad():
            x_prob = sphere_to_probs(x_gen_sphere, self.eps)

        return {
            "pos_gen": pos_gen.detach().cpu(),
            "gen_atom_types": x_prob.detach().cpu().argmax(dim=-1),
            "pos_real": batch.pos.detach().cpu(),
            "real_atom_types": batch.real_atom_types.detach().cpu(),
            "gen_batch_vec": gen_batch_vec.detach().cpu(),
            "batch_vec": batch.batch.detach().cpu(),
        }

    def on_validation_epoch_end(self):
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
        pos_gen, x_gen_sphere, gen_batch_vec = self._forward(batch)
        pos_real, x_real = batch.pos, batch.real_atom_types

        test_loss, _ = compute_aligning_drift_loss(
                pos_gen, pos_real, x_gen_sphere, x_real, gen_batch_vec, batch.batch
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
        torch.save(self.generator.state_dict(), f"{save_path}/generator.pth")

    def load_individual_components(self, folder_path) -> None:
        folder_path = Path(folder_path)
        self.generator.load_state_dict(
            torch.load(folder_path / "generator.pth", map_location=self.device)
        )

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