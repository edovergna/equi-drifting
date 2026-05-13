from pathlib import Path

import numpy as np
import torch
import wandb
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

from ..geometry import per_graph_center_norms
from ..sample_prior import compute_size_distribution, sample_prior_batch


class BaseDriftingMoleculeGenerator(LightningModule):
    _SAVE_COMPONENTS = ["generator"]
    _LOAD_COMPONENTS = ["generator"]

    def __init__(self, generator_cfg: dict, drift_cfg: dict):
        super().__init__()
        self.generator_cfg = generator_cfg
        self.drift_cfg = drift_cfg

        self.prior_pos_clamp = generator_cfg["prior_pos_clamp"]
        self.eps = 1e-8

        self._size_values: np.ndarray | None = None
        self._size_probs: np.ndarray | None = None

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

    def _log_drift_stats(
        self,
        stats: dict,
        prefix: str,
        bs: int,
        *,
        on_step: bool,
        on_epoch: bool,
        sync_dist: bool = False,
        accumulate_hist: bool = False,
    ) -> None:
        """Log scalar drift stats and handle wandb histograms.

        accumulate_hist=True stores histograms in _val_hist_stats for on_validation_epoch_end;
        False logs them immediately (training use case).
        """
        hist_stats = {k: v for k, v in stats.items() if isinstance(v, wandb.Histogram)}
        for key, val in stats.items():
            if key in hist_stats:
                continue
            self.log(
                f"{prefix}/{key}",
                val,
                batch_size=bs,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=sync_dist,
            )
        if hist_stats:
            if accumulate_hist:
                self._val_hist_stats = {f"{prefix}/{k}": v for k, v in hist_stats.items()}
            elif hasattr(self, "logger") and hasattr(self.logger, "experiment"):
                try:
                    self.logger.experiment.log(
                        {f"{prefix}/{k}": v for k, v in hist_stats.items()},
                        commit=False,
                    )
                except Exception:
                    pass

    def _log_geometry(
        self,
        pos_gen: torch.Tensor,
        gen_batch_vec: torch.Tensor,
        real_pos: torch.Tensor,
        real_batch_vec: torch.Tensor,
        bs: int,
    ) -> None:
        with torch.no_grad():
            pos_norms = pos_gen.norm(dim=-1)
            gen_center_norms = per_graph_center_norms(pos_gen, gen_batch_vec)
            real_center_norms = per_graph_center_norms(real_pos, real_batch_vec)
            max_dist = torch.cdist(pos_gen, pos_gen).max()

        self.log("geom/pos_gen_norm_mean", pos_norms.mean(), batch_size=bs)
        self.log("geom/pos_gen_norm_std", pos_norms.std(), batch_size=bs)
        self.log("geom/max_atom_dist", max_dist, batch_size=bs)
        self.log("debug/gen_center_norm_mean", gen_center_norms.mean(), batch_size=bs)
        self.log("debug/gen_center_norm_std", gen_center_norms.std(), batch_size=bs)
        self.log("debug/real_center_norm_mean", real_center_norms.mean(), batch_size=bs)

    def on_validation_epoch_end(self):
        hist_stats = getattr(self, "_val_hist_stats", {})
        if hist_stats and hasattr(self, "logger") and hasattr(self.logger, "experiment"):
            try:
                self.logger.experiment.log(hist_stats, commit=False)
            except Exception:
                pass
        self._val_hist_stats = {}

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

    def save_individual_components(self, save_path: str) -> None:
        torch.save(self.generator.state_dict(), f"{save_path}/generator.pth")

    def load_individual_components(self, folder_path) -> None:
        folder_path = Path(folder_path)
        self.generator.load_state_dict(
            torch.load(folder_path / "generator.pth", map_location=self.device)
        )
