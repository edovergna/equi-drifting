"""Callback for logging 3D molecule renders and atom-type distributions to WandB."""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from PIL import Image as PILImage

import wandb

_ATOM_NAMES = ["H", "C", "N", "O", "F"]

class AtomTypesMonitor(Callback):


    _REQUIRED_KEYS = {
        "real_atom_types",
        "gen_atom_types",
    }

    def __init__(
        self,
        every_n_epochs: int = 1,
    ):
        """Initialize the molecule atom types callback.

        Args:
            every_n_epochs: Log visualizations every n validation epochs.
        """
        self.every_n_epochs = every_n_epochs
        self._ref: dict | None = None
        self._gen_atom_types: list[torch.Tensor] = []
        self._real_atom_types: list[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate outputs from first validation batch and all atom types.

        Args:
            trainer: PyTorch Lightning Trainer.
            pl_module: Lightning module being trained.
            outputs: Output from the validation step (expects dict with required keys).
            batch: Current batch data.
            batch_idx: Index of current batch.
            dataloader_idx: Index if using multiple dataloaders.
        """
        if not isinstance(outputs, dict) or not self._REQUIRED_KEYS.issubset(outputs):
            return
        if batch_idx == 0:
            self._ref = {k: outputs[k] for k in self._REQUIRED_KEYS}
            self._ref["gen_atom_types"] = outputs.get("gen_atom_types")
        gen_atom_types = outputs.get("gen_atom_types")
        if gen_atom_types is not None:
            self._gen_atom_types.append(gen_atom_types.cpu().reshape(-1))
        self._real_atom_types.append(
            outputs["real_atom_types"].argmax(dim=-1).cpu().reshape(-1)
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Render 3D visualizations and log to wandb at validation epoch end.

        Args:
            trainer: PyTorch Lightning Trainer.
            pl_module: Lightning module being trained.
        """
        gen_types = torch.cat(self._gen_atom_types) if self._gen_atom_types else None
        real_types = torch.cat(self._real_atom_types) if self._real_atom_types else None
        self._gen_atom_types.clear()
        self._real_atom_types.clear()

        if trainer.current_epoch % self.every_n_epochs != 0 or self._ref is None:
            return
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        try:
            ref = self._ref

            real_atom_types = ref["real_atom_types"].argmax(dim=-1)
            images = {}

            if gen_types is not None and real_types is not None:
                images["mol/atom_type_dist"] = self._atom_dist_chart(
                    gen_types, real_types
                )

            logger.experiment.log(images, commit=False)

        except Exception as e:
            print(f"[AtomTypesMonitor] Skipped: {e}")


    def _atom_dist_chart(
        self, gen_types: torch.Tensor, real_types: torch.Tensor
    ) -> "wandb.Image":
        """Create bar chart comparing atom-type distributions.

        Args:
            gen_types: Generated atom type indices [total_nodes].
            real_types: Real atom type indices [total_nodes].

        Returns:
            WandB Image object with side-by-side bar chart.
        """
        n = len(_ATOM_NAMES)
        gen_frac = np.array(
            [(gen_types == t).sum().item() for t in range(n)], dtype=float
        )
        real_frac = np.array(
            [(real_types == t).sum().item() for t in range(n)], dtype=float
        )
        gen_frac /= gen_frac.sum() + 1e-8
        real_frac /= real_frac.sum() + 1e-8

        x, w = np.arange(n), 0.35
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(x - w / 2, real_frac, w, label="Real", color="steelblue", alpha=0.8)
        ax.bar(x + w / 2, gen_frac, w, label="Generated", color="tomato", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(_ATOM_NAMES)
        ax.set_ylabel("Fraction")
        ax.set_title("Atom type distribution")
        ax.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return wandb.Image(PILImage.open(buf).copy())
