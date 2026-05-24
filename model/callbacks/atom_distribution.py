"""Callback for monitoring atom-type distribution of sampled molecules."""

import io

import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch import Callback, LightningModule, Trainer
from PIL import Image as PILImage

import wandb

ATOM_TYPE_NAMES = ["H", "C", "N", "O", "F"]


class AtomTypeDistributionCallback(Callback):
    """Visualizes Dirichlet prior atom-type sampling distribution each epoch.

    Reads `pl_module._last_sampled_atom_probs` (set by _sample_prior_batch) every
    training step — a [total_nodes, num_atom_types] array of Dirichlet samples —
    and logs a bar chart of the mean probability per atom type to WandB at epoch end.
    """

    def __init__(self, log_every_n_epochs: int = 1):
        """Initialize the atom type distribution callback.

        Args:
            log_every_n_epochs: Log every n validation epochs.
        """
        self.log_every_n_epochs = log_every_n_epochs
        self._accumulated: list[np.ndarray] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Accumulate atom type probabilities from training batches.

        Args:
            trainer: PyTorch Lightning Trainer.
            pl_module: Lightning module being trained.
            outputs: Output from the training step.
            batch: Current batch data.
            batch_idx: Index of current batch.
        """
        probs = getattr(pl_module, "_last_sampled_atom_probs", None)
        if probs is not None:
            self._accumulated.append(probs.copy())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log atom type distribution at validation epoch end.

        Args:
            trainer: PyTorch Lightning Trainer.
            pl_module: Lightning module being trained.
        """
        if (
            trainer.current_epoch % self.log_every_n_epochs != 0
            or not self._accumulated
        ):
            self._accumulated.clear()
            return

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            self._accumulated.clear()
            return

        # all_probs: [total_nodes_across_epoch, num_atom_types]
        all_probs = np.concatenate(self._accumulated, axis=0)
        self._accumulated.clear()

        mean_probs = all_probs.mean(axis=0)  # [num_atom_types]

        try:
            x = np.arange(len(ATOM_TYPE_NAMES))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x, mean_probs, color="steelblue", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(ATOM_TYPE_NAMES)
            ax.set_xlabel("Atom type")
            ax.set_ylabel("Mean sampled probability")
            ax.set_title(
                f"Dirichlet prior atom-type distribution — epoch {trainer.current_epoch}"
            )
            ax.axhline(
                1.0 / len(ATOM_TYPE_NAMES),
                color="tomato",
                linestyle="--",
                linewidth=1,
                label="Uniform (1/5)",
            )
            ax.legend()
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            logger.experiment.log(
                {
                    "prior/atom_type_distribution": wandb.Image(
                        PILImage.open(buf).copy()
                    )
                },
                commit=False,
            )
        except Exception as e:
            print(f"[AtomTypeDistributionCallback] Skipped: {e}")
