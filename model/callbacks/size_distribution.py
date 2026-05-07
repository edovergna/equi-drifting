import io

import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch import Callback, LightningModule, Trainer
from PIL import Image as PILImage

import wandb


class SizeDistributionCallback(Callback):
    """
    Compares the empirical atom-count distribution of sampled molecules against
    the true QM9 distribution each epoch.

    Reads `pl_module._last_sampled_counts` (set by _sample_prior_batch) every
    training step and logs a side-by-side bar chart to WandB at epoch end.
    """

    def __init__(self, log_every_n_epochs: int = 1):
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
        counts = getattr(pl_module, "_last_sampled_counts", None)
        if counts is not None:
            self._accumulated.append(counts.copy())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
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

        true_sizes = getattr(pl_module, "_size_values", None)
        true_probs = getattr(pl_module, "_size_probs", None)
        if true_sizes is None or true_probs is None:
            self._accumulated.clear()
            return

        all_counts = np.concatenate(self._accumulated)
        self._accumulated.clear()

        max_size = int(true_sizes.max())
        emp_bins = np.bincount(all_counts, minlength=max_size + 1)
        emp_probs = emp_bins / emp_bins.sum()

        try:
            x = true_sizes
            w = 0.4
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(
                x - w / 2,
                true_probs,
                w,
                label="QM9 (true)",
                color="steelblue",
                alpha=0.8,
            )
            ax.bar(
                x + w / 2, emp_probs[x], w, label="Sampled", color="tomato", alpha=0.8
            )
            ax.set_xlabel("Atom count")
            ax.set_ylabel("Fraction")
            ax.set_title(f"Atom-count distribution — epoch {trainer.current_epoch}")
            ax.legend()
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            logger.experiment.log(
                {"prior/size_distribution": wandb.Image(PILImage.open(buf).copy())},
                commit=False,
            )
        except Exception as e:
            print(f"[SizeDistributionCallback] Skipped: {e}")
