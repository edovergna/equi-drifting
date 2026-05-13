import copy
import os

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

import wandb


class GeneratorCheckpointCallback(Callback):
    """
    Logs the best and final generator (and optionally feature extractor) weights
    to wandb at the end of training.

    Monitors a metric each validation epoch and keeps an in-memory copy of the
    state dicts whenever the metric improves. On training end (or interruption),
    both _best and _final variants are saved under the run's
    individual_components/ directory and uploaded to wandb.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.monitor = monitor
        self.mode = mode
        self._best_score = float("inf") if mode == "min" else float("-inf")
        self._best_state_dict: dict | None = None
        self._best_fe_state_dict: dict | None = None

    def _is_better(self, current: float) -> bool:
        return (
            current < self._best_score
            if self.mode == "min"
            else current > self._best_score
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        current = current.item() if hasattr(current, "item") else float(current)
        if self._is_better(current):
            self._best_score = current
            self._best_state_dict = copy.deepcopy(pl_module.generator.state_dict())
            

    def _save_and_log(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        run_dir = logger.experiment.dir
        save_path = os.path.join(run_dir, "individual_components")
        os.makedirs(save_path, exist_ok=True)

        final_path = os.path.join(save_path, "generator_final.pth")
        torch.save(pl_module.generator.state_dict(), final_path)
        wandb.save(final_path, base_path=run_dir)
        print("Logged generator_final.pth to wandb.")

        if self._best_state_dict is not None:
            best_path = os.path.join(save_path, "generator_best.pth")
            torch.save(self._best_state_dict, best_path)
            wandb.save(best_path, base_path=run_dir)
            print(
                f"Logged generator_best.pth to wandb "
                f"({self.monitor}={self._best_score:.6f})."
            )

    

    def load_best_weights(self, pl_module: LightningModule) -> bool:
        """Loads the best in-memory state dict(s) into pl_module. Returns True if applied."""
        if self._best_state_dict is None:
            return False
        pl_module.generator.load_state_dict(self._best_state_dict)
        print(
            f"Loaded best weights ({self.monitor}={self._best_score:.6f}) into model."
        )
        return True

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._save_and_log(trainer, pl_module)

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        print(
            f"\n[GeneratorCheckpointCallback] {type(exception).__name__} — saving weights."
        )
        self._save_and_log(trainer, pl_module)
