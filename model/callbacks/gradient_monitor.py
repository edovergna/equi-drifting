"""Callback for monitoring gradient statistics during training."""

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.optim import Optimizer


class GradientMonitorCallback(Callback):
    """Logs pre-clip gradient norms every training step."""

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        """Log gradient norms before optimizer step.

        Args:
            trainer: PyTorch Lightning Trainer.
            pl_module: Lightning module being trained.
            optimizer: Optimizer instance.
        """
        grads = [p.grad for p in pl_module.generator.parameters() if p.grad is not None]
        if not grads:
            return
        total_norm = torch.stack([g.detach().norm(2) for g in grads]).norm(2)
        max_abs = torch.stack([g.detach().abs().max() for g in grads]).max()
        pl_module.log("grad/total_norm", total_norm, on_step=True, on_epoch=False)
        pl_module.log("grad/max_abs", max_abs, on_step=True, on_epoch=False)
