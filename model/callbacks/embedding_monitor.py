import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer

import wandb


class EmbeddingMonitorCallback(Callback):
    """
    Collects φ_gen and φ_real from validation_step outputs and logs WandB histograms
    at the end of each validation epoch.

    validation_step must return a dict containing "phi_gen" and "phi_real" (CPU tensors).
    """

    MAX_EMBEDDINGS = 2048

    def __init__(self):
        self._phi_gen: list[torch.Tensor] = []
        self._phi_real: list[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict):
            return
        phi_gen = outputs.get("phi_gen")
        phi_real = outputs.get("phi_real")
        if phi_gen is None or phi_real is None:
            return
        collected = sum(t.size(0) for t in self._phi_gen)
        if collected >= self.MAX_EMBEDDINGS:
            return
        self._phi_gen.append(phi_gen.cpu().float())
        self._phi_real.append(phi_real.cpu().float())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self._phi_gen:
            return

        phi_gen = torch.cat(self._phi_gen, dim=0)[: self.MAX_EMBEDDINGS]
        phi_real = torch.cat(self._phi_real, dim=0)[: self.MAX_EMBEDDINGS]
        self._phi_gen.clear()
        self._phi_real.clear()

        phi_gen_unit = F.normalize(phi_gen, dim=-1)
        phi_real_unit = F.normalize(phi_real, dim=-1)
        cos_sim = (phi_gen_unit @ phi_real_unit.T).max(dim=1).values.numpy()

        gen_norms = phi_gen.norm(dim=-1).numpy()
        real_norms = phi_real.norm(dim=-1).numpy()

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        try:
            logger.experiment.log(
                {
                    "embed/cosine_sim_to_nn_hist": wandb.Histogram(cos_sim),
                    "embed/phi_gen_norm_hist": wandb.Histogram(gen_norms),
                    "embed/phi_real_norm_hist": wandb.Histogram(real_norms),
                },
                step=trainer.global_step,
            )
        except Exception:
            pass
