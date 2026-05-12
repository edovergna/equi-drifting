from collections import Counter

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer

import wandb

from ..mol_utils import (batch_to_stability, batch_to_validity,
                        heavy_atom_counts)


class ChemicalValidityCallback(Callback):
    """
    Computes chemical validity and uniqueness of generated molecules each validation epoch.

    Collects up to MAX_MOLS generated molecules, then logs:
    - chem/validity        — fraction of structurally valid molecules
    - chem/uniqueness      — fraction of unique valid molecules (unique SMILES / formulas)
    - chem/heavy_atom_mean — mean number of heavy (non-H) atoms per generated molecule
    - chem/valid_smiles    — WandB table of unique valid identifiers and their counts
    """

    MAX_MOLS = 512

    def __init__(self):
        self._pos: list[torch.Tensor] = []
        self._ahard: list[torch.Tensor] = []
        self._batch: list[torch.Tensor] = []
        self._offset: int = 0

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._pos.clear()
        self._ahard.clear()
        self._batch.clear()
        self._offset = 0

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
        pos = outputs.get("pos_gen")
        a_hard = outputs.get("gen_atom_types")
        bvec = outputs.get("gen_batch_vec", outputs.get("batch_vec"))
        if pos is None or a_hard is None or bvec is None:
            return
        if self._offset >= self.MAX_MOLS:
            return

        self._pos.append(pos.cpu())
        self._ahard.append(a_hard.cpu())
        self._batch.append(bvec.cpu() + self._offset)
        self._offset += int(bvec.max().item()) + 1

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self._pos:
            return

        pos = torch.cat(self._pos, dim=0)
        a_hard = torch.cat(self._ahard, dim=0)
        bvec = torch.cat(self._batch, dim=0)

        n_graphs = int(bvec.max().item()) + 1
        if n_graphs > self.MAX_MOLS:
            keep = bvec < self.MAX_MOLS
            pos, a_hard, bvec = pos[keep], a_hard[keep], bvec[keep]

        results = batch_to_validity(pos, a_hard, bvec)
        heavy = heavy_atom_counts(a_hard, bvec)
        atom_stable_frac, mol_stable_frac = batch_to_stability(pos, a_hard, bvec)

        n_total = len(results)
        n_valid = sum(1 for ok, _ in results if ok)
        valid_ids = [ident for ok, ident in results if ok and ident is not None]
        uniqueness = len(set(valid_ids)) / len(valid_ids) if valid_ids else 0.0
        validity = n_valid / n_total if n_total > 0 else 0.0
        heavy_mean = float(np.mean(heavy)) if heavy else 0.0

        pl_module.log("chem/validity", validity, on_epoch=True, on_step=False)
        pl_module.log("chem/uniqueness", uniqueness, on_epoch=True, on_step=False)
        pl_module.log("chem/heavy_atom_mean", heavy_mean, on_epoch=True, on_step=False)
        pl_module.log(
            "chem/atom_stability", atom_stable_frac, on_epoch=True, on_step=False
        )
        pl_module.log(
            "chem/mol_stability", mol_stable_frac, on_epoch=True, on_step=False
        )

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        try:
            table = wandb.Table(columns=["identifier", "count"])
            for ident, cnt in Counter(valid_ids).most_common(50):
                table.add_data(ident, cnt)
            logger.experiment.log(
                {"chem/valid_smiles": table}, commit=False
            )
        except Exception:
            pass
