"""Callback for chemical validity metrics broken down by atom count."""

from collections import defaultdict

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from ..mol_utils import batch_to_stability, batch_to_validity


class AtomSizeValidityCallback(Callback):
    """Logs validity, atom_stability, mol_stability per atom-count group.

    Produces wandb metrics like chem/validity_n3, chem/atom_stability_n9, etc.
    Requires validation_step to include "num_atoms" (int) in its output dict.
    """

    MAX_MOLS = 512

    def __init__(self):
        self._groups: dict = {}

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._groups = defaultdict(lambda: {"pos": [], "types": [], "batch": [], "offset": 0})

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
        num_atoms = outputs.get("num_atoms")

        if pos is None or a_hard is None or bvec is None or num_atoms is None:
            return

        g = self._groups[num_atoms]
        if g["offset"] >= self.MAX_MOLS:
            return

        g["pos"].append(pos.cpu())
        g["types"].append(a_hard.cpu())
        g["batch"].append(bvec.cpu() + g["offset"])
        g["offset"] += int(bvec.max().item()) + 1

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for num_atoms, g in self._groups.items():
            if not g["pos"]:
                continue

            pos = torch.cat(g["pos"], dim=0)
            types = torch.cat(g["types"], dim=0)
            bvec = torch.cat(g["batch"], dim=0)

            results = batch_to_validity(pos, types, bvec)
            atom_stable_frac, mol_stable_frac = batch_to_stability(pos, types, bvec)

            n_total = len(results)
            n_valid = sum(1 for ok, _ in results if ok)
            validity = n_valid / n_total if n_total > 0 else 0.0

            pl_module.log(f"chem/validity_n{num_atoms}", validity, on_epoch=True, on_step=False)
            pl_module.log(f"chem/atom_stability_n{num_atoms}", atom_stable_frac, on_epoch=True, on_step=False)
            pl_module.log(f"chem/mol_stability_n{num_atoms}", mol_stable_frac, on_epoch=True, on_step=False)
