from collections import Counter

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer

import wandb
from rdkit import rdBase

from ..mol_utils import (
    compact_batch_vec,
    compute_batch_rdkit_validity,
    compute_batch_stability,
    compute_batch_valence_histogram,
    compute_heavy_atom_counts,
    nearest_neighbor_distance_stats,
    pairwise_distance_stats,
)


class ChemicalMetricsCallback(Callback):
    """
    Logs chemical metrics for generated molecules during validation.

    Metrics:
      - chem/validity: strict RDKit validity from 3D coordinates
      - chem/uniqueness: unique canonical SMILES / valid molecules
      - chem/valid_unique: validity * uniqueness
      - chem/atom_stability: fraction of atoms with correct inferred valence
      - chem/mol_stability: fraction of molecules where all atoms are stable
      - chem/heavy_atom_mean: mean non-H atoms per molecule

    Stability is an EDM-style distance-threshold diagnostic, not an RDKit validity
    proxy. Interpret generated stability relative to the real-data baseline logged
    as chem/real_atom_stability and chem/real_mol_stability.
    """

    def __init__(self, max_mols: int = 512, charge: int = 0):
        self.max_mols = max_mols
        self.charge = charge

        self._pos: list[torch.Tensor] = []
        self._atom_types: list[torch.Tensor] = []
        self._batch: list[torch.Tensor] = []
        self._offset: int = 0
        self._real_pos: list[torch.Tensor] = []
        self._real_atom_types: list[torch.Tensor] = []
        self._real_batch: list[torch.Tensor] = []
        self._real_offset: int = 0
        self._printed_rdkit_version = False

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._pos.clear()
        self._atom_types.clear()
        self._batch.clear()
        self._offset = 0
        self._real_pos.clear()
        self._real_atom_types.clear()
        self._real_batch.clear()
        self._real_offset = 0

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

        can_collect_gen = self._offset < self.max_mols
        can_collect_real = self._real_offset < self.max_mols

        if not can_collect_gen and not can_collect_real:
            return

        if can_collect_gen:
            pos = outputs.get("pos_gen")
            atom_types = outputs.get("gen_atom_types")
            batch_vec = outputs.get("gen_batch_vec", outputs.get("batch_vec"))

            if pos is not None and atom_types is not None and batch_vec is not None:
                pos = pos.detach().cpu()
                atom_types = atom_types.detach().cpu()
                batch_vec = batch_vec.detach().cpu().long()

                self._pos.append(pos)
                self._atom_types.append(atom_types)
                self._batch.append(batch_vec + self._offset)

                self._offset += int(batch_vec.max().item()) + 1

        if can_collect_real:
            pos_real = outputs.get("pos_real")
            real_atom_types = outputs.get("real_atom_types")
            real_batch_vec = outputs.get("batch_vec")

            if (
                pos_real is not None
                and real_atom_types is not None
                and real_batch_vec is not None
            ):
                real_batch_vec = real_batch_vec.detach().cpu().long()
                self._real_pos.append(pos_real.detach().cpu())
                self._real_atom_types.append(real_atom_types.detach().cpu())
                self._real_batch.append(real_batch_vec + self._real_offset)
                self._real_offset += int(real_batch_vec.max().item()) + 1

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if not self._pos and not self._real_pos:
            return

        if not self._printed_rdkit_version:
            print(f"[ChemicalMetricsCallback] RDKit version: {rdBase.rdkitVersion}")
            self._printed_rdkit_version = True

        distance_stats = {}
        smiles = []
        valence_hist = {}

        if self._pos:
            pos = torch.cat(self._pos, dim=0)
            atom_types = torch.cat(self._atom_types, dim=0)
            batch_vec = torch.cat(self._batch, dim=0)

            keep = batch_vec < self.max_mols
            pos = pos[keep]
            atom_types = atom_types[keep]
            batch_vec = compact_batch_vec(batch_vec[keep])

            rdkit_metrics = compute_batch_rdkit_validity(
                pos=pos,
                atom_types=atom_types,
                batch_vec=batch_vec,
                charge=self.charge,
            )

            # EDM-style distance-threshold diagnostic, not an RDKit validity proxy.
            # Interpret relative to chem/real_atom_stability and
            # chem/real_mol_stability.
            stability_metrics = compute_batch_stability(
                pos=pos,
                atom_types=atom_types,
                batch_vec=batch_vec,
            )

            heavy_counts = compute_heavy_atom_counts(
                atom_types=atom_types,
                batch_vec=batch_vec,
            )

            valence_hist = compute_batch_valence_histogram(
                pos=pos,
                atom_types=atom_types,
                batch_vec=batch_vec,
            )
            distance_stats.update(
                {
                    "gen_nearest_neighbor": nearest_neighbor_distance_stats(
                        pos, batch_vec
                    ),
                    "gen_pairwise": pairwise_distance_stats(pos, batch_vec),
                }
            )

            validity = rdkit_metrics["validity"]
            uniqueness = rdkit_metrics["uniqueness"]
            valid_unique = rdkit_metrics["valid_unique"]
            smiles = rdkit_metrics["smiles"]

            atom_stability = stability_metrics["atom_stability"]
            mol_stability = stability_metrics["mol_stability"]

            heavy_atom_mean = float(np.mean(heavy_counts)) if heavy_counts else 0.0

            pl_module.log("chem/validity", validity, on_epoch=True, on_step=False)
            pl_module.log("chem/uniqueness", uniqueness, on_epoch=True, on_step=False)
            pl_module.log(
                "chem/valid_unique", valid_unique, on_epoch=True, on_step=False
            )
            pl_module.log(
                "chem/atom_stability",
                atom_stability,
                on_epoch=True,
                on_step=False,
            )
            pl_module.log(
                "chem/mol_stability",
                mol_stability,
                on_epoch=True,
                on_step=False,
            )
            pl_module.log(
                "chem/heavy_atom_mean",
                heavy_atom_mean,
                on_epoch=True,
                on_step=False,
            )

        real_valence_hist = {}
        if self._real_pos:
            pos_real = torch.cat(self._real_pos, dim=0)
            real_atom_types = torch.cat(self._real_atom_types, dim=0)
            real_batch_vec = torch.cat(self._real_batch, dim=0)

            real_keep = real_batch_vec < self.max_mols
            pos_real = pos_real[real_keep]
            real_atom_types = real_atom_types[real_keep]
            real_batch_vec = compact_batch_vec(real_batch_vec[real_keep])

            real_stability_metrics = compute_batch_stability(
                pos=pos_real,
                atom_types=real_atom_types,
                batch_vec=real_batch_vec,
            )

            real_valence_hist = compute_batch_valence_histogram(
                pos=pos_real,
                atom_types=real_atom_types,
                batch_vec=real_batch_vec,
            )
            distance_stats.update(
                {
                    "real_nearest_neighbor": nearest_neighbor_distance_stats(
                        pos_real, real_batch_vec
                    ),
                    "real_pairwise": pairwise_distance_stats(pos_real, real_batch_vec),
                }
            )

            pl_module.log(
                "chem/real_atom_stability",
                real_stability_metrics["atom_stability"],
                on_epoch=True,
                on_step=False,
            )
            pl_module.log(
                "chem/real_mol_stability",
                real_stability_metrics["mol_stability"],
                on_epoch=True,
                on_step=False,
            )

        for group, stats in distance_stats.items():
            for key, value in stats.items():
                pl_module.log(
                    f"chem/{group}_{key}",
                    value,
                    on_epoch=True,
                    on_step=False,
                )

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        try:
            smiles_table = wandb.Table(columns=["smiles", "count"])
            for smi, count in Counter(smiles).most_common(50):
                smiles_table.add_data(smi, count)

            valence_table = wandb.Table(columns=["element", "valence", "count"])
            for element, counts in valence_hist.items():
                for valence, count in sorted(counts.items()):
                    valence_table.add_data(element, int(valence), int(count))

            log_payload = {
                "chem/valid_smiles": smiles_table,
                "chem/gen_valence_histogram": valence_table,
            }

            if real_valence_hist:
                real_valence_table = wandb.Table(
                    columns=["element", "valence", "count"]
                )
                for element, counts in real_valence_hist.items():
                    for valence, count in sorted(counts.items()):
                        real_valence_table.add_data(element, int(valence), int(count))
                log_payload["chem/real_valence_histogram"] = real_valence_table

            logger.experiment.summary["chem/rdkit_version"] = rdBase.rdkitVersion
            logger.experiment.log(log_payload, step=trainer.global_step)

        except Exception as exc:
            print(f"[ChemicalMetricsCallback] WandB logging failed: {exc}")
