import io
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.optim import Optimizer

from .mol_utils import batch_to_validity, heavy_atom_counts

# QM9 atom ordering from EncodeAtomTypesTransform: {H, C, N, O, F}
_ATOM_NAMES = ["H", "C", "N", "O", "F"]
_ATOM_COLORS = ["lightgray", "dimgray", "steelblue", "tomato", "limegreen"]

import wandb


class GradientMonitorCallback(Callback):
    """Logs pre-clip gradient norms every training step."""

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        grads = [p.grad for p in pl_module.generator.parameters() if p.grad is not None]
        if not grads:
            return
        total_norm = torch.stack([g.detach().norm(2) for g in grads]).norm(2)
        max_abs = torch.stack([g.detach().abs().max() for g in grads]).max()
        pl_module.log("grad/total_norm", total_norm, on_step=True, on_epoch=False)
        pl_module.log("grad/max_abs", max_abs, on_step=True, on_epoch=False)


class EmbeddingMonitorCallback(Callback):
    """
    Collects φ_gen and φ_real from validation_step outputs and logs WandB histograms
    at the end of each validation epoch.

    validation_step must return a dict containing "phi_gen" and "phi_real" (CPU tensors).
    """

    MAX_EMBEDDINGS = 2048  # cap to avoid OOM during large cosine-sim matmul

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

        # Cosine similarity of each generated embedding to its nearest real neighbour
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
                    "val/cosine_sim_to_nn_hist": wandb.Histogram(cos_sim),
                    "val/phi_gen_norm_hist": wandb.Histogram(gen_norms),
                    "val/phi_real_norm_hist": wandb.Histogram(real_norms),
                },
                step=trainer.global_step,
            )
        except Exception:
            pass  # non-WandB loggers: silently skip histograms


class MoleculeVisualizationCallback(Callback):
    """
    Logs 3D molecule renders to WandB each validation epoch.

    Uses the first validation batch as a fixed reference so the same molecules
    are shown across all epochs, making qualitative progress easy to track.

    Renders four panels per epoch:
      - val/molecules/random_gen   — K random generated molecules
      - val/molecules/best_gen     — K closest to a real molecule (by embedding NN distance)
      - val/molecules/worst_gen    — K furthest from any real molecule
      - val/molecules/real_ref     — the K corresponding real reference molecules
      - val/atom_type_dist         — bar chart: predicted vs real atom type fractions
    """

    _REQUIRED_KEYS = {
        "phi_gen",
        "phi_real",
        "pos_gen",
        "a_soft_gen",
        "pos_real",
        "a_soft_real",
        "batch_vec",
    }

    def __init__(
        self, n_molecules: int = 4, bond_threshold: float = 2.0, every_n_epochs: int = 1
    ):
        self.n_molecules = n_molecules
        self.bond_threshold = bond_threshold
        self.every_n_epochs = every_n_epochs
        self._ref: dict | None = None
        self._gen_atom_types: list[torch.Tensor] = []
        self._real_atom_types: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Collection hooks
    # ------------------------------------------------------------------

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict) or not self._REQUIRED_KEYS.issubset(outputs):
            return
        if batch_idx == 0:
            self._ref = {k: outputs[k] for k in self._REQUIRED_KEYS}
        self._gen_atom_types.append(outputs["a_soft_gen"].argmax(dim=-1).cpu())
        self._real_atom_types.append(outputs["a_soft_real"].argmax(dim=-1).cpu())

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
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
            phi_gen = ref["phi_gen"].float()
            phi_real = ref["phi_real"].float()
            batch_vec = ref["batch_vec"]
            n_graphs = int(batch_vec.max().item()) + 1

            nn_dists = torch.cdist(phi_gen, phi_real).min(dim=1).values  # [G]
            all_idx = list(range(n_graphs))
            ranked = sorted(all_idx, key=lambda i: nn_dists[i].item())

            random_idx = all_idx[: self.n_molecules]
            best_idx = ranked[: self.n_molecules]
            worst_idx = ranked[-self.n_molecules :]

            def render_group(indices, pos, a_soft, label_prefix):
                return [
                    self._render_mol(pos, a_soft, batch_vec, i, f"{label_prefix} #{i}")
                    for i in indices
                ]

            images = {
                "val/molecules/random_gen": render_group(
                    random_idx, ref["pos_gen"], ref["a_soft_gen"], "gen"
                ),
                "val/molecules/best_gen": render_group(
                    best_idx,
                    ref["pos_gen"],
                    ref["a_soft_gen"],
                    f"best d={nn_dists[best_idx[0]]:.2f}",
                ),
                "val/molecules/worst_gen": render_group(
                    worst_idx,
                    ref["pos_gen"],
                    ref["a_soft_gen"],
                    f"worst d={nn_dists[worst_idx[0]]:.2f}",
                ),
                "val/molecules/real_ref": render_group(
                    random_idx, ref["pos_real"], ref["a_soft_real"], "real"
                ),
            }

            if gen_types is not None and real_types is not None:
                images["val/atom_type_dist"] = self._atom_dist_chart(
                    gen_types, real_types
                )

            logger.experiment.log(images, step=trainer.global_step)

        except Exception as e:
            print(f"[MoleculeVisualizationCallback] Skipped: {e}")

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_mol(
        self,
        pos: torch.Tensor,
        a_soft: torch.Tensor,
        batch_vec: torch.Tensor,
        graph_idx: int,
        title: str = "",
    ) -> "wandb.Image":
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        mask = batch_vec == graph_idx
        p = pos[mask].numpy()
        types = a_soft[mask].argmax(dim=-1).numpy()

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")

        for t, (color, name) in enumerate(zip(_ATOM_COLORS, _ATOM_NAMES)):
            m = types == t
            if m.any():
                ax.scatter(
                    p[m, 0],
                    p[m, 1],
                    p[m, 2],
                    c=color,
                    s=120,
                    label=name,
                    depthshade=True,
                    edgecolors="k",
                    linewidths=0.3,
                )

        # Draw bonds between atoms closer than the threshold
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if np.linalg.norm(p[i] - p[j]) < self.bond_threshold:
                    ax.plot(
                        [p[i, 0], p[j, 0]],
                        [p[i, 1], p[j, 1]],
                        [p[i, 2], p[j, 2]],
                        "k-",
                        alpha=0.25,
                        linewidth=0.8,
                    )

        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper right", fontsize=6, markerscale=0.7)
        ax.set_box_aspect([1, 1, 1])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return wandb.Image(PILImage.open(buf).copy())

    def _atom_dist_chart(
        self, gen_types: torch.Tensor, real_types: torch.Tensor
    ) -> "wandb.Image":
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

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
        self._asoft: list[torch.Tensor] = []
        self._batch: list[torch.Tensor] = []
        self._offset: int = 0

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._pos.clear()
        self._asoft.clear()
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
        a_soft = outputs.get("a_soft_gen")
        bvec = outputs.get("batch_vec")
        if pos is None or a_soft is None or bvec is None:
            return
        if self._offset >= self.MAX_MOLS:
            return

        self._pos.append(pos.cpu())
        self._asoft.append(a_soft.cpu())
        self._batch.append(bvec.cpu() + self._offset)
        self._offset += int(bvec.max().item()) + 1

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self._pos:
            return

        pos = torch.cat(self._pos, dim=0)
        a_soft = torch.cat(self._asoft, dim=0)
        bvec = torch.cat(self._batch, dim=0)

        # cap at MAX_MOLS graphs
        n_graphs = int(bvec.max().item()) + 1
        if n_graphs > self.MAX_MOLS:
            keep = bvec < self.MAX_MOLS
            pos, a_soft, bvec = pos[keep], a_soft[keep], bvec[keep]

        results = batch_to_validity(pos, a_soft, bvec)
        heavy = heavy_atom_counts(a_soft, bvec)

        n_total = len(results)
        n_valid = sum(1 for ok, _ in results if ok)
        valid_ids = [ident for ok, ident in results if ok and ident is not None]
        uniqueness = len(set(valid_ids)) / len(valid_ids) if valid_ids else 0.0
        validity = n_valid / n_total if n_total > 0 else 0.0
        heavy_mean = float(np.mean(heavy)) if heavy else 0.0

        pl_module.log("chem/validity", validity, on_epoch=True, on_step=False)
        pl_module.log("chem/uniqueness", uniqueness, on_epoch=True, on_step=False)
        pl_module.log("chem/heavy_atom_mean", heavy_mean, on_epoch=True, on_step=False)

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        try:
            table = wandb.Table(columns=["identifier", "count"])
            for ident, cnt in Counter(valid_ids).most_common(50):
                table.add_data(ident, cnt)
            logger.experiment.log(
                {"chem/valid_smiles": table}, step=trainer.global_step
            )
        except Exception:
            pass
