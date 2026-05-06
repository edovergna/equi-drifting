import io

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from lightning.pytorch import Callback, LightningModule, Trainer

import wandb

_ATOM_NAMES = ["H", "C", "N", "O", "F"]
_ATOM_COLORS = ["lightgray", "dimgray", "steelblue", "tomato", "limegreen"]


class MoleculeVisualizationCallback(Callback):
    """
    Logs 3D molecule renders to WandB each validation epoch.

    Uses the first validation batch as a fixed reference so the same molecules
    are shown across all epochs, making qualitative progress easy to track.

    Renders four panels per epoch:
      - mol/random_gen   — K random generated molecules
      - mol/best_gen     — K closest to a real molecule (by embedding NN distance)
      - mol/worst_gen    — K furthest from any real molecule
      - mol/real_ref     — the K corresponding real reference molecules
      - mol/atom_type_dist — bar chart: predicted vs real atom type fractions
    """

    _REQUIRED_KEYS = {
        "phi_gen",
        "phi_real",
        "pos_gen",
        "gen_atom_types",
        "pos_real",
        "real_atom_types",
        "gen_batch_vec",
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
        self._gen_atom_types.append(outputs["gen_atom_types"].cpu())
        self._real_atom_types.append(outputs["real_atom_types"].argmax(dim=-1).cpu())

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
            gen_batch_vec = ref["gen_batch_vec"]
            real_batch_vec = ref["batch_vec"]
            n_graphs = int(phi_gen.shape[0])

            nn_dists = torch.cdist(phi_gen, phi_real).min(dim=1).values
            all_idx = list(range(n_graphs))
            ranked = sorted(all_idx, key=lambda i: nn_dists[i].item())

            random_idx = all_idx[: self.n_molecules]
            best_idx = ranked[: self.n_molecules]
            worst_idx = ranked[-self.n_molecules :]

            def render_group(indices, pos, atom_types, bvec, label_prefix):
                return [
                    self._render_mol(pos, atom_types, bvec, i, f"{label_prefix} #{i}")
                    for i in indices
                ]

            real_atom_types = ref["real_atom_types"].argmax(dim=-1)
            images = {
                "mol/random_gen": render_group(
                    random_idx, ref["pos_gen"], ref["gen_atom_types"], gen_batch_vec, "gen"
                ),
                "mol/best_gen": render_group(
                    best_idx,
                    ref["pos_gen"],
                    ref["gen_atom_types"],
                    gen_batch_vec,
                    f"best d={nn_dists[best_idx[0]]:.2f}",
                ),
                "mol/worst_gen": render_group(
                    worst_idx,
                    ref["pos_gen"],
                    ref["gen_atom_types"],
                    gen_batch_vec,
                    f"worst d={nn_dists[worst_idx[0]]:.2f}",
                ),
                "mol/real_ref": render_group(
                    random_idx, ref["pos_real"], real_atom_types, real_batch_vec, "real"
                ),
            }

            if gen_types is not None and real_types is not None:
                images["mol/atom_type_dist"] = self._atom_dist_chart(gen_types, real_types)

            logger.experiment.log(images, step=trainer.global_step)

        except Exception as e:
            print(f"[MoleculeVisualizationCallback] Skipped: {e}")

    def _render_mol(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        batch_vec: torch.Tensor,
        graph_idx: int,
        title: str = "",
    ) -> "wandb.Image":

        mask = batch_vec == graph_idx
        p = pos[mask].numpy()
        types = atom_types[mask].numpy()

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")

        for t, (color, name) in enumerate(zip(_ATOM_COLORS, _ATOM_NAMES)):
            m = types == t
            if m.any():
                ax.scatter(
                    p[m, 0], p[m, 1], p[m, 2],
                    c=color, s=120, label=name,
                    depthshade=True, edgecolors="k", linewidths=0.3,
                )

        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if np.linalg.norm(p[i] - p[j]) < self.bond_threshold:
                    ax.plot(
                        [p[i, 0], p[j, 0]],
                        [p[i, 1], p[j, 1]],
                        [p[i, 2], p[j, 2]],
                        "k-", alpha=0.25, linewidth=0.8,
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
