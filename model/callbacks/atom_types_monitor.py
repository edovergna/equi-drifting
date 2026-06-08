"""Callback for logging atom-type diagnostics for type-only generation."""

import io
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from PIL import Image as PILImage

import wandb

_ATOM_NAMES = ["H", "C", "N", "O", "F"]

class AtomTypesCallback(Callback):
    _REQUIRED_KEYS = {
        "real_atom_types",
        "gen_atom_types",
        "gen_types_prob",
    }

    def __init__(
        self,
        every_n_epochs: int = 1,
    ):
        """Initialize the molecule atom types callback.

        Args:
            every_n_epochs: Log visualizations every n validation epochs.
        """
        self.every_n_epochs = every_n_epochs
        self._ref: dict | None = None
        self._gen_atom_types: list[torch.Tensor] = []
        self._real_atom_types: list[torch.Tensor] = []
        self._gen_atom_types_by_size: dict[int, list[torch.Tensor]] = defaultdict(list)
        self._real_atom_types_by_size: dict[int, list[torch.Tensor]] = defaultdict(list)
        self._gen_types_prob_by_size: dict[int, list[torch.Tensor]] = defaultdict(list)
        self._sample_rows_by_size: dict[int, tuple[str, str]] = {}
        self._gen_sequences_by_size: dict[int, list[tuple[int, ...]]] = defaultdict(list)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate outputs from first validation batch and all atom types.

        Args:
            trainer: PyTorch Lightning Trainer.
            pl_module: Lightning module being trained.
            outputs: Output from the validation step (expects dict with required keys).
            batch: Current batch data.
            batch_idx: Index of current batch.
            dataloader_idx: Index if using multiple dataloaders.
        """
        if not isinstance(outputs, dict) or not self._REQUIRED_KEYS.issubset(outputs):
            return
        if batch_idx == 0:
            self._ref = {k: outputs[k] for k in self._REQUIRED_KEYS}
            self._ref["gen_atom_types"] = outputs.get("gen_atom_types")
        gen_atom_types = outputs.get("gen_atom_types")
        gen_types_prob = outputs.get("gen_types_prob")
        real_atom_types_raw = outputs["real_atom_types"].argmax(dim=-1).cpu()
        real_atom_types = real_atom_types_raw.reshape(-1)
        num_atoms = self._batch_num_atoms(batch, gen_atom_types)
        if gen_types_prob is not None:
            gen_types_prob = gen_types_prob.cpu()
            self._gen_types_prob_by_size[num_atoms].append(
                gen_types_prob.reshape(-1, gen_types_prob.shape[-1])
            )
        if gen_atom_types is not None and num_atoms not in self._sample_rows_by_size:
            self._sample_rows_by_size[num_atoms] = (
                self._format_atom_sequence(gen_atom_types.cpu(), num_atoms),
                self._format_atom_sequence(real_atom_types_raw, num_atoms),
            )
        if gen_atom_types is not None:
            gen_atom_types = gen_atom_types.cpu()
            self._gen_sequences_by_size[num_atoms].extend(
                self._ordered_atom_sequences(gen_atom_types, num_atoms)
            )
            gen_atom_types = gen_atom_types.reshape(-1)
            self._gen_atom_types.append(gen_atom_types)
            self._gen_atom_types_by_size[num_atoms].append(gen_atom_types)
        self._real_atom_types.append(real_atom_types)
        self._real_atom_types_by_size[num_atoms].append(real_atom_types)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log atom-type diagnostics to wandb at validation epoch end.

        Args:
            trainer: PyTorch Lightning Trainer.
            pl_module: Lightning module being trained.
        """
        gen_types = torch.cat(self._gen_atom_types) if self._gen_atom_types else None
        real_types = torch.cat(self._real_atom_types) if self._real_atom_types else None
        gen_types_by_size = {
            size: torch.cat(parts)
            for size, parts in self._gen_atom_types_by_size.items()
            if parts
        }
        real_types_by_size = {
            size: torch.cat(parts)
            for size, parts in self._real_atom_types_by_size.items()
            if parts
        }
        probability_stats_by_size = {
            size: self._probability_stats(torch.cat(parts))
            for size, parts in self._gen_types_prob_by_size.items()
            if parts
        }
        sample_rows_by_size = dict(self._sample_rows_by_size)
        unique_counts_by_size = {
            size: len(set(sequences))
            for size, sequences in self._gen_sequences_by_size.items()
            if sequences
        }
        heavy_atom_mean_by_size = {
            size: float(np.mean([sum(type_idx != 0 for type_idx in sequence) for sequence in sequences]))
            for size, sequences in self._gen_sequences_by_size.items()
            if sequences
        }
        self._gen_atom_types.clear()
        self._real_atom_types.clear()
        self._gen_atom_types_by_size.clear()
        self._real_atom_types_by_size.clear()
        self._gen_types_prob_by_size.clear()
        self._sample_rows_by_size.clear()
        self._gen_sequences_by_size.clear()

        if trainer.current_epoch % self.every_n_epochs != 0 or self._ref is None:
            return
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return

        try:
            ref = self._ref

            real_atom_types = ref["real_atom_types"].argmax(dim=-1)
            images = {}

            if gen_types is not None and real_types is not None:
                images["mol/atom_type_dist"] = self._atom_dist_chart(
                    gen_types, real_types
                )
                for size in sorted(gen_types_by_size):
                    if size not in real_types_by_size:
                        continue
                    images[f"mol/atom_type_dist_size_{size}"] = self._atom_dist_chart(
                        gen_types_by_size[size],
                        real_types_by_size[size],
                        title=f"Atom type distribution ({size} atoms)",
                    )
                if sample_rows_by_size:
                    sample_table = wandb.Table(
                        columns=["num_atoms", "generated", "real_reference"]
                    )
                    for size in sorted(sample_rows_by_size):
                        generated, real_reference = sample_rows_by_size[size]
                        sample_table.add_data(size, generated, real_reference)
                    images["mol/atom_type_samples"] = sample_table
                if unique_counts_by_size:
                    images["mol/atom_type_uniqueness_by_size"] = (
                        self._uniqueness_by_size_chart(unique_counts_by_size)
                    )
                if heavy_atom_mean_by_size:
                    images["mol/heavy_atom_mean_by_size"] = (
                        self._heavy_atom_mean_by_size_chart(heavy_atom_mean_by_size)
                    )
                if probability_stats_by_size:
                    images["mol/probability_entropy_by_size"] = (
                        self._probability_entropy_by_size_chart(
                            probability_stats_by_size
                        )
                    )

            logger.experiment.log(images, commit=False)

        except Exception as e:
            print(f"[AtomTypesMonitor] Skipped: {e}")


    def _atom_dist_chart(
        self,
        gen_types: torch.Tensor,
        real_types: torch.Tensor,
        title: str = "Atom type distribution",
    ) -> "wandb.Image":
        """Create bar chart comparing atom-type distributions.

        Args:
            gen_types: Generated atom type indices [total_nodes].
            real_types: Real atom type indices [total_nodes].

        Returns:
            WandB Image object with side-by-side bar chart.
        """
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
        ax.set_title(title)
        ax.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return wandb.Image(PILImage.open(buf).copy())

    def _batch_num_atoms(self, batch, gen_atom_types: torch.Tensor | None) -> int:
        if hasattr(batch, "ptr") and batch.ptr.numel() > 1:
            return int((batch.ptr[1] - batch.ptr[0]).item())
        if gen_atom_types is not None and gen_atom_types.ndim >= 2:
            return int(gen_atom_types.shape[-1])
        return -1

    def _format_atom_sequence(self, atom_types: torch.Tensor, num_atoms: int) -> str:
        atom_types = atom_types.reshape(-1)[:num_atoms].long()
        return " ".join(_ATOM_NAMES[int(type_idx)] for type_idx in atom_types)

    def _ordered_atom_sequences(
        self, atom_types: torch.Tensor, num_atoms: int
    ) -> list[tuple[int, ...]]:
        if num_atoms <= 0:
            return []
        atom_types = atom_types.reshape(-1).long()
        n_complete = atom_types.numel() // num_atoms
        atom_types = atom_types[: n_complete * num_atoms].reshape(n_complete, num_atoms)
        return [tuple(row.tolist()) for row in atom_types]

    def _uniqueness_by_size_chart(
        self, unique_counts_by_size: dict[int, int]
    ) -> "wandb.Image":
        sizes = sorted(unique_counts_by_size)
        counts = [unique_counts_by_size[size] for size in sizes]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar([str(size) for size in sizes], counts, color="mediumseagreen", alpha=0.85)
        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("Unique generated sequences")
        ax.set_title("Order-sensitive atom-type uniqueness by size")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return wandb.Image(PILImage.open(buf).copy())

    def _heavy_atom_mean_by_size_chart(
        self, heavy_atom_mean_by_size: dict[int, float]
    ) -> "wandb.Image":
        sizes = sorted(heavy_atom_mean_by_size)
        means = [heavy_atom_mean_by_size[size] for size in sizes]

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar([str(size) for size in sizes], means, color="slateblue", alpha=0.85)
        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("Average non-H atoms")
        ax.set_title("Generated heavy-atom count by size")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return wandb.Image(PILImage.open(buf).copy())

    def _probability_stats(self, probs: torch.Tensor) -> tuple[float, float, float]:
        probs = probs.float()
        num_types = probs.shape[-1]
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        normalized_entropy = entropy / np.log(num_types)
        mean_max_prob = probs.max(dim=-1).values.mean()
        uniform_max_prob = 1.0 / num_types
        return (
            float(normalized_entropy.mean().item()),
            float(mean_max_prob.item()),
            uniform_max_prob,
        )

    def _probability_entropy_by_size_chart(
        self, probability_stats_by_size: dict[int, tuple[float, float, float]]
    ) -> "wandb.Image":
        sizes = sorted(probability_stats_by_size)
        normalized_entropy = [probability_stats_by_size[size][0] for size in sizes]
        mean_max_prob = [probability_stats_by_size[size][1] for size in sizes]
        uniform_max_prob = probability_stats_by_size[sizes[0]][2]
        x = np.arange(len(sizes))

        fig, ax = plt.subplots(figsize=(7, 3.5))
        entropy_bars = ax.bar(
            x,
            normalized_entropy,
            color="mediumpurple",
            alpha=0.75,
            label="Normalized entropy",
        )
        ax2 = ax.twinx()
        max_prob_line = ax2.plot(
            x,
            mean_max_prob,
            color="darkorange",
            marker="o",
            linewidth=2,
            label="Mean max probability per atom",
        )
        uniform_line = ax2.axhline(
            uniform_max_prob,
            color="darkorange",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Uniform max-prob baseline ({uniform_max_prob:.2f})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.set_ylim(0.0, 1.05)
        ax2.set_ylim(0.0, 1.05)
        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("Normalized entropy (1 = uniform, 0 = one-hot)")
        ax2.set_ylabel("Mean max probability (1 = one-hot)")
        ax.set_title("Generated probability sharpness by size")
        handles = [entropy_bars, *max_prob_line, uniform_line]
        labels = [handle.get_label() for handle in handles]
        ax.legend(handles, labels, fontsize=8, loc="best")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return wandb.Image(PILImage.open(buf).copy())
