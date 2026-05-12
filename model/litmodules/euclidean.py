from pathlib import Path

import torch
import torch.nn.functional as F

from ept.ept_loader import load_ept_feature_extractor

from ..losses import TrainingDivergedException
from ..losses.euclidean import (
    compute_inverse_attn_drift_loss,
    compute_norm_based_drift_loss,
    compute_position_drift_loss,
    original_compute_drift_loss,
)
from ..egnn import EGNN
from ..geometry import center_positions_per_graph, per_graph_center_norms
from ..mol_utils import infer_types_from_pos_batch
from .base import BaseDriftingMoleculeGenerator


class EuclideanGenerator(BaseDriftingMoleculeGenerator):
    _SAVE_COMPONENTS = ["generator"]
    _LOAD_COMPONENTS = ["generator"]

    def __init__(self, generator_cfg=None, drift_cfg=None):
        default_generator_cfg = {
            "hidden_nf": 128,
            "n_layers": 2,
            "num_atom_types": 5,
            "num_bond_types": 5,
            "coordinate_clamp_range": 3.0,
            "predict_bond_types": False,
            "predict_atom_types": True,
            "pos_clamp": 20.0,
            "pos_clamp_type": "geom",
            "c_pos_clamp": 5.0,
            "p_pos_clamp": 4.0,
            "norm_pos_clamp": 10.0,
            "prior_pos_clamp": 3.0,
            "use_feature_extractor": True,
            "infer_types_from_pos": False,
            "infer_method": "heuristic",
        }
        default_drift_cfg = {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "temperatures": [0.02, 0.05, 0.2],
            "loss_variant": "norm_based",
            "atom_type_temp": 1.0,
            "pct_start": 0.1,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
        }

        merged_generator_cfg = {**default_generator_cfg, **(generator_cfg or {})}
        merged_drift_cfg = {**default_drift_cfg, **(drift_cfg or {})}

        super().__init__(merged_generator_cfg, merged_drift_cfg)
        self.save_hyperparameters(
            {"generator_cfg": self.generator_cfg, "drift_cfg": self.drift_cfg}
        )

        self.generator = self._init_generator(self.generator_cfg)
        self.use_feature_extractor = self.generator_cfg["use_feature_extractor"]
        self.feature_extractor = (
            self._init_feature_extractor() if self.use_feature_extractor else None
        )
        if self.feature_extractor is not None:
            self._freeze_feature_extractor()
            self._SAVE_COMPONENTS = ["generator", "feature_extractor"]
            self._LOAD_COMPONENTS = ["generator", "feature_extractor"]

        self.temperatures = self.drift_cfg["temperatures"]
        self.loss_variant = self.drift_cfg["loss_variant"]
        self.atom_type_temp = self.drift_cfg.get("atom_type_temp", 1.0)
        self.n_gen_molecules = self.drift_cfg.get("n_gen_molecules", 64)
        self.pos_clamp = self.generator_cfg["pos_clamp"]
        self.pos_clamp_type = self.generator_cfg["pos_clamp_type"]
        self.c_pos_clamp = self.generator_cfg["c_pos_clamp"]
        self.p_pos_clamp = self.generator_cfg["p_pos_clamp"]
        self.norm_pos_clamp = self.generator_cfg["norm_pos_clamp"]
        self.infer_types_from_pos = self.generator_cfg.get("infer_types_from_pos", False)
        self.infer_method = self.generator_cfg.get("infer_method", "heuristic")

        self._norm_rescale_grad: float | None = None

    def _init_generator(self, cfg) -> EGNN:
        return EGNN(
            hidden_nf=cfg["hidden_nf"],
            n_layers=cfg["n_layers"],
            num_atom_types=cfg["num_atom_types"],
            num_bond_types=cfg["num_bond_types"],
            predict_bond_types=cfg["predict_bond_types"],
            predict_atom_types=cfg["predict_atom_types"],
        )

    def _init_feature_extractor(self):
        return load_ept_feature_extractor()

    def _freeze_feature_extractor(self) -> None:
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def _is_feature_extractor_trainable(self) -> bool:
        if self.feature_extractor is None:
            return False
        return any(p.requires_grad for p in self.feature_extractor.parameters())

    def _forward(self, batch):
        """Shared forward pass: prior → EGNN → center → hard atoms → EPT embeddings."""
        x_prior, pos_prior, gen_batch_vec, gen_dense_edge_index = (
            self._sample_prior_batch(self.n_gen_molecules)
        )

        x_gen, _, pos_gen = self.generator(x_prior, pos_prior, gen_dense_edge_index)
        pos_gen = center_positions_per_graph(pos_gen, gen_batch_vec)

        self._norm_rescale_grad = None
        if self.pos_clamp_type == "hard":
            pos_gen = pos_gen.clamp(-self.pos_clamp, self.pos_clamp)
        elif self.pos_clamp_type == "tanh":
            norm = pos_gen.norm(dim=-1, keepdim=True)
            rescale = torch.tanh(norm / self.norm_pos_clamp) / (
                norm / self.norm_pos_clamp + 1e-8
            )
            if not self.trainer.sanity_checking and self.trainer.validating is False:
                rescale.register_hook(
                    lambda g: setattr(self, "_norm_rescale_grad", g.abs().mean().item())
                )
            pos_gen = pos_gen * rescale
        else:  # geom
            norm = pos_gen.norm(dim=-1)
            rescale = 1 / (1 + (norm / self.c_pos_clamp) ** self.p_pos_clamp)
            if not self.trainer.sanity_checking and self.trainer.validating is False:
                rescale.register_hook(
                    lambda g: setattr(self, "_norm_rescale_grad", g.abs().mean().item())
                )
            pos_gen = pos_gen * rescale.unsqueeze(-1)

        if self.infer_types_from_pos:
            with torch.no_grad():
                gen_atom_types = infer_types_from_pos_batch(
                    pos_gen, gen_batch_vec, self.device,
                    self.generator_cfg["num_atom_types"], self.infer_method,
                )
        else:
            gen_atom_types = (
                F.gumbel_softmax(x_gen, tau=self.atom_type_temp, hard=True)
                if x_gen is not None
                else None
            )

        if not self.use_feature_extractor:
            phi_gen = pos_gen
            phi_real = batch.pos
        else:
            phi_gen = self.feature_extractor(
                pos=pos_gen,
                atom_types=gen_atom_types,
                block_id=torch.arange(pos_gen.shape[0], device=self.device),
                batch_id=gen_batch_vec,
                dense_edge_index=gen_dense_edge_index,
            )
            phi_real = self.feature_extractor(
                pos=batch.pos,
                atom_types=batch.real_atom_types,
                block_id=torch.arange(batch.num_nodes, device=batch.batch.device),
                batch_id=batch.batch,
                dense_edge_index=batch.dense_edge_index,
            )

        return pos_gen, gen_atom_types, phi_gen, phi_real, gen_batch_vec

    def _compute_loss(
        self,
        phi_gen: torch.Tensor,
        phi_real: torch.Tensor,
        gen_batch_vec: torch.Tensor | None = None,
        real_batch_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if not self.use_feature_extractor:
            return compute_position_drift_loss(
                phi_gen, phi_real, gen_batch_vec, real_batch_vec, self.temperatures
            )
        if self.loss_variant == "original":
            return original_compute_drift_loss(phi_gen, phi_real, self.temperatures)
        elif self.loss_variant == "inverse_attn":
            return compute_inverse_attn_drift_loss(phi_gen, phi_real, self.temperatures)
        else:
            return compute_norm_based_drift_loss(phi_gen, phi_real, self.temperatures)

    def on_after_backward(self):
        if self._norm_rescale_grad is not None:
            self.log(
                "geom/norm_rescale_grad",
                self._norm_rescale_grad,
                on_step=True,
                on_epoch=False,
            )

    def training_step(self, batch, batch_idx):
        pos_gen, _, phi_gen, phi_real, gen_batch_vec = self._forward(batch)

        if not (torch.isfinite(phi_gen).all() and torch.isfinite(phi_real).all()):
            if self.feature_extractor is None:
                self.print(
                    f"\n[Step {self.global_step}] Non-finite positions found. "
                    "Stopping training."
                )
                self.trainer.should_stop = True
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            with torch.no_grad():
                bad_gen_mask = ~torch.isfinite(phi_gen).all(dim=-1)
                bad_real_mask = ~torch.isfinite(phi_real).all(dim=-1)
                bad_mol_mask = bad_gen_mask | bad_real_mask

                n_bad_gen = bad_gen_mask.sum().item()
                n_bad_real = bad_real_mask.sum().item()
                n_total = bad_mol_mask.shape[0]

                atom_mask = bad_mol_mask[gen_batch_vec]
                pos_bad = pos_gen[atom_mask]

                pos_norms_bad = pos_bad.norm(dim=-1)
                max_dist_bad = (
                    torch.cdist(pos_bad, pos_bad).max()
                    if pos_bad.shape[0] > 1
                    else pos_bad.new_tensor(0.0)
                )

                gen_center_norms = per_graph_center_norms(pos_gen, gen_batch_vec)
                real_center_norms = per_graph_center_norms(batch.pos, batch.batch)
                bad_gen_cn = gen_center_norms[bad_mol_mask]
                bad_real_cn = real_center_norms[bad_mol_mask]

                self.print(
                    f"\n[Step {self.global_step}] Non-finite embeddings — "
                    f"{n_bad_gen} gen mol(s), {n_bad_real} real mol(s) out of {n_total}.\n"
                    f"  [bad mols] pos_gen norm   mean={pos_norms_bad.mean():.3f}  std={pos_norms_bad.std():.3f}\n"
                    f"  [bad mols] max pairwise dist={max_dist_bad:.3f}\n"
                    f"  [bad mols] gen center norm  mean={bad_gen_cn.mean():.3f}  std={bad_gen_cn.std():.3f}\n"
                    f"  [bad mols] real center norm mean={bad_real_cn.mean():.3f}  std={bad_real_cn.std():.3f}\n"
                    f"  Stopping."
                )

            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        try:
            loss, stats = self._compute_loss(phi_gen, phi_real, gen_batch_vec, batch.batch)
        except TrainingDivergedException as e:
            self.print(f"\n[Step {self.global_step}] {e}\nStopping training.")
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        bs = self.n_gen_molecules
        self.log("train_loss", loss, batch_size=bs, on_step=True, on_epoch=True)
        self._log_drift_stats(stats, "drift_train", bs, on_step=True, on_epoch=False)
        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )
        self._log_geometry(pos_gen, gen_batch_vec, batch.pos, batch.batch, bs)

        return loss

    def validation_step(self, batch, batch_idx):
        pos_gen, gen_atom_types, phi_gen, phi_real, gen_batch_vec = self._forward(batch)
        val_loss, stats = self._compute_loss(phi_gen, phi_real, gen_batch_vec, batch.batch)

        bs = self.n_gen_molecules
        self.log(
            "val_loss", val_loss, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True
        )
        self._log_drift_stats(
            stats, "drift_val", bs,
            on_step=False, on_epoch=True, sync_dist=True, accumulate_hist=True,
        )

        with torch.no_grad():
            gen_cn = per_graph_center_norms(pos_gen, gen_batch_vec)
            real_cn = per_graph_center_norms(batch.pos, batch.batch)
        self.log(
            "debug/val_gen_center_norm_mean", gen_cn.mean(), batch_size=bs,
            on_epoch=True, sync_dist=True,
        )
        self.log(
            "debug/val_real_center_norm_mean", real_cn.mean(), batch_size=bs,
            on_epoch=True, sync_dist=True,
        )

        return {
            "phi_gen": phi_gen.detach().cpu(),
            "phi_real": phi_real.detach().cpu(),
            "pos_gen": pos_gen.detach().cpu(),
            "gen_atom_types": (
                gen_atom_types.detach().cpu().argmax(dim=-1)
                if gen_atom_types is not None
                else None
            ),
            "pos_real": batch.pos.detach().cpu(),
            "real_atom_types": batch.real_atom_types.detach().cpu(),
            "gen_batch_vec": gen_batch_vec.detach().cpu(),
            "batch_vec": batch.batch.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        _, _, phi_gen, phi_real, gen_batch_vec = self._forward(batch)
        test_loss, _ = self._compute_loss(phi_gen, phi_real, gen_batch_vec, batch.batch)

        bs = self.n_gen_molecules
        self.log(
            "test_loss", test_loss, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True
        )
        return test_loss

    def save_individual_components(self, save_path: str) -> None:
        torch.save(self.generator.state_dict(), f"{save_path}/generator.pth")
        if self._is_feature_extractor_trainable():
            torch.save(
                self.feature_extractor.ept_model.state_dict(),
                f"{save_path}/feature_extractor.pth",
            )

    def load_individual_components(self, folder_path) -> None:
        folder_path = Path(folder_path)
        self.generator.load_state_dict(
            torch.load(folder_path / "generator.pth", map_location=self.device)
        )
        if self.feature_extractor is None:
            return
        fe_path = folder_path / "feature_extractor.pth"
        if fe_path.exists():
            self.feature_extractor.ept_model.load_state_dict(
                torch.load(fe_path, map_location=self.device)
            )
            print("Loaded fine-tuned feature extractor weights.")
