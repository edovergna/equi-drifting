import torch
import torch.nn.functional as F

from ..egnn import EGNN
from ..geometry import batch_size_for_logging, center_positions_per_graph, per_graph_center_norms
from ..losses import TrainingDivergedException
from ..losses.geometry import probs_to_sphere, sphere_to_probs
from ..losses.riemannian import compute_molecule_based_drift_loss
from .base import BaseDriftingMoleculeGenerator


class RiemmanianGenerator(BaseDriftingMoleculeGenerator):
    _SAVE_COMPONENTS = ["generator"]
    _LOAD_COMPONENTS = ["generator"]

    def __init__(self, generator_cfg=None, drift_cfg=None):
        default_generator_cfg = {
            "hidden_nf": 128,
            "n_layers": 2,
            "num_atom_types": 5,
            "num_bond_types": 5,
            "coordinate_clamp_range": 5.0,
            "predict_bond_types": False,
            "pos_clamp": 10.0,
            "prior_pos_clamp": 5.0,
        }
        default_drift_cfg = {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "temperatures": [0.02, 0.05, 0.2],
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
        self.pos_clamp = self.generator_cfg["pos_clamp"]

    def _init_generator(self, cfg) -> EGNN:
        return EGNN(
            hidden_nf=cfg["hidden_nf"],
            n_layers=cfg["n_layers"],
            num_atom_types=cfg["num_atom_types"],
            num_bond_types=cfg["num_bond_types"],
            predict_bond_types=cfg["predict_bond_types"],
        )

    def _is_feature_extractor_trainable(self) -> bool:
        return False

    def _forward(self, batch):
        """Forward pass: prior → EGNN → center → soft atoms → spherical projection."""
        n_molecules = batch_size_for_logging(batch)
        x_prior, pos_prior, gen_batch_vec, gen_dense_edge_index = (
            self._sample_prior_batch(n_molecules)
        )
        x_logits, _, pos_gen = self.generator(x_prior, pos_prior, gen_dense_edge_index)
        pos_gen = center_positions_per_graph(pos_gen, gen_batch_vec)
        pos_gen = pos_gen.clamp(-self.pos_clamp, self.pos_clamp)

        x_prob = F.softmax(x_logits, dim=-1)
        x_sphere = probs_to_sphere(x_prob, self.eps)

        return pos_gen, x_sphere, gen_batch_vec

    def training_step(self, batch, batch_idx):
        pos_gen, x_gen, gen_batch_vec = self._forward(batch)
        pos_real, x_real = batch.pos, batch.real_atom_types

        try:
            loss, stats = compute_molecule_based_drift_loss(
                pos_gen, x_gen, pos_real, x_real,
                gen_index=gen_batch_vec, real_index=batch.batch, eps=self.eps,
            )
        except TrainingDivergedException as e:
            self.print(f"\n[Step {self.global_step}] {e}\nStopping training.")
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        bs = batch_size_for_logging(batch)
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
        pos_gen, x_gen, gen_batch_vec = self._forward(batch)
        pos_real, x_real = batch.pos, batch.real_atom_types
        val_loss, stats = compute_molecule_based_drift_loss(
            pos_gen, x_gen, pos_real, x_real,
            gen_index=gen_batch_vec, real_index=batch.batch, eps=self.eps,
        )

        bs = batch_size_for_logging(batch)
        if self.trainer is not None and not self.trainer.sanity_checking:
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

        with torch.no_grad():
            x_prob = sphere_to_probs(x_gen, self.eps)

        return {
            "pos_gen": pos_gen.detach().cpu(),
            "gen_atom_types": x_prob.detach().cpu().argmax(dim=-1),
            "pos_real": batch.pos.detach().cpu(),
            "real_atom_types": batch.real_atom_types.detach().cpu(),
            "gen_batch_vec": gen_batch_vec.detach().cpu(),
            "batch_vec": batch.batch.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        pos_gen, x_gen, gen_batch_vec = self._forward(batch)
        pos_real, x_real = batch.pos, batch.real_atom_types

        test_loss, _ = compute_molecule_based_drift_loss(
            pos_gen, x_gen, pos_real, x_real,
            gen_index=gen_batch_vec, real_index=batch.batch, eps=self.eps,
        )

        bs = batch_size_for_logging(batch)
        self.log(
            "test_loss", test_loss, batch_size=bs, on_step=False, on_epoch=True, sync_dist=True
        )
        return test_loss
