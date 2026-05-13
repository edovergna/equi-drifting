from pathlib import Path

import torch
import torch.nn.functional as F

from ept.ept_loader import load_ept_feature_extractor
from torch_geometric.nn import global_mean_pool

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
            # KL divergence weight for atom-type distribution matching.
            # Prevents the generator from collapsing to a single atom type.
            "atom_type_loss_weight": 0.0,
            # Geometry loss: penalise atom overlap and isolated atoms.
            # Overlap term: pairs closer than 0.7 Å.
            # Isolation term: atoms with no neighbour within 2.5 Å.
            "geom_loss_weight": 0.0,
            # Soft valence loss: penalise wrong bond counts per atom type.
            # Directly targets validity. Anneals naturally alongside geom_loss.
            "valence_loss_weight": 0.0,
            # How to build the per-molecule fingerprint fed to the drift loss:
            #   'graph_repr' — EPT's variance-preserving sum of atom features [G, 512] (default/legacy)
            #   'moments'    — cat(mean(H_atoms), std(H_atoms)) per graph [G, 1024];
            #                  captures atom-type distribution + chemical diversity;
            #                  implicitly encodes valence/geometry without auxiliary losses
            "phi_mode": "graph_repr",
            # How to append the equivariant output to the fingerprint:
            #   'norm'   — append ||phi_equiv|| (1 scalar, SE(3)-invariant) [default]
            #   'vector' — append phi_equiv as 3 raw components (equivariant, orientation-dependent)
            #   'off'    — do not append anything (ablation)
            "equiv_phi_mode": "norm",
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
        self.atom_type_loss_weight = self.drift_cfg.get("atom_type_loss_weight", 0.0)
        self.geom_loss_weight = self.drift_cfg.get("geom_loss_weight", 0.0)
        self.valence_loss_weight = self.drift_cfg.get("valence_loss_weight", 0.0)
        self.phi_mode = self.drift_cfg.get("phi_mode", "graph_repr")
        self.equiv_phi_mode = self.drift_cfg.get("equiv_phi_mode", "norm")
        self.pos_clamp = self.generator_cfg["pos_clamp"]
        self.pos_clamp_type = self.generator_cfg["pos_clamp_type"]
        self.c_pos_clamp = self.generator_cfg["c_pos_clamp"]
        self.p_pos_clamp = self.generator_cfg["p_pos_clamp"]
        self.norm_pos_clamp = self.generator_cfg["norm_pos_clamp"]
        self.infer_types_from_pos = self.generator_cfg.get(
            "infer_types_from_pos", False
        )
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
            if not self.trainer.sanity_checking and self.trainer.training:
                rescale.register_hook(
                    lambda g: setattr(self, "_norm_rescale_grad", g.abs().mean().item())
                )
            pos_gen = pos_gen * rescale
        else:  # geom
            norm = pos_gen.norm(dim=-1)
            rescale = 1 / (1 + (norm / self.c_pos_clamp) ** self.p_pos_clamp)
            if not self.trainer.sanity_checking and self.trainer.training:
                rescale.register_hook(
                    lambda g: setattr(self, "_norm_rescale_grad", g.abs().mean().item())
                )
            pos_gen = pos_gen * rescale.unsqueeze(-1)

        if self.infer_types_from_pos:
            with torch.no_grad():
                gen_atom_types = infer_types_from_pos_batch(
                    pos_gen,
                    gen_batch_vec,
                    self.device,
                    self.generator_cfg["num_atom_types"],
                    self.infer_method,
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
            # EPT expects block_id[i] = block index for atom i (each atom is its own block,
            # so block index = atom index), and batch_id[j] = graph index for block j.
            # feature_extractor returns (graph_repr [G, D], phi_equiv [G, 3], H_atoms [N, D])
            phi_gen_base, phi_gen_equiv, H_gen = self.feature_extractor(
                pos=pos_gen,
                atom_types=gen_atom_types,
                block_id=torch.arange(pos_gen.shape[0], device=self.device),
                batch_id=gen_batch_vec,
                dense_edge_index=gen_dense_edge_index,
            )
            phi_real_base, phi_real_equiv, H_real = self.feature_extractor(
                pos=batch.pos,
                atom_types=batch.real_atom_types,
                block_id=torch.arange(batch.num_nodes, device=batch.batch.device),
                batch_id=batch.batch,
                dense_edge_index=batch.dense_edge_index,
            )

            # Build per-molecule fingerprint according to phi_mode.
            if self.phi_mode == "moments":
                # mean(H) + std(H) per graph: [G, 2D=1024].
                # Captures atom-type distribution (mean) and chemical diversity (std).
                # EPT already encoded valence & geometry into H_atoms during pretraining,
                # so this implicitly enforces chemistry without auxiliary losses.
                mean_gen = global_mean_pool(H_gen, gen_batch_vec)  # [G_gen,  D]
                mean_sq_gen = global_mean_pool(H_gen.pow(2), gen_batch_vec)
                std_gen = (mean_sq_gen - mean_gen.pow(2)).clamp(min=0).sqrt()
                phi_gen = torch.cat([mean_gen, std_gen], dim=-1)  # [G_gen,  2D]

                mean_real = global_mean_pool(H_real, batch.batch)  # [G_real, D]
                mean_sq_real = global_mean_pool(H_real.pow(2), batch.batch)
                std_real = (mean_sq_real - mean_real.pow(2)).clamp(min=0).sqrt()
                phi_real = torch.cat([mean_real, std_real], dim=-1)  # [G_real, 2D]
            else:
                # 'graph_repr': legacy variance-preserving sum [G, D=512]
                phi_gen = phi_gen_base
                phi_real = phi_real_base

            # Append equivariant information to the fingerprint according to equiv_phi_mode.
            norm_scale = phi_real_equiv.norm(dim=-1).mean().clamp(min=1e-3).detach()
            if self.equiv_phi_mode == "vector":
                phi_gen = torch.cat([phi_gen, phi_gen_equiv / norm_scale], dim=-1)
                phi_real = torch.cat([phi_real, phi_real_equiv / norm_scale], dim=-1)
            elif self.equiv_phi_mode == "norm":
                norm_feat_gen = phi_gen_equiv.norm(dim=-1, keepdim=True) / norm_scale
                norm_feat_real = phi_real_equiv.norm(dim=-1, keepdim=True) / norm_scale
                phi_gen = torch.cat([phi_gen, norm_feat_gen], dim=-1)
                phi_real = torch.cat([phi_real, norm_feat_real], dim=-1)
            # else 'off': leave phi unchanged (ablation)

            # Normalise to unit sphere so pairwise distances in the drift loss are
            # always O(1) regardless of EPT embedding scale.
            phi_gen = F.normalize(phi_gen, dim=-1)
            phi_real = F.normalize(phi_real, dim=-1)

        return pos_gen, gen_atom_types, phi_gen, phi_real, gen_batch_vec

    def _compute_geom_loss(
        self,
        pos: torch.Tensor,
        batch_vec: torch.Tensor,
        d_min: float = 0.7,
        d_bond_max: float = 2.5,
    ) -> torch.Tensor:
        """Differentiable geometry loss on raw 3-D positions.

        overlap_loss   — penalise atom pairs closer than d_min (Å)
        isolation_loss — penalise atoms whose nearest neighbour is beyond d_bond_max (Å)

        Both terms use a squared-hinge (ReLU²) penalty for smooth gradients.
        """
        n_graphs = int(batch_vec.max().item()) + 1
        overlap_total = pos.new_zeros(())
        isolation_total = pos.new_zeros(())

        for g in range(n_graphs):
            p = pos[batch_vec == g]  # [N_g, 3]
            n = p.shape[0]
            if n < 2:
                continue
            # Safe pairwise distances: avoids NaN gradient of torch.cdist at d=0.
            # eps=1e-2 bounds the position gradient to ≤1/sqrt(eps)=10 even when atoms coincide.
            diff = p.unsqueeze(1) - p.unsqueeze(0)  # [N_g, N_g, 3]
            dists = (diff.pow(2).sum(dim=-1) + 1e-2).sqrt()  # [N_g, N_g]
            eye = torch.eye(n, device=p.device, dtype=torch.bool)
            pair_dists = dists[~eye]  # [N_g*(N_g-1)]

            overlap_total = overlap_total + F.relu(d_min - pair_dists).pow(2).mean()

            dists_no_self = dists.masked_fill(eye, float("inf"))
            nn_dist = dists_no_self.min(dim=-1).values  # [N_g]
            isolation_total = (
                isolation_total + F.relu(nn_dist - d_bond_max).pow(2).mean()
            )

        return (overlap_total + isolation_total) / n_graphs

    # Covalent radii (Å) per QM9 atom type index: H=0, C=1, N=2, O=3, F=4
    _COV_RADII = torch.tensor([0.31, 0.76, 0.71, 0.66, 0.57])
    # Stable (target) valence per atom type
    _STABLE_VALENCE = torch.tensor([1.0, 4.0, 3.0, 2.0, 1.0])

    def _compute_valence_loss(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        batch_vec: torch.Tensor,
        bond_factor: float = 1.3,
        temperature: float = 0.3,
    ) -> torch.Tensor:
        """Differentiable soft-valence loss.

        For each atom, counts expected bonds via a sigmoid over pairwise distances
        using atom-type-aware covalent radii, then penalises deviation from the
        stable valence ({H:1, C:4, N:3, O:2, F:1}).

        pos:        [N, 3]
        atom_types: [N, 5] one-hot (Gumbel straight-through)
        batch_vec:  [N]
        """
        cov_r = self._COV_RADII.to(pos.device)  # [5]
        stable_v = self._STABLE_VALENCE.to(pos.device)  # [5]

        # Per-atom radius and target valence via soft lookup (one-hot so exact).
        # Detach atom_types so the valence loss only trains positions, not atom types.
        # Atom types are trained separately by atom_type_loss to avoid collapse.
        at = atom_types.detach()
        atom_radii = (at * cov_r).sum(dim=-1)  # [N]
        atom_valence = (at * stable_v).sum(dim=-1)  # [N]

        n_graphs = int(batch_vec.max().item()) + 1
        valence_loss = pos.new_zeros(())

        for g in range(n_graphs):
            mask = batch_vec == g
            p = pos[mask]  # [N_g, 3]
            r = atom_radii[mask]  # [N_g]
            v = atom_valence[mask]  # [N_g]
            n = p.shape[0]
            if n < 2:
                continue

            # Safe pairwise distances: avoids NaN gradient of torch.cdist at d=0.
            # eps=1e-2 bounds the position gradient to ≤1/sqrt(eps)=10 even when atoms coincide.
            diff = p.unsqueeze(1) - p.unsqueeze(0)  # [N_g, N_g, 3]
            dists = (diff.pow(2).sum(dim=-1) + 1e-2).sqrt()  # [N_g, N_g]
            # Bond threshold matrix: d < bond_factor * (r_i + r_j)
            bond_thresh = bond_factor * (r.unsqueeze(1) + r.unsqueeze(0))  # [N_g, N_g]
            # Soft bond count per atom: sigmoid so gradient flows through distances
            eye = torch.eye(n, device=p.device, dtype=torch.bool)
            soft_bonds = torch.sigmoid((bond_thresh - dists) / temperature)
            soft_bonds = soft_bonds.masked_fill(eye, 0.0)  # exclude self
            soft_valence = soft_bonds.sum(dim=-1)  # [N_g]

            valence_loss = valence_loss + F.mse_loss(soft_valence, v)

        return valence_loss / n_graphs

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
        pos_gen, gen_atom_types, phi_gen, phi_real, gen_batch_vec = self._forward(batch)

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
                n_bad_gen = bad_gen_mask.sum().item()
                n_bad_real = (~torch.isfinite(phi_real).all(dim=-1)).sum().item()
                self.print(
                    f"\n[Step {self.global_step}] Non-finite embeddings — "
                    f"{n_bad_gen}/{bad_gen_mask.shape[0]} gen, "
                    f"{n_bad_real}/{phi_real.shape[0]} real. Skipping step."
                )
                self.log("train/nan_skip", 1.0, on_step=True, on_epoch=False)
            return None

        try:
            loss, stats = self._compute_loss(
                phi_gen, phi_real, gen_batch_vec, batch.batch
            )
        except TrainingDivergedException as e:
            self.print(f"\n[Step {self.global_step}] {e}\nStopping training.")
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        bs = self.n_gen_molecules

        if self.atom_type_loss_weight > 0.0:
            gen_type_dist = gen_atom_types.float().mean(dim=0)  # [5], STE grad
            real_type_dist = batch.real_atom_types.float().mean(dim=0).detach()  # [5]
            atom_type_loss = F.mse_loss(gen_type_dist, real_type_dist)
            loss = loss + self.atom_type_loss_weight * atom_type_loss
            self.log(
                "train/atom_type_loss",
                atom_type_loss,
                batch_size=bs,
                on_step=True,
                on_epoch=False,
            )

        if self.geom_loss_weight > 0.0:
            geom_loss = self._compute_geom_loss(pos_gen, gen_batch_vec)
            loss = loss + self.geom_loss_weight * geom_loss
            self.log(
                "train/geom_loss",
                geom_loss,
                batch_size=bs,
                on_step=True,
                on_epoch=False,
            )

        if self.valence_loss_weight > 0.0:
            valence_loss = self._compute_valence_loss(
                pos_gen, gen_atom_types, gen_batch_vec
            )
            loss = loss + self.valence_loss_weight * valence_loss
            self.log(
                "train/valence_loss",
                valence_loss,
                batch_size=bs,
                on_step=True,
                on_epoch=False,
            )

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
        val_loss, stats = self._compute_loss(
            phi_gen, phi_real, gen_batch_vec, batch.batch
        )

        if self.atom_type_loss_weight > 0.0:
            gen_type_dist = gen_atom_types.float().mean(dim=0)
            real_type_dist = batch.real_atom_types.float().mean(dim=0).detach()
            atom_type_loss = F.mse_loss(gen_type_dist, real_type_dist)
            val_loss = val_loss + self.atom_type_loss_weight * atom_type_loss
            self.log(
                "val/atom_type_loss",
                atom_type_loss,
                batch_size=self.n_gen_molecules,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if self.geom_loss_weight > 0.0:
            geom_loss = self._compute_geom_loss(pos_gen, gen_batch_vec)
            val_loss = val_loss + self.geom_loss_weight * geom_loss
            self.log(
                "val/geom_loss",
                geom_loss,
                batch_size=self.n_gen_molecules,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if self.valence_loss_weight > 0.0:
            valence_loss = self._compute_valence_loss(
                pos_gen, gen_atom_types, gen_batch_vec
            )
            val_loss = val_loss + self.valence_loss_weight * valence_loss
            self.log(
                "val/valence_loss",
                valence_loss,
                batch_size=self.n_gen_molecules,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        bs = self.n_gen_molecules
        self.log(
            "val_loss",
            val_loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self._log_drift_stats(
            stats,
            "drift_val",
            bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            accumulate_hist=True,
        )

        with torch.no_grad():
            gen_cn = per_graph_center_norms(pos_gen, gen_batch_vec)
            real_cn = per_graph_center_norms(batch.pos, batch.batch)
        self.log(
            "debug/val_gen_center_norm_mean",
            gen_cn.mean(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "debug/val_real_center_norm_mean",
            real_cn.mean(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
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

    def on_validation_epoch_end(self):
        hist_stats = getattr(self, "_val_hist_stats", {})
        if (
            hist_stats
            and hasattr(self, "logger")
            and hasattr(self.logger, "experiment")
        ):
            try:
                self.logger.experiment.log(hist_stats, commit=False)
            except Exception:
                pass
        self._val_hist_stats = {}

    def test_step(self, batch, batch_idx):
        _, _, phi_gen, phi_real, gen_batch_vec = self._forward(batch)
        test_loss, _ = self._compute_loss(phi_gen, phi_real, gen_batch_vec, batch.batch)

        bs = self.n_gen_molecules
        self.log(
            "test_loss",
            test_loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
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
