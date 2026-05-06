from pathlib import Path

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR

from ept.ept_loader import load_ept_feature_extractor

from .drift_loss import (TrainingDivergedException,
                         compute_drift_loss)
from .egnn import EGNN
from .geometry import (batch_size_for_logging, center_positions_per_graph,
                       per_graph_center_norms)


class DriftingMoleculeGenerator(LightningModule):
    _SAVE_COMPONENTS = ["generator", "feature_extractor"]
    _LOAD_COMPONENTS = ["generator", "feature_extractor"]

    def __init__(self, generator_cfg=None, drift_cfg=None):
        super().__init__()

        default_generator_cfg = {
            "in_node_nf": 7,
            "hidden_nf": 128,
            "n_layers": 2,
            "num_atom_types": 5,
            "num_bond_types": 5,
            "predict_bond_types": False,
            "pos_clamp": 20.0,
        }
        default_drift_cfg = {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "temperatures": [0.02, 0.05, 0.2],
            "pct_start": 0.1,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
        }

        self.generator_cfg = {**default_generator_cfg, **(generator_cfg or {})}
        self.drift_cfg = {**default_drift_cfg, **(drift_cfg or {})}
        self.save_hyperparameters(
            {"generator_cfg": self.generator_cfg, "drift_cfg": self.drift_cfg}
        )

        self.generator = self._init_generator(self.generator_cfg)
        self.feature_extractor = self._init_feature_extractor()
        self._freeze_feature_extractor()

        self.temperatures = self.drift_cfg["temperatures"]
        self.pos_clamp = self.generator_cfg["pos_clamp"]

    def _init_generator(self, cfg) -> EGNN:
        return EGNN(
            in_node_nf=cfg["in_node_nf"],
            hidden_nf=cfg["hidden_nf"],
            n_layers=cfg["n_layers"],
            num_atom_types=cfg["num_atom_types"],
            num_bond_types=cfg["num_bond_types"],
            predict_bond_types=cfg["predict_bond_types"],
        )

    def _init_feature_extractor(self):
        return load_ept_feature_extractor()

    def _freeze_feature_extractor(self) -> None:
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def sample_prior(self, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:

        # Sample positions
        pos = torch.randn(num_nodes, 3, device=self.device)

        # Sample node features
        # We sample a 7-dimensional feature space, because the
        # node features are composed by a 5-dim one-hot encoding
        # of the atom type + 6 more dimensions for the other features (charge, etc.).
        # We can summarize the 5-dim one-hot encoding in a single dimension,
        # hence, we sample 7 dimensions to cover all the node features
        x = torch.randn(num_nodes, 7, device=self.device)

        return x, pos

    def _forward(self, batch):
        """Shared forward pass: prior → EGNN → center → soft atoms → EPT embeddings."""
        x_prior, pos_prior = self.sample_prior(batch.num_nodes)
        x_gen, _, pos_gen = self.generator(x_prior, pos_prior, batch.dense_edge_index)
        pos_gen = center_positions_per_graph(pos_gen, batch.batch)
        pos_gen = pos_gen.clamp(-self.pos_clamp, self.pos_clamp)
        a_soft_gen = F.gumbel_softmax(x_gen, tau=1.0, hard=False, dim=-1)

        # EPT expects block_id[i] = block index for atom i (each atom is its own block,
        # so block index = atom index), and batch_id[j] = graph index for block j.
        block_id = torch.arange(batch.num_nodes, device=batch.batch.device)
        batch_id = batch.batch

        phi_gen = self.feature_extractor(
            pos=pos_gen,
            a_soft=a_soft_gen,
            block_id=block_id,
            batch_id=batch_id,
            dense_edge_index=batch.dense_edge_index,
        )
        phi_real = self.feature_extractor(
            pos=batch.pos,
            a_soft=batch.a_soft_real,
            block_id=block_id,
            batch_id=batch_id,
            dense_edge_index=batch.dense_edge_index,
        )
        return pos_gen, a_soft_gen, phi_gen, phi_real

    def training_step(self, batch, batch_idx):
        pos_gen, _, phi_gen, phi_real = self._forward(batch)

        if not (torch.isfinite(phi_gen).all() and torch.isfinite(phi_real).all()):
            with torch.no_grad():
                bad_gen_mask = ~torch.isfinite(phi_gen).all(dim=-1)   # [num_graphs]
                bad_real_mask = ~torch.isfinite(phi_real).all(dim=-1)
                bad_mol_mask = bad_gen_mask | bad_real_mask

                n_bad_gen = bad_gen_mask.sum().item()
                n_bad_real = bad_real_mask.sum().item()
                n_total = bad_mol_mask.shape[0]

                atom_mask = bad_mol_mask[batch.batch]
                pos_bad = pos_gen[atom_mask]

                pos_norms_bad = pos_bad.norm(dim=-1)
                max_dist_bad = torch.cdist(pos_bad, pos_bad).max() if pos_bad.shape[0] > 1 else pos_bad.new_tensor(0.0)

                gen_center_norms = per_graph_center_norms(pos_gen, batch.batch)
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
            loss, stats = compute_drift_loss(
                phi_gen, phi_real, temperatures=self.temperatures
            )
        except TrainingDivergedException as e:
            self.print(f"\n[Step {self.global_step}] {e}\nStopping training.")
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        bs = batch_size_for_logging(batch)
        self.log("train_loss", loss, batch_size=bs, on_step=True, on_epoch=True)

        for key, val in stats.items():
            self.log(f"drift_train/{key}", val, batch_size=bs, on_step=True, on_epoch=False)

        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )

        with torch.no_grad():
            pos_norms = pos_gen.norm(dim=-1)
            gen_center_norms = per_graph_center_norms(pos_gen, batch.batch)
            real_center_norms = per_graph_center_norms(batch.pos, batch.batch)
            max_dist = torch.cdist(pos_gen, pos_gen).max()

        self.log("geom/pos_gen_norm_mean", pos_norms.mean(), batch_size=bs)
        self.log("geom/pos_gen_norm_std", pos_norms.std(), batch_size=bs)
        self.log("geom/max_atom_dist", max_dist, batch_size=bs)
        self.log("debug/gen_center_norm_mean", gen_center_norms.mean(), batch_size=bs)
        self.log("debug/gen_center_norm_std", gen_center_norms.std(), batch_size=bs)
        self.log("debug/real_center_norm_mean", real_center_norms.mean(), batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        pos_gen, a_soft_gen, phi_gen, phi_real = self._forward(batch)
        val_loss, stats = compute_drift_loss(
            phi_gen, phi_real, temperatures=self.temperatures
        )

        bs = batch_size_for_logging(batch)
        self.log(
            "val_loss",
            val_loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for key, val in stats.items():
            self.log(
                f"drift_val/{key}",
                val,
                batch_size=bs,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        with torch.no_grad():
            gen_cn = per_graph_center_norms(pos_gen, batch.batch)
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
            "a_soft_gen": a_soft_gen.detach().cpu(),
            "pos_real": batch.pos.detach().cpu(),
            "a_soft_real": batch.a_soft_real.detach().cpu(),
            "batch_vec": batch.batch.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        _, _, phi_gen, phi_real = self._forward(batch)
        test_loss, _ = compute_drift_loss(
            phi_gen, phi_real, self.temperatures
        )

        bs = batch_size_for_logging(batch)
        self.log(
            "test_loss",
            test_loss,
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return test_loss

    def _is_feature_extractor_trainable(self) -> bool:
        return any(p.requires_grad for p in self.feature_extractor.parameters())

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
        fe_path = folder_path / "feature_extractor.pth"
        if fe_path.exists():
            self.feature_extractor.ept_model.load_state_dict(
                torch.load(fe_path, map_location=self.device)
            )
            print("Loaded fine-tuned feature extractor weights.")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.drift_cfg["lr"],
            weight_decay=self.drift_cfg["weight_decay"],
            eps=1e-8,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.drift_cfg["lr"],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.drift_cfg["pct_start"],
            anneal_strategy="cos",
            div_factor=self.drift_cfg["div_factor"],
            final_div_factor=self.drift_cfg["final_div_factor"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
