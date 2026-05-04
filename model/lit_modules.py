import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR


from ept import EPTFeatureExtractor
from .egnn import EGNN


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""


class DriftingMoleculeGenerator(LightningModule):
    def __init__(self, generator_cfg=None, drift_cfg=None):
        super().__init__()

        default_generator_cfg = {
            "in_node_nf": 7,
            "hidden_nf": 128,
            "n_layers": 2,
            "num_atom_types": 5,
            "num_bond_types": 5,
            "compute_heads": False,
            "predict_bond_types": False,
        }
        default_drift_cfg = {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "temperatures": [0.02, 0.05, 0.2],
            "pct_start": 0.1,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
        }

        generator_cfg = generator_cfg or {}
        drift_cfg = drift_cfg or {}

        self.generator_cfg = {**default_generator_cfg, **generator_cfg}
        self.drift_cfg = {**default_drift_cfg, **drift_cfg}
        self.save_hyperparameters(
            {"generator_cfg": self.generator_cfg, "drift_cfg": self.drift_cfg}
        )

        # Generator: Your EGNN or GNN architecture
        self.generator = self._init_generator(self.generator_cfg)

        # Feature Space: The paper suggests drifting in a feature space
        # For now, this can be a simple linear layer or a small GNN encoder
        self.feature_extractor = self._init_feature_extractor()
        self._freeze_feature_extractor()

        # Hyperparameters for Drifting Field V
        self.temperatures = self.drift_cfg["temperatures"]

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
        # Check if the checkpoint has been downloaded and download it if not
        root_path = Path(__file__).parents[1]

        # Hard coding path for now, unsure if making it configurable is worth it.
        ckpt_path = Path("hybrid_noaf/epoch49_step215752.ckpt")

        full_ckpt_path = root_path / ckpt_path

        if not full_ckpt_path.exists():
            print(f"Checkpoint not found at {full_ckpt_path}. Downloading EPT from Google Drive...")
            result = subprocess.run(
                [
                    "gdown",
                    "--folder",
                    "https://drive.google.com/drive/folders/1tBqGwC_jcTdq3QArFZox_auSCzxDjA0P",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"EPT checkpoint download failed (exit code {result.returncode}).\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}\n"
                    "Ensure gdown is installed: pip install gdown"
                )
            if not full_ckpt_path.exists():
                raise RuntimeError(
                    f"Download appeared to succeed but checkpoint not found at {full_ckpt_path}.\n"
                    f"gdown output: {result.stdout}\n"
                    "Check that the folder structure matches the expected path."
                )
            print("Download complete.")

        # Add the "ept" directory to sys.path so that torch.load finds the EPT modules
        ept_path = str(root_path / "ept")
        if ept_path not in sys.path:
            sys.path.append(ept_path)

        return EPTFeatureExtractor(
            str(full_ckpt_path), torch.device("cpu")
        )  # start on cpu; Lightning will move module later

    def _freeze_feature_extractor(self):
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    def on_before_optimizer_step(self, optimizer):
        # Log pre-clip gradient norms so we can see the raw signal before Lightning clips it.
        grads = [p.grad for p in self.generator.parameters() if p.grad is not None]
        if grads:
            total_norm = torch.stack([g.detach().norm(2) for g in grads]).norm(2)
            max_abs = torch.stack([g.detach().abs().max() for g in grads]).max()
            self.log("grad/total_norm", total_norm, on_step=True, on_epoch=False)
            self.log("grad/max_abs", max_abs, on_step=True, on_epoch=False)

    def on_fit_start(self):
        self.feature_extractor.to(self.device)  # here
        self.feature_extractor.eval()

    def on_validation_start(self):
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

    def on_test_start(self):
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

    def compute_v(self, x, y_pos, y_neg, tau):  # chat generated
        """
        Implements Algorithm 2: Computing the drifting field V.
        x: [B, D] Generated graph-level features
        y_pos: [B_pos, D] Real data graph-level features
        y_neg: [B_neg, D] Generated graph-level features (usually identical to x)
        tau: Temperature scaling factor
        """
        N = x.size(0)

        # 1. Compute Pairwise Distances
        dist_pos = torch.cdist(x, y_pos)  # [N, B_pos]
        dist_neg = torch.cdist(x, y_neg)  # [N, B_neg]

        # Ignore self in repulsion.
        # If y_neg is x, the diagonal distance is 0. We artificially inflate it
        # so a sample doesn't infinitely repulse itself.
        if x is y_neg or (N == y_neg.size(0) and torch.allclose(x, y_neg)):
            dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

        # 2. Compute Logits
        logit_pos = -dist_pos / tau  # might need to scale this when getting NaN
        logit_neg = -dist_neg / tau

        # Concatenate for joint normalization along the sample axis
        logit = torch.cat([logit_pos, logit_neg], dim=1)  # [N, B_pos + B_neg]

        # 3. Normalization along both dimensions (Anti-symmetry)
        # Softmax over the sample axis (columns)
        A_row = F.softmax(logit, dim=-1)
        # Softmax over the x axis (rows)
        A_col = F.softmax(logit, dim=-2)

        # Geometric mean to balance the normalizations
        A = torch.sqrt(A_row * A_col)

        # Split back into positive and negative attention matrices
        A_pos, A_neg = torch.split(A, [y_pos.size(0), y_neg.size(0)], dim=1)

        # 4. Compute Weighted Drift
        # Normalize weights so they sum to 1.0 for each row to compute a valid expectation
        W_pos = A_pos / (A_pos.sum(dim=1, keepdim=True) + 1e-8)
        W_neg = A_neg / (A_neg.sum(dim=1, keepdim=True) + 1e-8)

        # Multiply weights by the actual feature vectors
        drift_pos = W_pos @ y_pos  # [N, D]
        drift_neg = W_neg @ y_neg  # [N, D]

        # Final vector field
        v_field = drift_pos - drift_neg

        return v_field

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

    def _per_graph_center_norms(
        self, pos: torch.Tensor, batch_vec: torch.Tensor
    ) -> torch.Tensor:
        """Returns a [G] tensor of per-graph center L2 norms. Should be ~0 if centered."""
        num_graphs = int(batch_vec.max().item()) + 1
        sums = torch.zeros(num_graphs, 3, device=pos.device, dtype=pos.dtype)
        sums.index_add_(0, batch_vec, pos)
        counts = torch.bincount(batch_vec, minlength=num_graphs).clamp_min(1).to(pos.dtype)
        centers = sums / counts.unsqueeze(-1)
        return centers.norm(dim=-1)

    def _batch_size_for_logging(self, batch) -> int:
        if hasattr(batch, "num_graphs") and batch.num_graphs is not None:
            return int(batch.num_graphs)

        if hasattr(batch, "batch") and batch.batch is not None:
            return int(batch.batch.max().item()) + 1

        return 1

    def _center_positions_per_graph(
        self, pos: torch.Tensor, batch_vec: torch.Tensor
    ) -> torch.Tensor:
        """Zero-center coordinates independently for each graph in a batch."""
        if batch_vec is None or batch_vec.numel() == 0:
            return pos - pos.mean(dim=0, keepdim=True)

        num_graphs = int(batch_vec.max().item()) + 1
        sums = torch.zeros(num_graphs, pos.size(-1), device=pos.device, dtype=pos.dtype)
        sums.index_add_(0, batch_vec, pos)

        counts = torch.bincount(batch_vec, minlength=num_graphs).to(pos.device)
        counts = counts.clamp_min(1).unsqueeze(-1).to(pos.dtype)

        centers = sums / counts
        return pos - centers[batch_vec]

    def training_step(self, batch, batch_idx):
        x_prior, pos_prior = self.sample_prior(batch.num_nodes)

        # 1. Use dense_edge_index (No Data Leakage)
        x_gen, edge_bond_logits, pos_gen = self.generator(
            x_prior,
            pos_prior,
            batch.dense_edge_index,
        )

        # 2. Zero-center coordinates per graph (Prevent Spatial Drift)
        pos_gen = self._center_positions_per_graph(pos_gen, batch.batch)

        # Log center norms to verify centering is working for both real and generated molecules
        with torch.no_grad():
            gen_center_norms = self._per_graph_center_norms(pos_gen, batch.batch)
            real_center_norms = self._per_graph_center_norms(batch.pos, batch.batch)

        a_soft_gen = F.gumbel_softmax(x_gen, tau=1.0, hard=False, dim=-1)

        phi_gen = self.feature_extractor(
            pos=pos_gen,
            a_soft=a_soft_gen,
            batch_vec=batch.batch,
            dense_edge_index=batch.dense_edge_index,
        )
        phi_real = self.feature_extractor(
            pos=batch.pos,
            a_soft=batch.a_soft_real,
            batch_vec=batch.batch,
            dense_edge_index=batch.dense_edge_index,
        )

        # Early NaN/Inf check before computing loss — gives a clearer error than a cryptic NaN
        if not (torch.isfinite(phi_gen).all() and torch.isfinite(phi_real).all()):
            bad_gen = (~torch.isfinite(phi_gen)).sum().item()
            bad_real = (~torch.isfinite(phi_real)).sum().item()
            self.print(
                f"\n[Step {self.global_step}] Non-finite embeddings detected — "
                f"phi_gen: {bad_gen} bad values, phi_real: {bad_real} bad values. "
                "Stopping training."
            )
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 3. Use the robust normalized drifting loss
        try:
            loss, stats = self.compute_normalized_drift_loss(phi_gen, phi_real)
        except TrainingDivergedException as e:
            self.print(f"\n[Step {self.global_step}] {e}\nStopping training.")
            self.trainer.should_stop = True
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        batch_size = self._batch_size_for_logging(batch)
        self.log("train_loss", loss, batch_size=batch_size, on_step=True, on_epoch=True)

        # Drift internals: scale, per-temperature lambda / drift magnitude / attention entropy
        for key, val in stats.items():
            self.log(f"drift/{key}", val, batch_size=batch_size, on_step=True, on_epoch=False)

        # Learning rate (OneCycleLR updates every step — make it visible)
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=True, on_epoch=False)

        # Geometry: coordinate magnitudes and pairwise extent of generated molecules
        with torch.no_grad():
            pos_norms = pos_gen.norm(dim=-1)
            max_dist = torch.cdist(pos_gen, pos_gen).max()
        self.log("geom/pos_gen_norm_mean", pos_norms.mean(), batch_size=batch_size)
        self.log("geom/pos_gen_norm_std", pos_norms.std(), batch_size=batch_size)
        self.log("geom/max_atom_dist", max_dist, batch_size=batch_size)

        # Center norm diagnostics (should be ~0 if centering is correct)
        self.log("debug/gen_center_norm_mean", gen_center_norms.mean(), batch_size=batch_size)
        self.log("debug/gen_center_norm_std", gen_center_norms.std(), batch_size=batch_size)
        self.log("debug/real_center_norm_mean", real_center_norms.mean(), batch_size=batch_size)

        return loss

    def compute_normalized_drift_loss(
        self, phi_gen: torch.Tensor, phi_real: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Returns (loss, stats) where stats is a flat dict of float diagnostics."""
        # Run pairwise-distance math in float32 for AMP stability.
        phi_gen = torch.nan_to_num(phi_gen.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        phi_real = torch.nan_to_num(phi_real.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        D = phi_gen.shape[-1]

        dist_pos = torch.cdist(phi_gen, phi_real)
        dist_neg = torch.cdist(phi_gen, phi_gen)
        dist_neg.fill_diagonal_(1e6)

        all_dists = torch.cat([dist_pos.flatten(), dist_neg.flatten()])
        valid_mask = torch.isfinite(all_dists) & (all_dists < 1e5)
        valid_dists = all_dists[valid_mask]

        if valid_dists.numel() == 0:
            S = torch.tensor(1.0, device=phi_gen.device, dtype=phi_gen.dtype)
        else:
            S = (valid_dists.mean() / (D**0.5)).detach()
        S = torch.clamp(S, min=1e-5, max=1e3)

        phi_gen_norm = phi_gen / S
        phi_real_norm = phi_real / S

        norm_dist_pos = dist_pos / S
        norm_dist_neg = dist_neg / S

        aggregated_v_norm = torch.zeros_like(phi_gen_norm)

        stats: dict[str, float] = {}
        with torch.no_grad():
            stats["scale_S"] = S.item()
            gen_norms = phi_gen_norm.norm(dim=-1)
            real_norms = phi_real_norm.norm(dim=-1)
            stats["phi_gen_norm_mean"] = gen_norms.mean().item()
            stats["phi_gen_norm_std"] = gen_norms.std().item()
            stats["phi_real_norm_mean"] = real_norms.mean().item()

        for tau in self.temperatures:
            tau_key = str(tau).replace(".", "_")
            tau_tilde = tau * (D**0.5)

            logit_pos = -norm_dist_pos / tau_tilde
            logit_neg = -norm_dist_neg / tau_tilde
            logit = torch.clamp(torch.cat([logit_pos, logit_neg], dim=1), min=-100.0, max=50.0)

            A_row = F.softmax(logit, dim=-1)
            A_col = F.softmax(logit, dim=-2)
            A = torch.sqrt(torch.clamp(A_row * A_col, min=1e-30))

            A_pos, A_neg = torch.split(A, [phi_real.size(0), phi_gen.size(0)], dim=1)

            denom_pos = A_pos.sum(dim=1, keepdim=True).clamp_min(1e-5)
            denom_neg = A_neg.sum(dim=1, keepdim=True).clamp_min(1e-5)
            W_pos = A_pos / denom_pos
            W_neg = A_neg / denom_neg

            drift_pos = W_pos @ phi_real_norm
            drift_neg = W_neg @ phi_gen_norm
            V_tau = drift_pos - drift_neg

            v_sq_norm = (V_tau**2).sum(dim=-1)
            lambda_tau = torch.sqrt(
                torch.clamp(v_sq_norm.mean() / D, min=1e-10)
            ).detach()
            lambda_tau = torch.clamp(lambda_tau, min=1e-5, max=1e3)

            V_tau_norm = V_tau / lambda_tau
            aggregated_v_norm += V_tau_norm

            with torch.no_grad():
                # Entropy of row-softmax attention: 0 = all mass on one neighbour, log(N) = uniform
                row_entropy = -(A_row * (A_row + 1e-30).log()).sum(dim=-1).mean()
                stats[f"attn_entropy_{tau_key}"] = row_entropy.item()
                stats[f"lambda_{tau_key}"] = lambda_tau.item()
                stats[f"v_norm_{tau_key}"] = V_tau_norm.norm(dim=-1).mean().item()

        target = (phi_gen_norm + aggregated_v_norm).detach()
        loss = F.mse_loss(phi_gen_norm, target)

        if not torch.isfinite(loss):
            raise TrainingDivergedException(
                f"Non-finite loss ({loss.item()!r}) after drift computation. "
                f"phi_gen range: [{phi_gen.min().item():.3g}, {phi_gen.max().item():.3g}], "
                f"phi_real range: [{phi_real.min().item():.3g}, {phi_real.max().item():.3g}], "
                f"S={S.item():.3g}"
            )

        return loss, stats

    def validation_step(self, batch, batch_idx):
        x_prior, pos_prior = self.sample_prior(batch.num_nodes)

        # 1. Use dense_edge_index to prevent data leakage
        x_gen, edge_bond_logits, pos_gen = self.generator(
            x_prior,
            pos_prior,
            batch.dense_edge_index,
        )

        # 2. Zero-center coordinates per graph to prevent cross-graph leakage
        pos_gen = self._center_positions_per_graph(pos_gen, batch.batch)

        with torch.no_grad():
            gen_center_norms = self._per_graph_center_norms(pos_gen, batch.batch)
            real_center_norms = self._per_graph_center_norms(batch.pos, batch.batch)

        a_soft_gen = F.gumbel_softmax(x_gen, tau=1.0, hard=False, dim=-1)

        phi_gen = self.feature_extractor(
            pos=pos_gen,
            a_soft=a_soft_gen,
            batch_vec=batch.batch,
            dense_edge_index=batch.dense_edge_index,
        )

        phi_real = self.feature_extractor(
            pos=batch.pos,
            a_soft=batch.a_soft_real,
            batch_vec=batch.batch,
            dense_edge_index=batch.dense_edge_index,
        )

        # 3. Use the exact same normalized objective
        val_loss, stats = self.compute_normalized_drift_loss(phi_gen, phi_real)

        batch_size = self._batch_size_for_logging(batch)

        for key, val in stats.items():
            self.log(f"val/{key}", val, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)

        self.log("debug/val_gen_center_norm_mean", gen_center_norms.mean(), batch_size=batch_size, on_epoch=True, sync_dist=True)
        self.log("debug/val_real_center_norm_mean", real_center_norms.mean(), batch_size=batch_size, on_epoch=True, sync_dist=True)
        self.log(
            "val_loss",
            val_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return val_loss

    def test_step(self, batch, batch_idx):
        x_prior, pos_prior = self.sample_prior(batch.num_nodes)

        x_gen, edge_bond_logits, pos_gen = self.generator(
            x_prior,
            pos_prior,
            batch.dense_edge_index,
        )

        pos_gen = self._center_positions_per_graph(pos_gen, batch.batch)

        a_soft_gen = F.gumbel_softmax(x_gen, tau=1.0, hard=False, dim=-1)

        phi_gen = self.feature_extractor(
            pos=pos_gen,
            a_soft=a_soft_gen,
            batch_vec=batch.batch,
            dense_edge_index=batch.dense_edge_index,
        )
        phi_real = self.feature_extractor(
            pos=batch.pos,
            a_soft=batch.a_soft_real,
            batch_vec=batch.batch,
            dense_edge_index=batch.dense_edge_index,
        )

        test_loss, _ = self.compute_normalized_drift_loss(phi_gen, phi_real)

        batch_size = self._batch_size_for_logging(batch)
        self.log(
            "test_loss",
            test_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.drift_cfg["lr"],
            weight_decay=self.drift_cfg["weight_decay"],
            eps=1e-8,  # Adam epsilon (fine at 1e-8 here, as gradients are fp32)
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
                "interval": "step",  # Update the LR every batch, not every epoch
                "frequency": 1,
            },
        }
