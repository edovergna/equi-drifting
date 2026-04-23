import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from ept import EPTFeatureExtractor
from model.egnn import EGNN

from .egnn import EGNN


class DriftingMoleculeGenerator(LightningModule):
    def __init__(self, generator_cfg, drift_cfg):
        super().__init__()
        self.save_hyperparameters()

        # Generator: Your EGNN or GNN architecture
        self.generator = self._init_generator(generator_cfg)

        # Feature Space: The paper suggests drifting in a feature space
        # For now, this can be a simple linear layer or a small GNN encoder
        self.feature_extractor = self._init_feature_extractor()

        # Hyperparameters for Drifting Field V
        self.temperatures = [0.02, 0.05, 0.2]

    def _init_generator(self, cfg) -> EGNN:
        # Placeholder for your EGNN initialization
        # TODO: add configuration options to parse args.
        return EGNN(
            in_node_nf=7,  # Example: 5 for one-hot atom type + 2 for other features
            hidden_nf=128,
            n_layers=2,
            num_atom_types=5,  # Example: C, O, N, S, H
            num_bond_types=5,  # Example: single, double, triple, aromatic, no bond
        )

    def _init_feature_extractor(self):
        # Check if the checkpoint has been downloaded and download it if not
        root_path = Path(__file__).parents[1]

        # Hard coding path for now, unsure if making it configurable is worth it.
        ckpt_path = "hybrid_noaf/epoch49_step215752.ckpt"

        full_ckpt_path = root_path / ckpt_path

        if not full_ckpt_path.exists():
            print(f"Checkpoint not found at {full_ckpt_path}. Downloading...")
            print("Downloading EPT from google drive...")
            print(
                "os.system('gdown --folder https://drive.google.com/drive/folders/1tBqGwC_jcTdq3QArFZox_auSCzxDjA0P')"
            )
            os.system(
                "gdown --folder https://drive.google.com/drive/folders/1tBqGwC_jcTdq3QArFZox_auSCzxDjA0P"
            )
            print("Download complete.")

        # Add the "ept" directory to sys.path so that torch.load finds the EPT modules
        ept_path = str(root_path / "ept")
        if ept_path not in sys.path:
            sys.path.append(ept_path)

        return EPTFeatureExtractor(ckpt_path, torch.device("cpu")) # start it on cpu bc lightning will fix it in next helper funcs

    def on_fit_start(self):
        self.feature_extractor.to(self.device) # here
        self.feature_extractor.eval()

    def on_test_start(self):
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

    def compute_v(self, x, y_pos, y_neg, tau):
        """
        Implements Algorithm 2: Computing the drifting field V.
        x: Generated features (q)
        y_pos: Real data features (p)
        y_neg: Other generated samples (q) for repulsion
        """
        # 1. Compute Pairwise Distances
        dist_pos = torch.cdist(x, y_pos)
        dist_neg = torch.cdist(x, y_neg)

        # Ignore self in repulsion (if y_neg is x)
        # dist_neg += torch.eye(x.size(0)).to(x.device) * 1e6

        # 2. Compute Logits and Kernels
        logit_pos = -dist_pos / tau
        logit_neg = -dist_neg / tau

        # 3. Normalization along both dimensions (Anti-symmetry)
        # Normalized kernels (A_pos, A_neg)
        # Placeholder for softmax/sqrt normalization logic from Alg 2

        # 4. Compute Weighted Drift
        # V = V_attraction (from p) - V_repulsion (from q)
        v_field = torch.zeros_like(x)  # Placeholder
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

    def _batch_size_for_logging(self, batch) -> int:
        if hasattr(batch, "num_graphs") and batch.num_graphs is not None:
            return int(batch.num_graphs)

        if hasattr(batch, "batch") and batch.batch is not None:
            return int(batch.batch.max().item()) + 1

        return 1

    def training_step(self, batch, batch_idx):
        """
        Training-time evolution of the pushforward distribution.
        """
        # Sample from the prior distribution
        x_prior, pos_prior = self.sample_prior(batch.num_nodes)

        # Forward Pass: Map Prior (e) to Generated (x)
        x_gen, edge_bond_logits, pos_gen = self.generator(
            x_prior,
            pos_prior,
            batch.edge_index,
        )

        # breakpoint()

        a_soft_gen = F.gumbel_softmax(x_gen, tau=1.0, hard=False, dim=-1)
        
        phi_gen = self.feature_extractor(
            pos=pos_gen, 
            a_soft=a_soft_gen, 
            batch_vec=batch.batch, 
            dense_edge_index=batch.dense_edge_index # <--- FIXED: Must be dense!
        )

        with torch.no_grad(): 
            phi_real = self.feature_extractor(
                pos=batch.pos, 
                a_soft=batch.a_soft_real, # <--- Cleanly pulled straight from the batch!
                batch_vec=batch.batch, 
                dense_edge_index=batch.dense_edge_index # <--- Matches the generated side perfectly.
            )

        # Compute the Aggregated Drifting Field V
        # Usually summed across multiple temperatures
        total_v = torch.zeros_like(phi_gen)
        for t in self.temperatures:
            total_v += self.compute_v(phi_gen, phi_real, phi_gen, t)

        # Stop-Gradient Loss
        # We move x towards (x + V) without backpropping through V itself
        target = (phi_gen + total_v).detach()

        # Equation 6
        loss = F.mse_loss(phi_gen, target)

        batch_size = self._batch_size_for_logging(batch) # problem with progressbar fix
        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_batch_size",
            float(batch_size),
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
        )
 
        return loss

    def test_step(self, batch, batch_idx):
        x_prior, pos_prior = self.sample_prior(batch.num_nodes)
        x_gen, edge_bond_logits, pos_gen = self.generator(
            x_prior,
            pos_prior,
            batch.edge_index,
        )
        a_soft_gen = F.gumbel_softmax(x_gen, tau=1.0, hard=False, dim=-1)
        phi_gen = self.feature_extractor(
            pos=pos_gen, 
            a_soft=a_soft_gen, 
            batch_vec=batch.batch, 
            dense_edge_index=batch.dense_edge_index
        )
        phi_real = self.feature_extractor(
            pos=batch.pos, 
            a_soft=batch.a_soft_real, 
            batch_vec=batch.batch, 
            dense_edge_index=batch.dense_edge_index
        )           
        test_loss = F.mse_loss(phi_gen, phi_real)
        batch_size = self._batch_size_for_logging(batch)
        self.log(
            "test_loss",
            test_loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_batch_size",
            float(batch_size),
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )


    # def validation_step(self, batch, batch_idx): # commented out for quick train-test loop testing
    #     x_prior, pos_prior = self.sample_prior(batch.num_nodes)
    #     x_gen, edge_bond_logits, pos_gen = self.generator(
    #         x_prior,
    #         pos_prior,
    #         batch.edge_index,
    #     )
    #     a_soft_gen = F.gumbel_softmax(x_gen, tau=1.0, hard=False, dim=-1)
    #     phi_gen = self.feature_extractor(
    #         pos=pos_gen,
    #         a_soft=a_soft_gen,
    #         batch_vec=batch.batch,
    #         dense_edge_index=batch.dense_edge_index,
    #     )
    #     phi_real = self.feature_extractor(
    #         pos=batch.pos,
    #         a_soft=batch.a_soft_real,
    #         batch_vec=batch.batch,
    #         dense_edge_index=batch.dense_edge_index,
    #     )
    #     val_loss = F.mse_loss(phi_gen, phi_real)
    #     batch_size = self._batch_size_for_logging(batch)
    #     self.log(
    #         "val_loss",
    #         val_loss,
    #         batch_size=batch_size,
    #         prog_bar=True,
    #         on_step=False,
    #         on_epoch=True,
    #     )
  

    def configure_optimizers(self):
        # The paper uses AdamW with specific beta values
        return torch.optim.AdamW(self.parameters(), lr=4e-4, betas=(0.9, 0.95))
