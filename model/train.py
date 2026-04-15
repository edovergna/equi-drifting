# train_qm9_lightning.py

import torch
import torch.nn.functional as F
from torch.utils.data import random_split

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from EGNN import EGNNVelocity


ROOT = "data/QM9"
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-6
TYPE_LOSS_WEIGHT = 0.1


class QM9DataModule(pl.LightningDataModule):
    def __init__(self, root=ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = QM9(self.root)

        n = len(dataset)
        n_train = int(0.9 * n)
        n_val = int(0.05 * n)
        n_test = n - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class LitFlowMatching(pl.LightningModule):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=4,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        type_loss_weight=TYPE_LOSS_WEIGHT,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = EGNNVelocity(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_atom_types=5,  # H, C, N, O, F
            num_bond_types=4,  # single, double, triple, aromatic
            space_dim=3,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _sample_flow_batch(self, batch):
        # Original molecule positions
        X1 = batch.pos.float()
        batch_idx = batch.batch

        # Gaussian source, zero-centered per graph
        X0 = torch.randn_like(X1)
        X0 = X0 - scatter(X0, batch_idx, dim=0, reduce="mean")[batch_idx]

        # One time per graph
        t_graph = torch.rand(batch.num_graphs, device=X1.device)
        t_node = t_graph[batch_idx].unsqueeze(-1)

        # Linear interpolation path x_t = (1-t)x0 + t x1
        Xt = (1.0 - t_node) * X0 + t_node * X1

        # Conditional flow matching target
        Vt = X1 - X0

        return Xt, Vt, t_graph

    def _shared_step(self, batch, stage: str):
        # Convert QM9 batch fields
        _, A, C, edge_index, E, batch_idx = self.model.qm9_to_inputs(batch)

        Xt, Vt, t = self._sample_flow_batch(batch)

        out = self.model(
            X=Xt,
            A=A,
            C=C,
            edge_index=edge_index,
            E=E,
            time=t,
            batch=batch_idx,
        )

        loss_v = F.mse_loss(out["velocity"], Vt)
        loss_type = F.cross_entropy(out["type_logits"], A)
        loss = loss_v + self.hparams.type_loss_weight * loss_type

        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}/loss_v", loss_v, batch_size=batch.num_graphs)
        self.log(f"{stage}/loss_type", loss_type, batch_size=batch.num_graphs)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")
