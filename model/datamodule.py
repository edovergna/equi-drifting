import lightning.pytorch as pl
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Center, Compose

import torch.nn.functional as F

class EncodeAtomTypesTransform:
    """A PyG transform that converts atomic numbers to one-hot vectors."""
    def __call__(self, data):
        z_to_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        
        # We use data.z.device to ensure it stays on the right hardware
        real_indices = torch.tensor([z_to_index[int(v.item())] for v in data.z], device=data.z.device)
        
        # Attach the result directly to the PyG Data object
        data.a_soft_real = F.one_hot(real_indices, num_classes=5).float()
        
        return data
    
class FullyConnectedTransform:
    """A PyG transform that adds a fully connected dense_edge_index to the data."""

    def __call__(self, data: Data) -> Data:
        # Get device and number of nodes
        device = (
            data.edge_index.device
            if data.edge_index is not None
            else torch.device("cpu")
        )
        n = data.num_nodes

        # Generate all pairs
        row = torch.arange(n, device=device).repeat_interleave(n)
        col = torch.arange(n, device=device).repeat(n)

        # Remove self-loops
        mask = row != col

        edge_index = torch.stack([row[mask], col[mask]], dim=0)

        data.dense_edge_index = edge_index

        return data


class QM9DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "data/QM9",
        batch_size: int = 128,
        num_workers: int = 4,
        force_reload: bool = False,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage=None):
        # Note that the pre_transform is applied once when loading the dataset.
        # We are centering the molecules and adding a fully connected dense_edge_index
        # to each graph.
        dataset = QM9(
            self.root,
            pre_transform=Compose([
                Center(), 
                FullyConnectedTransform(),
                EncodeAtomTypesTransform()
            ]),
            force_reload=self.force_reload,
        )

        n = len(dataset)
        n_train = int(0.8 * n)
        n_val = int(0.10 * n)
        n_test = n - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
        #### to see if overfits on train, skipping val and straight to test
        # n_train = int(0.025 * n)
        # n_val = int(0.95 * n)
        # n_test = n - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )


    

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
