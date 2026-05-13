import sys

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Center, Compose

from .sample_prior import get_dense_edge_index

# Standard QM9 split sizes (after excluding 3,054 uncharacterized molecules).
# Matches gen_splits_gdb9 in the EPT preprocessing script: seed=0, 110k/10k/rest.
_N_TRAIN = 110_000
_N_VAL = 10_000


class EncodeAtomTypesTransform:
    """A PyG transform that converts atomic numbers to one-hot vectors."""

    def __call__(self, data):
        z_to_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

        # We use data.z.device to ensure it stays on the right hardware
        real_indices = torch.tensor(
            [z_to_index[int(v.item())] for v in data.z], device=data.z.device
        )

        # Attach the result directly to the PyG Data object
        data.real_atom_types = F.one_hot(real_indices, num_classes=5).float()

        return data


class FullyConnectedTransform:
    """A PyG transform that adds a fully connected dense_edge_index to the data."""

    def __call__(self, data: Data) -> Data:
        device = (
            data.edge_index.device
            if data.edge_index is not None
            else torch.device("cpu")
        )
        data.dense_edge_index = get_dense_edge_index(data.num_nodes, device)
        return data

# TODO: pre-transform for training molecules
class QM9DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "data/QM9",
        n_real_molecules: int = 128,
        num_workers: int = 4,
        force_reload: bool = False,
        sample_frac: float = 1.0,
        max_num_atoms: int | None = None,
    ):
        super().__init__()
        self.root = root
        self.n_real_molecules = n_real_molecules
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.sample_frac = sample_frac
        self.max_num_atoms = max_num_atoms
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage=None):
        # rdkit ≥2026 returns None for ~41 malformed molecules in gdb9.sdf, but
        # torch_geometric's QM9.process() has no None guard. Setting rdkit entries
        # to None in sys.modules makes `import rdkit` raise ImportError inside QM9,
        # forcing it to download and use the pre-processed qm9_v3.pt instead.
        _rdkit_saved = {
            k: v
            for k, v in sys.modules.items()
            if k == "rdkit" or k.startswith("rdkit.")
        }
        for k in list(_rdkit_saved):
            sys.modules[k] = None  # type: ignore[assignment]
        sys.modules.setdefault("rdkit", None)  # type: ignore[assignment]
        try:
            dataset = QM9(
                self.root,
                pre_transform=Compose(
                    [Center(), FullyConnectedTransform(), EncodeAtomTypesTransform()]
                ),
                force_reload=self.force_reload,
            )
        finally:
            for k in list(sys.modules):
                if sys.modules[k] is None and (k == "rdkit" or k.startswith("rdkit.")):
                    del sys.modules[k]
            sys.modules.update(_rdkit_saved)

        if self.max_num_atoms is not None:
            keep = [
                i for i, d in enumerate(dataset) if d.num_nodes <= self.max_num_atoms
            ]
            dataset = dataset.index_select(keep)

        n = len(dataset)
        if n == 0:
            raise ValueError(
                "No QM9 molecules remain after filtering. "
                "Relax --max_num_atoms or remove the filter."
            )

        if n >= _N_TRAIN + _N_VAL:
            n_train = min(_N_TRAIN, n)
            n_val = min(_N_VAL, n - n_train)
            n_test = n - n_train - n_val
        else:
            # When filters such as --max_num_atoms shrink the dataset below the
            # canonical QM9 split sizes, keep the same rough split proportions.
            n_val = max(1, int(round(n * _N_VAL / (_N_TRAIN + _N_VAL))))
            n_test = max(1, int(round(n * 0.1))) if n >= 3 else 0
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                n_val = max(0, n - n_train - n_test)

        # Reproducible permutation matching the EPT standard split (seed=0)
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        if self.sample_frac < 1.0:
            # Subsample each split proportionally, with a fixed secondary seed
            srng = np.random.default_rng(42)
            train_idx = self._subsample_split(srng, train_idx)
            val_idx = self._subsample_split(srng, val_idx)
            test_idx = self._subsample_split(srng, test_idx)

        self.train_set = Subset(dataset, train_idx)
        self.val_set = Subset(dataset, val_idx)
        self.test_set = Subset(dataset, test_idx)

    def _subsample_split(
        self, rng: np.random.Generator, indices: np.ndarray
    ) -> np.ndarray:
        if len(indices) == 0:
            return indices
        size = max(1, int(self.sample_frac * len(indices)))
        return rng.choice(indices, size=size, replace=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.n_real_molecules,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.n_real_molecules,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.n_real_molecules,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
