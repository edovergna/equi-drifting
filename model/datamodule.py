"""PyTorch Lightning DataModule for QM9 molecular dataset.

Provides train/val/test dataloaders with atom-count-balanced batching and
transforms for encoding atom types and building fully-connected graphs.
"""

import random
import sys
from collections import defaultdict
from collections.abc import Iterator

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, Subset
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Center, Compose

from .sample_prior import get_dense_edge_index

_N_TRAIN = 110_000
_N_VAL = 10_000


class EncodeAtomTypesTransform:
    """A PyG transform that converts atomic numbers to one-hot vectors."""

    def __call__(self, data):
        """Convert atomic numbers in data.z to one-hot atom type vectors.

        Args:
            data: PyTorch Geometric Data object with z (atomic numbers) attribute.

        Returns:
            Modified Data object with real_atom_types added as one-hot encoding.
        """
        z_to_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

        real_indices = torch.tensor(
            [z_to_index[int(v.item())] for v in data.z],
            device=data.z.device,
        )
        data.real_atom_types = F.one_hot(real_indices, num_classes=5).float()

        return data


class FullyConnectedTransform:
    """A PyG transform that adds a fully connected dense_edge_index to the data."""

    def __call__(self, data: Data) -> Data:
        """Create a fully-connected graph edge index for a molecule.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Modified Data object with dense_edge_index added.
        """
        device = (
            data.edge_index.device
            if data.edge_index is not None
            else torch.device("cpu")
        )
        data.dense_edge_index = get_dense_edge_index(data.num_nodes, device)
        return data


class AtomCountBatchSampler(Sampler[list[int]]):
    """Yield batches where every molecule has the same number of atoms."""

    def __init__(
        self,
        indices_by_num_atoms: dict[int, list[int]],
        batch_size: int,
        shuffle: bool,
        replacement: bool,
        drop_last: bool = False,
    ):
        """Initialize the sampler with atom count groups.

        Args:
            indices_by_num_atoms: Dict mapping atom count to list of dataset indices.
            batch_size: Size of each batch.
            shuffle: Whether to shuffle molecule order within batches.
            replacement: Whether to allow sampling with replacement across batches.
            drop_last: If True, skip incomplete batches.
        """
        self.groups = {
            n_atoms: list(indices)
            for n_atoms, indices in indices_by_num_atoms.items()
            if len(indices) > 0
        }
        if not self.groups:
            raise ValueError("No non-empty atom-count groups available.")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replacement = replacement
        self.drop_last = drop_last

        if replacement:
            self.num_batches = max(
                1,
                sum(len(indices) for indices in self.groups.values()) // batch_size,
            )
        else:
            self.num_batches = sum(
                len(indices) // batch_size
                if drop_last
                else int(np.ceil(len(indices) / batch_size))
                for indices in self.groups.values()
            )

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices, one group at a time.

        Each batch contains molecules with the same number of atoms.
        """
        if self.replacement:
            atom_counts = list(self.groups)
            for _ in range(self.num_batches):
                n_atoms = random.choice(atom_counts) if self.shuffle else atom_counts[0]
                indices = self.groups[n_atoms]

                batch_size = min(self.batch_size, len(indices))
                yield random.sample(indices, batch_size)
            return

        for n_atoms, indices in self.groups.items():
            indices = indices.copy()
            if self.shuffle:
                random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        """Return the total number of batches."""
        return self.num_batches


class QM9DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the QM9 molecular dataset."""

    def __init__(
        self,
        root: str = "data/QM9",
        n_real_molecules: int = 128,
        num_workers: int = 4,
        force_reload: bool = False,
        sample_frac: float = 1.0,
        max_num_atoms: int | None = None,
        min_num_atoms: int | None = None,
    ):
        """Initialize the QM9 DataModule.

        Args:
            root: Root directory to cache the QM9 dataset.
            n_real_molecules: Batch size for dataloaders.
            num_workers: Number of worker processes for loading.
            force_reload: Force re-download and processing of the dataset.
            sample_frac: Fraction of data to use (for quick iteration).
            max_num_atoms: Filter molecules with more than this many atoms.
            min_num_atoms: Filter molecules with fewer than this many atoms.
        """
        super().__init__()
        self.root = root
        self.n_real_molecules = n_real_molecules
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.sample_frac = sample_frac
        self.max_num_atoms = max_num_atoms
        self.min_num_atoms = min_num_atoms
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage=None):
        """Load and prepare train/val/test splits from QM9 dataset.

        Applies atom count filtering and splits data proportionally by atom count
        to ensure balanced representation in all splits.
        """
        dataset = self._load_dataset()

        keep = [
            i
            for i, data in enumerate(dataset)
            if self._passes_atom_filter(int(data.num_nodes))
        ]
        dataset = dataset.index_select(keep)

        n = len(dataset)
        if n == 0:
            raise ValueError(
                "No QM9 molecules remain after filtering. "
                "Relax --max_num_atoms or --min_num_atoms or remove the filter."
            )

        n_train, n_val, _ = self._split_sizes(n)

        rng = np.random.default_rng(0)
        all_indices_by_num_atoms = self._group_by_num_atoms(dataset, np.arange(n))

        train_parts = []
        val_parts = []
        test_parts = []

        for num_atoms, indices in all_indices_by_num_atoms.items():
            indices = np.array(indices)
            rng.shuffle(indices)

            n_group = len(indices)

            if n_group == 1:
                n_train, n_val, n_test = 1, 0, 0
            elif n_group == 2:
                n_train, n_val, n_test = 1, 1, 0
            else:
                n_val = max(1, int(round(n_group * _N_VAL / (_N_TRAIN + _N_VAL))))
                n_test = max(1, int(round(n_group * 0.1)))
                n_train = n_group - n_val - n_test

                if n_train < 1:
                    n_train = 1
                    n_val = max(0, n_group - n_train - n_test)

            train_parts.append(indices[:n_train])
            val_parts.append(indices[n_train:n_train + n_val])
            test_parts.append(indices[n_train + n_val:])

        train_idx = np.concatenate(train_parts)
        val_idx = np.concatenate(val_parts)
        test_idx = np.concatenate(test_parts)

        if self.sample_frac < 1.0:
            srng = np.random.default_rng(42)
            train_idx = self._subsample_split(srng, train_idx)
            val_idx = self._subsample_split(srng, val_idx)
            test_idx = self._subsample_split(srng, test_idx)

        self.dataset = dataset

        self.train_set = Subset(dataset, train_idx)
        self.val_set = Subset(dataset, val_idx)
        self.test_set = Subset(dataset, test_idx)

        self.train_indices_by_num_atoms = self._group_by_num_atoms(dataset, train_idx)
        self.val_indices_by_num_atoms = self._group_by_num_atoms(dataset, val_idx)
        self.test_indices_by_num_atoms = self._group_by_num_atoms(dataset, test_idx)

        self.available_train_num_atoms = sorted(self.train_indices_by_num_atoms)
        self.available_val_num_atoms = sorted(self.val_indices_by_num_atoms)
        self.available_test_num_atoms = sorted(self.test_indices_by_num_atoms)

    def _load_dataset(self):
        """Load QM9 dataset with transforms, handling rdkit module issues.

        Returns:
            PyTorch Geometric Dataset with atom type encoding and fully-connected edges.
        """
        _rdkit_saved = {
            k: v
            for k, v in sys.modules.items()
            if k == "rdkit" or k.startswith("rdkit.")
        }
        for k in list(_rdkit_saved):
            sys.modules[k] = None  # type: ignore[assignment]
        sys.modules.setdefault("rdkit", None)  # type: ignore[assignment]

        try:
            return QM9(
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

    def _passes_atom_filter(self, num_atoms: int) -> bool:
        """Check if a molecule's atom count passes the filtering criteria.

        Args:
            num_atoms: Number of atoms in the molecule.

        Returns:
            True if the molecule should be kept, False otherwise.
        """
        if self.min_num_atoms is not None and num_atoms < self.min_num_atoms:
            return False
        if self.max_num_atoms is not None and num_atoms > self.max_num_atoms:
            return False
        return True

    def _split_sizes(self, n: int) -> tuple[int, int, int]:
        """Compute train/val/test split sizes for n total samples.

        Args:
            n: Total number of samples.

        Returns:
            Tuple of (n_train, n_val, n_test).
        """
        if n >= _N_TRAIN + _N_VAL:
            n_train = min(_N_TRAIN, n)
            n_val = min(_N_VAL, n - n_train)
            n_test = n - n_train - n_val
            return n_train, n_val, n_test

        n_val = max(1, int(round(n * _N_VAL / (_N_TRAIN + _N_VAL))))
        n_test = max(1, int(round(n * 0.1))) if n >= 3 else 0
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = 1
            n_val = max(0, n - n_train - n_test)

        return n_train, n_val, n_test

    def _group_by_num_atoms(
        self,
        dataset,
        indices: np.ndarray,
    ) -> dict[int, list[int]]:
        """Group dataset indices by the number of atoms in each molecule.

        Args:
            dataset: PyTorch Geometric Dataset.
            indices: Array of indices to group.

        Returns:
            Dict mapping atom count to list of indices.
        """
        groups = defaultdict(list)
        for idx in indices:
            idx = int(idx)
            groups[int(dataset[idx].num_nodes)].append(idx)
        return dict(groups)

    def _subsample_split(
        self,
        rng: np.random.Generator,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Randomly subsample a split to sample_frac of its original size.

        Args:
            rng: NumPy random generator.
            indices: Array of indices to subsample.

        Returns:
            Subsampled indices array.
        """
        if len(indices) == 0:
            return indices
        size = max(1, int(self.sample_frac * len(indices)))
        return rng.choice(indices, size=size, replace=False)

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader with atom-count-balanced batches.

        Returns:
            DataLoader with AtomCountBatchSampler.
        """
        return DataLoader(
            self.dataset,
            batch_sampler=AtomCountBatchSampler(
                self.train_indices_by_num_atoms,
                batch_size=self.n_real_molecules,
                shuffle=True,
                replacement=False,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader with atom-count-balanced batches.

        Returns:
            DataLoader with AtomCountBatchSampler.
        """
        return DataLoader(
            self.dataset,
            batch_sampler=AtomCountBatchSampler(
                self.val_indices_by_num_atoms,
                batch_size=self.n_real_molecules,
                shuffle=False,
                replacement=False,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader with atom-count-balanced batches.

        Returns:
            DataLoader with AtomCountBatchSampler.
        """
        return DataLoader(
            self.dataset,
            batch_sampler=AtomCountBatchSampler(
                self.test_indices_by_num_atoms,
                batch_size=self.n_real_molecules,
                shuffle=False,
                replacement=False,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )