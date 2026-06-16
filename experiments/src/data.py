from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.datasets import QM9

from .utils import center_positions_per_mol

DATA_ROOT = Path(__file__).parents[2] / "data" / "QM9"


@dataclass
class MolBatch:
    """A batch of molecules with per-atom tensors."""
    pos: torch.Tensor         # (n_mols * num_atoms, 3)
    atom_types: torch.Tensor  # (n_mols * num_atoms,)
    batch: torch.Tensor       # (n_mols * num_atoms,) — molecule index per atom
    edge_index: torch.Tensor  # (2, n_edges)
    n_mols: int
    num_atoms: int

    @property
    def pos_flat(self) -> torch.Tensor:
        """Positions reshaped to (n_mols, num_atoms * 3)."""
        return self.pos.view(self.n_mols, -1)


@dataclass
class MolData:
    """Container pairing real and generated molecule batches."""
    real: MolBatch
    gen: MolBatch
    n_real_mols: int
    n_gen_mols: int
    mol_indices: torch.Tensor

    @property
    def real_pos_repeated(self) -> torch.Tensor:
        """Real positions repeated once per gen mol: (total_mols * num_atoms, 3)."""
        if self.n_gen_mols > self.n_real_mols:
            return self.real.pos.repeat(self.n_gen_mols // self.n_real_mols, 1)
        return self.real.pos
    
    @property
    def real_pos_3d(self) -> torch.Tensor:
        """Real positions reshaped to (n_mols, num_atoms, 3)."""
        return self.real.pos.view(self.n_real_mols, self.real.num_atoms, 3)

    @property
    def real_pos_flattened(self) -> torch.Tensor:
        """Real positions flattened and repeated: (total_mols, num_atoms * 3)."""
        return self.real.pos.view(self.n_real_mols, -1).repeat(self.n_gen_mols, 1)


def load_and_filter_data(
    num_atoms: int, n_real_mols: int, n_gen_mols: int, device: torch.device
) -> MolData:
    """
    Load the QM9 dataset and filter it to only include molecules with a specific number of atoms.

    Args:
        num_atoms: The number of atoms that the molecules should have to be included in the filtered
                    dataset.
        n_real_mols: The number of molecules to include in the filtered dataset.
        n_gen_mols: The number of generated molecules to include.
        device: The device to move the tensors to.

    Returns:
        A MolData object containing the real and generated molecule batches.
    """
    assert n_gen_mols % n_real_mols == 0, "n_gen_mols must be a multiple of n_real_mols for proper batching."
    
    dataset = QM9(DATA_ROOT)

    keep = []
    for i, d in enumerate(dataset):
        if d.num_nodes == num_atoms:
            keep.append(i)
        if len(keep) >= n_real_mols:
            break
    filtered_dataset = dataset.index_select(keep)

    # Create batch vector for real molecules and center their positions
    real_batch_vec = torch.arange(n_real_mols, device=device).repeat_interleave(num_atoms)
    # Zero center the real molecule positions
    real_pos = filtered_dataset.pos.to(device)
    real_pos = center_positions_per_mol(real_pos, real_batch_vec)

    one_hot_real_types = filtered_dataset.real_atom_types.argmax(dim=-1).to(device)
    print(f"Real molecule atom types shape: {one_hot_real_types.shape}")

    real = MolBatch(
        pos=real_pos,
        atom_types=one_hot_real_types,
        batch=real_batch_vec,
        edge_index=filtered_dataset.dense_edge_index.to(device),
        n_mols=n_real_mols,
        num_atoms=num_atoms,
    )

    # Repeat the real molecule types for each generated molecule.
    # This way we can align each generated molecule to each real molecule during training.
    if n_gen_mols > n_real_mols:
        gen_atom_types = one_hot_real_types.repeat(n_gen_mols // n_real_mols)
    else:
        gen_atom_types = one_hot_real_types
    print(f"Gen atom types shape: {gen_atom_types.shape}")

    # Sample noise once with the same shape as the real molecules.
    # We generate N_GEN_MOLS for each of the N_REAL_MOLS,
    # so we need to repeat the noise accordingly.
    gen_batch_vec = torch.arange(n_gen_mols, device=device).repeat_interleave(num_atoms)
    pos_noise = torch.randn((n_gen_mols * num_atoms, 3), device=device)
    pos_noise = center_positions_per_mol(pos_noise, gen_batch_vec)
    print(f"Pos noise shape: {pos_noise.shape}")
    print(f"Gen batch vec shape: {gen_batch_vec.shape}")

    # Build edge index for all (n_gen_mols * n_real_mols) generated molecules
    edge_index_list = []
    for offset in range(n_gen_mols // n_real_mols):
        offset_edge_index = filtered_dataset.dense_edge_index + offset * num_atoms
        edge_index_list.append(offset_edge_index)
    dense_edge_index = torch.cat(edge_index_list, dim=1).to(device)

    gen = MolBatch(
        pos=pos_noise,
        atom_types=gen_atom_types,
        batch=gen_batch_vec,
        edge_index=dense_edge_index,
        n_mols=n_gen_mols,
        num_atoms=num_atoms,
    )
    
    mol_indices = torch.arange(n_gen_mols, device=device)
    return MolData(
        real=real,
        gen=gen,
        n_real_mols=n_real_mols,
        n_gen_mols=n_gen_mols,
        mol_indices=mol_indices
    )
