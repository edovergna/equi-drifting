import numpy as np
import torch

from .bonds import get_bond_order
from .constants import _ATOM_NAMES, _ATOMIC_NUMS, _STABLE_VALENCE


def heavy_atom_counts(atom_types: torch.Tensor, batch_vec: torch.Tensor) -> list[int]:
    """Returns the number of non-hydrogen atoms per graph in the batch."""
    n_graphs = int(batch_vec.max().item()) + 1
    counts = []
    for g in range(n_graphs):
        mask = batch_vec == g
        types = atom_types[mask]
        counts.append(int((types != 0).sum().item()))
    return counts


def batch_to_stability(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
) -> tuple[float, float]:
    """
    Compute atom-level and molecule-level stability.

    An atom is stable iff its bond count equals _STABLE_VALENCE for its element.
    A molecule is stable iff every atom in it is stable.

    Returns:
        atom_stable_frac  — fraction of all atoms that are stable
        mol_stable_frac   — fraction of molecules where all atoms are stable
    """
    n_graphs = int(batch_vec.max().item()) + 1
    total_atoms = 0
    n_stable_atoms = 0
    n_stable_mols = 0

    for g in range(n_graphs):
        mask = batch_vec == g
        p = pos[mask].numpy().astype(np.float64)
        types = atom_types[mask].numpy()
        stable = _per_atom_stability(p, types)
        n_stable_atoms += int(stable.sum())
        total_atoms += len(stable)
        if stable.all():
            n_stable_mols += 1

    atom_stable_frac = n_stable_atoms / total_atoms if total_atoms > 0 else 0.0
    mol_stable_frac = n_stable_mols / n_graphs if n_graphs > 0 else 0.0
    return atom_stable_frac, mol_stable_frac


def _per_atom_stability(
    positions: np.ndarray, atom_type_indices: np.ndarray
) -> np.ndarray:
    """
    Compute atom-level and molecule-level stability.

    An atom is stable iff its bond count equals _STABLE_VALENCE for its element.
    A molecule is stable iff every atom in it is stable.

    Returns:
        atom_stable_frac  — fraction of all atoms that are stable
        mol_stable_frac   — fraction of molecules where all atoms are stable
    """
    atom_names = [_ATOM_NAMES[int(i)] for i in atom_type_indices]
    atomic_nums = [_ATOMIC_NUMS[int(i)] for i in atom_type_indices]
    n = len(atom_names)
    bond_orders = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            order = get_bond_order(atom_names[i], atom_names[j], dist)
            bond_orders[i] += order
            bond_orders[j] += order

    target = np.array([_STABLE_VALENCE[a] for a in atomic_nums], dtype=int)
    return bond_orders == target
