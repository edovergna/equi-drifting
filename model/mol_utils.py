from collections import Counter

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Conformer, RWMol
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.rdBase import BlockLogs

# QM9 atom ordering matches EncodeAtomTypesTransform: {H, C, N, O, F}
_ATOMIC_NUMS = [1, 6, 7, 8, 9]
_ATOM_NAMES = ["H", "C", "N", "O", "F"]

# Covalent radii in Ångströms (Alvarez 2008)
_COV_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}
_BOND_FACTOR = 1.3  # bond exists when dist < factor * (r_i + r_j)

# Maximum valence per element (conservative: allows for ionic/charged forms)
_MAX_VALENCE = {1: 1, 6: 4, 7: 4, 8: 3, 9: 1}

# Typical (target) valence used for stability: atom is stable iff bond_count == this
_STABLE_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _ensure_atom_ids(atom_types: torch.Tensor) -> torch.Tensor:
    # One-hot / probabilities
    if atom_types.ndim == 2:
        return atom_types.argmax(dim=-1)

    # Already integer IDs
    return atom_types


def batch_to_validity(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
) -> list[tuple[bool, str | None]]:
    """
    Convert a batched PyG tensor to per-molecule (is_valid, identifier) pairs.

    `identifier` is a canonical SMILES string when RDKit is available, or a
    molecular formula string otherwise (useful for uniqueness tracking).
    """
    atom_types = _ensure_atom_ids(atom_types)
    n_graphs = int(batch_vec.max().item()) + 1
    results: list[tuple[bool, str | None]] = []
    for g in range(n_graphs):
        mask = batch_vec == g
        p = pos[mask].numpy().astype(np.float64)
        types = atom_types[mask].numpy()
        results.append(_assess_molecule(p, types))
    return results


def heavy_atom_counts(atom_types: torch.Tensor, batch_vec: torch.Tensor) -> list[int]:
    """Returns the number of non-hydrogen atoms per graph in the batch."""
    atom_types = _ensure_atom_ids(atom_types)
    n_graphs = int(batch_vec.max().item()) + 1
    counts = []
    for g in range(n_graphs):
        mask = batch_vec == g
        types = atom_types[mask]
        counts.append(int((types != 0).sum().item()))  # 0 == H in our encoding
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
    atom_types = _ensure_atom_ids(atom_types)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assess_molecule(
    positions: np.ndarray, atom_type_indices: np.ndarray
) -> tuple[bool, str | None]:
    """Try RDKit first, fall back to pure-numpy valence check."""
    try:
        return _rdkit_assess(positions, atom_type_indices)
    except ImportError:
        return _numpy_assess(positions, atom_type_indices)


def _rdkit_assess(
    positions: np.ndarray, atom_type_indices: np.ndarray
) -> tuple[bool, str | None]:
    atomic_nums = [_ATOMIC_NUMS[int(i)] for i in atom_type_indices]
    n = len(atomic_nums)

    with BlockLogs():
        mol = RWMol()
        conf = Conformer(n)
        for i, anum in enumerate(atomic_nums):
            mol.AddAtom(Chem.Atom(int(anum)))
            conf.SetAtomPosition(i, positions[i].tolist())
        mol.AddConformer(conf, assignId=True)

        try:
            DetermineBonds(mol, charge=0)
        except Exception:
            # DetermineBonds may have partially added bonds before failing; rebuild clean.
            mol = RWMol()
            conf = Conformer(n)
            for i, anum in enumerate(atomic_nums):
                mol.AddAtom(Chem.Atom(int(anum)))
                conf.SetAtomPosition(i, positions[i].tolist())
            mol.AddConformer(conf, assignId=True)
            _add_bonds_by_distance(mol, positions, atomic_nums)

        if Chem.SanitizeMol(mol, catchErrors=True):
            return False, None
        try:
            smi = Chem.MolToSmiles(mol.GetMol())
            return True, smi
        except Exception:
            return False, None


def _add_bonds_by_distance(mol, positions: np.ndarray, atomic_nums: list[int]) -> None:
    from rdkit.Chem.rdchem import BondType
    
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < _BOND_FACTOR * (
                _COV_RADII[atomic_nums[i]] + _COV_RADII[atomic_nums[j]]
            ):
                mol.AddBond(i, j, BondType.SINGLE)


def _numpy_assess(
    positions: np.ndarray, atom_type_indices: np.ndarray
) -> tuple[bool, str | None]:
    """Fallback: distance-based bond assignment + max-valence check."""
    atomic_nums = [_ATOMIC_NUMS[int(i)] for i in atom_type_indices]
    n = len(atomic_nums)
    degrees = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < _BOND_FACTOR * (
                _COV_RADII[atomic_nums[i]] + _COV_RADII[atomic_nums[j]]
            ):
                degrees[i] += 1
                degrees[j] += 1

    for i, anum in enumerate(atomic_nums):
        if degrees[i] == 0 or degrees[i] > _MAX_VALENCE[anum]:
            return False, None

    # Use molecular formula as a proxy identifier (not unique per structure)
    counts = Counter(int(t) for t in atom_type_indices)
    formula = "".join(f"{_ATOM_NAMES[k]}{v}" for k, v in sorted(counts.items()))
    return True, formula


def _per_atom_stability(
    positions: np.ndarray, atom_type_indices: np.ndarray
) -> np.ndarray:
    """Returns a boolean array: True for each atom that has its target valence."""
    atomic_nums = [_ATOMIC_NUMS[int(i)] for i in atom_type_indices]
    n = len(atomic_nums)
    degrees = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < _BOND_FACTOR * (
                _COV_RADII[atomic_nums[i]] + _COV_RADII[atomic_nums[j]]
            ):
                degrees[i] += 1
                degrees[j] += 1

    target = np.array([_STABLE_VALENCE[a] for a in atomic_nums], dtype=int)
    return degrees == target
