from collections import Counter

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem.rdchem import BondType


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

# Generic distance threshold (Å) used before atom types are known
_GENERIC_BOND_THRESHOLD = 2.0

# Bond length tables (pm) for QM9 atoms — ported from the EDM/E-NF reference.
# Source: http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
_BONDS1 = {
    'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92},
    'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135},
    'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136},
    'O': {'H': 96,  'C': 143, 'N': 140, 'O': 148, 'F': 142},
    'F': {'H': 92,  'C': 135, 'N': 136, 'O': 142, 'F': 142},
}
_BONDS2 = {
    'C': {'C': 134, 'N': 129, 'O': 120},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121},
}
_BONDS3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113},
}
_MARGIN1, _MARGIN2, _MARGIN3 = 10, 5, 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def infer_types_from_pos_batch(
    pos: torch.Tensor,
    batch_vec: torch.Tensor,
    device: torch.device,
    num_atom_types: int = 5,
) -> torch.Tensor:
    """
    Infer QM9 atom types from generated positions using connectivity degree.

    Counts neighbors within 2 Å per atom and maps degree → type:
    C (≥4), N (3), O (2), H (≤1).

    Returns:
        atom_types: [N, num_atom_types] one-hot float tensor on `device`
    """
    pos_np = pos.detach().cpu().numpy().astype(np.float64)
    batch_np = batch_vec.detach().cpu().numpy()
    n_total = pos_np.shape[0]
    n_graphs = int(batch_np.max()) + 1

    type_indices = np.zeros(n_total, dtype=np.int64)
    for g in range(n_graphs):
        mask = batch_np == g
        type_indices[mask] = _infer_types_from_degree(pos_np[mask])

    atom_types = torch.zeros(n_total, num_atom_types)
    atom_types[torch.arange(n_total), torch.from_numpy(type_indices)] = 1.0
    return atom_types.to(device)


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


def get_bond_order(atom1: str, atom2: str, distance: float) -> int:
    """Return bond order (0=none, 1=single, 2=double, 3=triple).

    Ported directly from the EDM/E-NF reference implementation.
    distance must be in Angstroms; it is converted to pm internally.
    """
    dist_pm = distance * 100
    if atom1 not in _BONDS1 or atom2 not in _BONDS1[atom1]:
        return 0
    if dist_pm < _BONDS1[atom1][atom2] + _MARGIN1:
        if atom1 in _BONDS2 and atom2 in _BONDS2.get(atom1, {}):
            if dist_pm < _BONDS2[atom1][atom2] + _MARGIN2:
                if atom1 in _BONDS3 and atom2 in _BONDS3.get(atom1, {}):
                    if dist_pm < _BONDS3[atom1][atom2] + _MARGIN3:
                        return 3
                return 2
        return 1
    return 0


# Bond length threshold (Å) separating H (~1.0-1.1) from F (~1.3-1.4)
# for terminal atoms (degree 1). Mid-point between typical C-H and C-F lengths.
_H_F_BOND_THRESHOLD = 1.2


def _infer_types_from_degree(positions: np.ndarray) -> np.ndarray:
    """Assign QM9 atom type indices from pairwise-distance connectivity degrees.

    Heavy atoms: C (≥4 neighbors), N (3), O (2).
    Terminal atoms (degree 1): F if bond length > _H_F_BOND_THRESHOLD, else H.
    Isolated atoms (degree 0): default to H.
    """
    n = len(positions)
    if n == 0:
        return np.array([], dtype=np.int64)
    dists = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    adj = (dists < _GENERIC_BOND_THRESHOLD) & (dists > 0)
    degrees = adj.sum(axis=1)

    type_indices = np.where(
        degrees >= 4, 1,           # C
        np.where(degrees == 3, 2,  # N
        np.where(degrees == 2, 3,  # O
        0)),                       # H or F (degree <= 1), resolved below
    )

    # For terminal atoms, use bond length to distinguish H from F
    terminal_mask = degrees == 1
    if terminal_mask.any():
        terminal_idx = np.where(terminal_mask)[0]
        for i in terminal_idx:
            neighbor = np.argmin(np.where(adj[i], dists[i], np.inf))
            if dists[i, neighbor] > _H_F_BOND_THRESHOLD:
                type_indices[i] = 4  # F

    return type_indices.astype(np.int64)


def _assess_molecule(
    positions: np.ndarray, atom_type_indices: np.ndarray
) -> tuple[bool, str | None]:
    """Try RDKit first, fall back to pure-numpy valence check."""
    try:
        return _rdkit_assess(positions, atom_type_indices)
    except ImportError:
        return _numpy_assess(positions, atom_type_indices)


_RDKIT_BOND_TYPES = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]


def _rdkit_assess(
    positions: np.ndarray, atom_type_indices: np.ndarray
) -> tuple[bool, str | None]:
    """Build molecule with get_bond_order, matching the EDM/E-NF reference exactly."""
    atom_names = [_ATOM_NAMES[int(i)] for i in atom_type_indices]
    atomic_nums = [_ATOMIC_NUMS[int(i)] for i in atom_type_indices]
    n = len(atom_names)

    mol = RWMol()
    for anum in atomic_nums:
        mol.AddAtom(Chem.Atom(int(anum)))

    for i in range(n):
        for j in range(i):  # lower triangle only, matching build_xae_molecule
            dist = np.linalg.norm(positions[i] - positions[j])
            order = get_bond_order(atom_names[i], atom_names[j], dist)
            if order > 0:
                mol.AddBond(j, i, _RDKIT_BOND_TYPES[order])

    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False, None

    try:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest = max(frags, default=mol, key=lambda m: m.GetNumAtoms())
        return True, Chem.MolToSmiles(largest)
    except Exception:
        return False, None


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
    """Returns a boolean array: True for each atom that has its target valence.

    Bond orders (single=1, double=2, triple=3) are summed per atom, matching
    the EDM/E-NF reference implementation exactly.
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
