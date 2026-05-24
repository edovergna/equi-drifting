"""Molecular validity assessment utilities."""

from collections import Counter

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from .bonds import get_bond_order, _RDKIT_BOND_TYPES
from .constants import (
    _ATOM_NAMES,
    _ATOMIC_NUMS,
    _BOND_FACTOR,
    _COV_RADII,
    _MAX_VALENCE,
)


def batch_to_validity(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
) -> list[tuple[bool, str | None]]:
    """Assess validity and get identifiers for molecules in a batch.

    Args:
        pos: Atomic positions [total_nodes, 3].
        atom_types: Atom type indices [total_nodes].
        batch_vec: Batch indices [total_nodes] mapping atoms to molecules.

    Returns:
        List of (is_valid, identifier) tuples where identifier is a canonical
        SMILES string (if RDKit available) or molecular formula.
    """
    n_graphs = int(batch_vec.max().item()) + 1
    results: list[tuple[bool, str | None]] = []
    for g in range(n_graphs):
        mask = batch_vec == g
        p = pos[mask].numpy().astype(np.float64)
        types = atom_types[mask].numpy()
        results.append(_assess_molecule(p, types))
    return results


def _assess_molecule(
    positions: np.ndarray, atom_type_indices: np.ndarray
) -> tuple[bool, str | None]:
    """Assess molecular validity and extract canonical identifier.

    Try RDKit first, fall back to pure-numpy valence check.

    Args:
        positions: Atomic positions [n_atoms, 3] in Angstroms.
        atom_type_indices: Atom type indices [n_atoms] into QM9 atom list.

    Returns:
        Tuple of (is_valid, identifier) where identifier is SMILES or formula.
    """
    try:
        return _rdkit_assess(positions, atom_type_indices)
    except ImportError:
        return _numpy_assess(positions, atom_type_indices)


def _rdkit_assess(
    positions: np.ndarray, atom_type_indices: np.ndarray
) -> tuple[bool, str | None]:
    """Use RDKit to assess validity and extract SMILES for identified molecule.

    Args:
        positions: Atomic positions [n_atoms, 3].
        atom_type_indices: Atom type indices [n_atoms].

    Returns:
        Tuple of (is_valid, smiles).
    """
    atom_names = [_ATOM_NAMES[int(i)] for i in atom_type_indices]
    atomic_nums = [_ATOMIC_NUMS[int(i)] for i in atom_type_indices]
    n = len(atom_names)

    mol = RWMol()
    for anum in atomic_nums:
        mol.AddAtom(Chem.Atom(int(anum)))

    for i in range(n):
        for j in range(i):
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
    """Fallback validity check using distance-based bonding and max-valence rules.

    Args:
        positions: Atomic positions [n_atoms, 3].
        atom_type_indices: Atom type indices [n_atoms].

    Returns:
        Tuple of (is_valid, formula).
    """
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

    counts = Counter(int(t) for t in atom_type_indices)
    formula = "".join(f"{_ATOM_NAMES[k]}{v}" for k, v in sorted(counts.items()))
    return True, formula
