from collections import Counter, defaultdict

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Conformer, RWMol
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.rdBase import BlockLogs


# QM9 ordering: H, C, N, O, F
ATOM_ENCODER = {
    0: 1,
    1: 6,
    2: 7,
    3: 8,
    4: 9,
}

ATOM_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}

TARGET_VALENCE = {
    1: 1,
    6: 4,
    7: 3,
    8: 2,
    9: 1,
}


BOND_MARGIN = 0.05


# Simple QM9-style bond-order thresholds in Å.
# Stability only. Do not use this as RDKit validity.
BOND_THRESHOLDS = {
    (1, 1): [(1, 0.74)],
    (1, 6): [(1, 1.20)],
    (1, 7): [(1, 1.15)],
    (1, 8): [(1, 1.10)],
    (1, 9): [(1, 1.05)],

    (6, 6): [(3, 1.20), (2, 1.34), (1, 1.54)],
    (6, 7): [(3, 1.16), (2, 1.30), (1, 1.47)],
    (6, 8): [(2, 1.23), (1, 1.43)],
    (6, 9): [(1, 1.35)],

    (7, 7): [(3, 1.10), (2, 1.25), (1, 1.45)],
    (7, 8): [(2, 1.25), (1, 1.40)],
    (7, 9): [(1, 1.30)],

    (8, 8): [(2, 1.21), (1, 1.48)],
    (8, 9): [(1, 1.25)],

    (9, 9): [(1, 1.42)],
}


def compute_batch_rdkit_validity(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
    charge: int = 0,
) -> dict:
    smiles = []
    invalid_idxs = []

    n_graphs = num_graphs(batch_vec)

    for graph_idx in range(n_graphs):
        positions, type_indices = get_molecule_tensors(
            pos, atom_types, batch_vec, graph_idx
        )
        smi = rdkit_smiles_from_geometry(positions, type_indices, charge=charge)

        if smi is None:
            invalid_idxs.append(graph_idx)
        else:
            smiles.append(smi)

    n_valid = len(smiles)

    validity = n_valid / n_graphs if n_graphs > 0 else 0.0
    uniqueness = len(set(smiles)) / n_valid if n_valid > 0 else 0.0

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "valid_unique": validity * uniqueness,
        "smiles": smiles,
        "invalid_idxs": invalid_idxs,
    }


def compute_batch_stability(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
) -> dict:
    n_graphs = num_graphs(batch_vec)

    n_atoms_total = 0
    n_atoms_stable = 0
    n_mols_stable = 0
    unstable_idxs = []

    for graph_idx in range(n_graphs):
        positions, type_indices = get_molecule_tensors(
            pos, atom_types, batch_vec, graph_idx
        )

        atom_stable, mol_stable = molecule_stability(positions, type_indices)

        n_atoms_total += len(atom_stable)
        n_atoms_stable += int(atom_stable.sum())

        if mol_stable:
            n_mols_stable += 1
        else:
            unstable_idxs.append(graph_idx)

    return {
        "atom_stability": n_atoms_stable / n_atoms_total if n_atoms_total > 0 else 0.0,
        "mol_stability": n_mols_stable / n_graphs if n_graphs > 0 else 0.0,
        "unstable_idxs": unstable_idxs,
    }


def compute_heavy_atom_counts(
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
) -> list[int]:
    if atom_types.ndim == 2:
        atom_types = atom_types.argmax(dim=-1)

    counts = []

    for graph_idx in range(num_graphs(batch_vec)):
        mask = batch_vec == graph_idx
        counts.append(int((atom_types[mask] != 0).sum().item()))

    return counts


def compute_batch_valence_histogram(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
) -> dict:
    hist = defaultdict(Counter)

    for graph_idx in range(num_graphs(batch_vec)):
        positions, type_indices = get_molecule_tensors(
            pos, atom_types, batch_vec, graph_idx
        )
        atomic_numbers, valences = valences_from_geometry(positions, type_indices)

        for atomic_num, valence in zip(atomic_numbers, valences):
            symbol = ATOM_SYMBOLS[int(atomic_num)]
            hist[symbol][int(valence)] += 1

    return {symbol: dict(counts) for symbol, counts in hist.items()}


def compact_batch_vec(batch_vec: torch.Tensor) -> torch.Tensor:
    if batch_vec.numel() == 0:
        return batch_vec

    unique = torch.unique(batch_vec, sorted=True)
    compact = torch.empty_like(batch_vec)

    for new_idx, old_idx in enumerate(unique):
        compact[batch_vec == old_idx] = new_idx

    return compact


def nearest_neighbor_distance_stats(
    pos: torch.Tensor,
    batch_vec: torch.Tensor,
) -> dict:
    nearest = []

    pos, batch_vec = tensors_to_cpu(pos, batch_vec)
    for graph_idx in range(num_graphs(batch_vec)):
        p = pos[batch_vec == graph_idx]
        if p.shape[0] < 2:
            continue

        d = torch.cdist(p, p)
        d.fill_diagonal_(float("inf"))
        nearest.extend(d.min(dim=1).values.tolist())

    return percentile_stats(nearest)


def pairwise_distance_stats(
    pos: torch.Tensor,
    batch_vec: torch.Tensor,
) -> dict:
    distances = []

    pos, batch_vec = tensors_to_cpu(pos, batch_vec)
    for graph_idx in range(num_graphs(batch_vec)):
        p = pos[batch_vec == graph_idx]
        if p.shape[0] < 2:
            continue

        d = torch.pdist(p)
        distances.extend(d.tolist())

    return percentile_stats(distances)


def rdkit_smiles_from_geometry(
    positions: np.ndarray,
    atom_type_indices: np.ndarray,
    charge: int = 0,
) -> str | None:
    positions = np.asarray(positions, dtype=np.float64)
    atomic_numbers = atom_indices_to_atomic_numbers(atom_type_indices)

    if len(atomic_numbers) == 0:
        return None

    with BlockLogs():
        mol = RWMol()
        conf = Conformer(len(atomic_numbers))

        for i, atomic_num in enumerate(atomic_numbers):
            mol.AddAtom(Chem.Atom(int(atomic_num)))
            conf.SetAtomPosition(i, positions[i].tolist())

        mol.AddConformer(conf, assignId=True)

        try:
            DetermineBonds(mol, charge=charge)
        except Exception:
            return None

        if mol.GetNumBonds() == 0:
            return None

        if Chem.SanitizeMol(mol, catchErrors=True):
            return None

        try:
            return Chem.MolToSmiles(mol.GetMol(), canonical=True)
        except Exception:
            return None


def molecule_stability(
    positions: np.ndarray,
    atom_type_indices: np.ndarray,
) -> tuple[np.ndarray, bool]:
    atomic_numbers, valences = valences_from_geometry(positions, atom_type_indices)
    target = np.array([TARGET_VALENCE[int(a)] for a in atomic_numbers], dtype=np.int64)

    atom_stable = valences == target
    mol_stable = bool(atom_stable.all()) if len(atom_stable) > 0 else False

    return atom_stable, mol_stable


def valences_from_geometry(
    positions: np.ndarray,
    atom_type_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    atomic_numbers = atom_indices_to_atomic_numbers(atom_type_indices)

    n_atoms = len(atomic_numbers)
    valences = np.zeros(n_atoms, dtype=np.int64)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(positions[i] - positions[j])
            order = bond_order(atomic_numbers[i], atomic_numbers[j], distance)
            valences[i] += order
            valences[j] += order

    return atomic_numbers, valences


def bond_order(
    atomic_num_i: int,
    atomic_num_j: int,
    distance: float,
) -> int:
    pair = tuple(sorted((int(atomic_num_i), int(atomic_num_j))))

    for order, max_distance in BOND_THRESHOLDS.get(pair, []):
        if distance <= max_distance + BOND_MARGIN:
            return order

    return 0


def atom_indices_to_atomic_numbers(atom_type_indices: np.ndarray) -> np.ndarray:
    atom_type_indices = np.asarray(atom_type_indices)

    if atom_type_indices.ndim == 2:
        atom_type_indices = atom_type_indices.argmax(axis=-1)

    return np.array(
        [ATOM_ENCODER[int(i)] for i in atom_type_indices],
        dtype=np.int64,
    )


def get_molecule_tensors(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
    graph_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = batch_vec == graph_idx

    positions = pos[mask]
    type_indices = atom_types[mask]

    if type_indices.ndim == 2:
        type_indices = type_indices.argmax(dim=-1)

    return (
        positions.detach().cpu().numpy().astype(np.float64),
        type_indices.detach().cpu().numpy().astype(np.int64),
    )


def num_graphs(batch_vec: torch.Tensor) -> int:
    if batch_vec.numel() == 0:
        return 0

    return int(batch_vec.max().item()) + 1


def tensors_to_cpu(
    pos: torch.Tensor,
    batch_vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(pos):
        pos = pos.detach().cpu()
    if torch.is_tensor(batch_vec):
        batch_vec = batch_vec.detach().cpu().long()

    return pos, batch_vec


def percentile_stats(values: list[float]) -> dict:
    if not values:
        return {}

    values_np = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values_np.min()),
        "p01": float(np.percentile(values_np, 1)),
        "p05": float(np.percentile(values_np, 5)),
        "p25": float(np.percentile(values_np, 25)),
        "median": float(np.percentile(values_np, 50)),
        "p75": float(np.percentile(values_np, 75)),
        "p95": float(np.percentile(values_np, 95)),
        "max": float(values_np.max()),
    }
