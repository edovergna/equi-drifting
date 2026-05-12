import numpy as np
import torch

from .bonds import get_bond_order
from .constants import (
    _ATOM_NAMES,
    _ATOMIC_NUMS,
    _STABLE_VALENCE,
    _GENERIC_BOND_THRESHOLD,
    _H_BOND_MAX,
    _CARBONYL_O_MAX,
    _O_AVG_BOND_MAX,
    _N_AVG_BOND_MAX,
)

_H_F_BOND_THRESHOLD = 1.2


def infer_types_from_pos_batch(
    pos: torch.Tensor,
    batch_vec: torch.Tensor,
    device: torch.device,
    num_atom_types: int = 5,
    method: str = "heuristic",
) -> torch.Tensor:
    """
    Infer QM9 atom types from generated positions.

    method:
      "degree"    — simple connectivity-degree mapping (fast, less accurate)
      "heuristic" — QM9-specific rules: bond lengths + neighborhood chemistry
      "stability" — heuristic seed refined by greedy bond-order stability maximisation

    Returns:
        atom_types: [N, num_atom_types] one-hot float tensor on `device`
    """
    infer_fn = _INFER_FNS[method]
    pos_np = pos.detach().cpu().numpy().astype(np.float64)
    batch_np = batch_vec.detach().cpu().numpy()
    n_total = pos_np.shape[0]
    n_graphs = int(batch_np.max()) + 1

    type_indices = np.zeros(n_total, dtype=np.int64)
    for g in range(n_graphs):
        mask = batch_np == g
        type_indices[mask] = infer_fn(pos_np[mask])

    atom_types = torch.zeros(n_total, num_atom_types)
    atom_types[torch.arange(n_total), torch.from_numpy(type_indices)] = 1.0
    return atom_types.to(device)


def infer_types_single(positions: np.ndarray, method: str = "heuristic") -> np.ndarray:
    """Per-molecule wrapper for use outside the batched training loop (e.g. visualisation)."""
    return _INFER_FNS[method](positions)


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

    type_indices = np.where(degrees >= 2, 1, 0)

    terminal_mask = degrees == 1
    if terminal_mask.any():
        terminal_idx = np.where(terminal_mask)[0]
        for i in terminal_idx:
            neighbor = np.argmin(np.where(adj[i], dists[i], np.inf))
            if dists[i, neighbor] > _H_F_BOND_THRESHOLD:
                type_indices[i] = 4  # F

    return type_indices.astype(np.int64)


def _infer_types_heuristic(positions: np.ndarray) -> np.ndarray:
    """QM9-specific heuristic: bond lengths + local chemistry priors.

    Priority order:
      isolated (deg 0)                              → H
      terminal (deg 1), bond < 1.20 Å              → H
      terminal (deg 1), bond in [1.20,1.30) & deg≥3 neighbour → O (carbonyl)
      terminal (deg 1), otherwise                  → F
      deg 2: avg bond < 1.45 Å                     → O
             avg bond < 1.52 Å                     → N
             else                                  → C
      deg 3: avg bond < 1.52 Å                     → N; else C
      deg ≥ 4                                      → C
    """
    n = len(positions)
    if n == 0:
        return np.array([], dtype=np.int64)

    dists = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    adj = (dists < _GENERIC_BOND_THRESHOLD) & (dists > 0)
    degrees = adj.sum(axis=1)
    types = np.full(n, -1, dtype=np.int64)

    types[degrees == 0] = 0

    for i in np.where(degrees == 1)[0]:
        nbr = int(np.where(adj[i])[0][0])
        d = dists[i, nbr]
        if d < _H_BOND_MAX:
            types[i] = 0
        elif d < _CARBONYL_O_MAX and degrees[nbr] >= 3:
            types[i] = 3
        else:
            types[i] = 4

    for i in np.where((degrees == 2) & (types == -1))[0]:
        avg = dists[i, adj[i]].mean()
        min_bond = dists[i, adj[i]].min()
        if avg < _O_AVG_BOND_MAX:
            # min_bond < 1.25 Å → C=O double bond (~1.20); otherwise imine C=N (~1.29)
            types[i] = 3 if min_bond < 1.25 else 2
        elif avg < _N_AVG_BOND_MAX:
            types[i] = 2
        else:
            types[i] = 1

    for i in np.where((degrees == 3) & (types == -1))[0]:
        avg = dists[i, adj[i]].mean()
        types[i] = 2 if avg < _N_AVG_BOND_MAX else 1

    types[types == -1] = 1
    return types


def _infer_types_stability_guided(positions: np.ndarray) -> np.ndarray:
    """Heuristic seed refined by greedy bond-order stability maximisation.

    Uses the EDM bond-length tables (via get_bond_order) to compute per-atom
    bond totals, then iteratively flips each unstable atom to the candidate
    type that gains the most stable atoms, until fully stable or no improvement.
    """
    types = _infer_types_heuristic(positions).copy()
    n = len(types)
    if n == 0:
        return types

    dists = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)

    def _bond_counts(t):
        counts = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                o = get_bond_order(_ATOM_NAMES[t[i]], _ATOM_NAMES[t[j]], dists[i, j])
                counts[i] += o
                counts[j] += o
        return counts

    def _n_stable(t, counts):
        return sum(counts[i] == _STABLE_VALENCE[_ATOMIC_NUMS[t[i]]] for i in range(n))

    for _ in range(n):
        counts = _bond_counts(types)
        n_stable = _n_stable(types, counts)
        if n_stable == n:
            break

        stable_mask = np.array([
            counts[i] == _STABLE_VALENCE[_ATOMIC_NUMS[types[i]]] for i in range(n)
        ])
        best_gain, best_i, best_c = 0, -1, -1

        for i in np.where(~stable_mask)[0]:
            for c in range(5):
                if c == types[i]:
                    continue
                trial = types.copy()
                trial[i] = c
                gain = _n_stable(trial, _bond_counts(trial)) - n_stable
                if gain > best_gain:
                    best_gain, best_i, best_c = gain, i, c

        if best_i == -1:
            break
        types[best_i] = best_c

    return types


_INFER_FNS = {
    "degree": _infer_types_from_degree,
    "heuristic": _infer_types_heuristic,
    "stability": _infer_types_stability_guided,
}
