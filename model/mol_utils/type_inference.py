import numpy as np
import torch

from .constants import (
    _ATOM_NAMES,
    _ATOMIC_NUMS,
    _STABLE_VALENCE,
    _BONDS1,
    _BONDS2,
    _BONDS3,
    _MARGIN1,
    _MARGIN2,
    _MARGIN3,
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

    type_indices = np.where(
        degrees >= 4, 1,
        np.where(degrees == 3, 2,
        np.where(degrees == 2, 3,
        0)),
    )

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


def _build_bo_table(n: int, dists: np.ndarray) -> np.ndarray:
    """Precompute bo_table[i, j, c1, c2]: bond order when atom i has type c1, j has type c2.

    Uses vectorised numpy threshold comparisons over the full n×n distance matrix
    for each of the 25 QM9 type-pair combinations — no per-pair Python loop.
    """
    n_types = 5
    bo_table = np.zeros((n, n, n_types, n_types), dtype=np.int8)
    for c1 in range(n_types):
        for c2 in range(n_types):
            a1, a2 = _ATOM_NAMES[c1], _ATOM_NAMES[c2]
            t1 = _BONDS1.get(a1, {}).get(a2) or _BONDS1.get(a2, {}).get(a1)
            if t1 is None:
                continue
            t2 = _BONDS2.get(a1, {}).get(a2) or _BONDS2.get(a2, {}).get(a1)
            t3 = _BONDS3.get(a1, {}).get(a2) or _BONDS3.get(a2, {}).get(a1)
            slab = bo_table[:, :, c1, c2]
            slab[dists < (t1 + _MARGIN1) / 100.0] = 1
            if t2 is not None:
                slab[dists < (t2 + _MARGIN2) / 100.0] = 2
            if t3 is not None:
                slab[dists < (t3 + _MARGIN3) / 100.0] = 3
            np.fill_diagonal(slab, 0)
    return bo_table


def _infer_types_stability_guided(positions: np.ndarray) -> np.ndarray:
    """Heuristic seed refined by greedy bond-order stability maximisation.

    Precomputes all possible bond orders into a lookup table, then uses
    vectorised incremental delta updates — no per-pair loops in the refinement.
    """
    types = _infer_types_heuristic(positions).copy()
    n = len(types)
    if n == 0:
        return types

    dists = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    bo_table = _build_bo_table(n, dists)

    stable_vals = np.array([_STABLE_VALENCE[_ATOMIC_NUMS[c]] for c in range(5)])
    idx = np.arange(n)

    # counts[i] = sum_j bo_table[i, j, types[i], types[j]]
    counts = bo_table[idx[:, None], idx[None, :], types[:, None], types[None, :]].sum(axis=1)

    for _ in range(n):
        targets = stable_vals[types]
        stable_mask = counts == targets
        if stable_mask.all():
            break

        best_gain, best_i, best_c = 0, -1, -1
        n_curr = int(stable_mask.sum())

        for i in np.where(~stable_mask)[0]:
            for c in range(5):
                if c == types[i]:
                    continue
                # O(n) vectorised delta: bond count changes when atom i flips to type c
                delta = (bo_table[i, idx, c, types] - bo_table[i, idx, types[i], types]).astype(int)
                delta[i] = 0  # no self-bond
                trial_counts = counts + delta
                trial_counts[i] += delta.sum()
                trial_targets = targets.copy()
                trial_targets[i] = stable_vals[c]
                gain = int((trial_counts == trial_targets).sum()) - n_curr
                if gain > best_gain:
                    best_gain, best_i, best_c = gain, i, c

        if best_i == -1:
            break

        # Apply the best flip and update counts incrementally
        delta = (bo_table[best_i, idx, best_c, types] - bo_table[best_i, idx, types[best_i], types]).astype(int)
        delta[best_i] = 0
        counts += delta
        counts[best_i] += delta.sum()
        types[best_i] = best_c

    return types


_INFER_FNS = {
    "degree": _infer_types_from_degree,
    "heuristic": _infer_types_heuristic,
    "stability": _infer_types_stability_guided,
}
