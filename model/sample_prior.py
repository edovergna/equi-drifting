import numpy as np
import torch

# ---------------------------------------------------------------------------
# Dense edge-index cache
# ---------------------------------------------------------------------------

_dense_edge_index_cache: dict[int, torch.Tensor] = {}


def get_dense_edge_index(n: int, device: torch.device) -> torch.Tensor:
    """Return a cached fully-connected (no self-loops) edge index for n nodes."""
    if n not in _dense_edge_index_cache:
        row = torch.arange(n).repeat_interleave(n)
        col = torch.arange(n).repeat(n)
        mask = row != col
        _dense_edge_index_cache[n] = torch.stack([row[mask], col[mask]], dim=0)
    return _dense_edge_index_cache[n].to(device)


# ---------------------------------------------------------------------------
# QM9 atom-count distribution
# ---------------------------------------------------------------------------


def compute_size_distribution(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical atom-count distribution over the full underlying QM9 dataset."""
    underlying = dataset.dataset if hasattr(dataset, "dataset") else dataset

    if hasattr(underlying, "slices") and "pos" in underlying.slices:
        sizes = torch.diff(underlying.slices["pos"])  # [n_molecules]
    else:
        sizes = torch.tensor([underlying[i].num_nodes for i in range(len(underlying))])

    counts = torch.bincount(sizes.long())
    mask = counts > 0
    unique = torch.where(mask)[0].numpy().astype(int)
    probs = (counts[mask].float() / counts[mask].sum()).numpy()
    return unique, probs


# ---------------------------------------------------------------------------
# Node-feature samplers
# ---------------------------------------------------------------------------


def sample_atom_dirichlet_noise(
    total_nodes: int,
    num_atom_types: int = 5,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample flat simplex-valued atom probabilities and their square-root map.

    Returns:
        A_prob: [total_nodes, num_atom_types]
        S_sqrt: [total_nodes, num_atom_types]
    """
    alpha_atom = torch.ones(num_atom_types, device=device, dtype=dtype)
    dist = torch.distributions.Dirichlet(alpha_atom)
    A_prob = dist.sample((total_nodes,))
    S_sqrt = torch.sqrt(A_prob.clamp_min(1e-12))
    return A_prob, S_sqrt


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------


def sample_prior_batch(
    n_molecules: int,
    size_values: np.ndarray,
    size_probs: np.ndarray,
    num_atom_types: int,
    prior_pos_clamp: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Sample n_molecules from the prior using the QM9 atom-count distribution.

    1. Draws atom counts for each molecule from the empirical QM9 distribution.
    2. Samples Gaussian positions (clamped) and Dirichlet atom features per node.
    3. Builds the batched dense_edge_index (fully-connected, no self-loops, offset
       per molecule) and the batch membership vector.

    Returns:
        x:                 [total_nodes, num_atom_types]
        pos:               [total_nodes, 3]
        batch_vec:         [total_nodes]  — molecule index per node
        dense_edge_index:  [2, total_edges]
        atom_counts:       [n_molecules]  — numpy array of per-molecule sizes
    """
    atom_counts = np.random.choice(size_values, size=n_molecules, p=size_probs)
    total_nodes = int(atom_counts.sum())

    pos = torch.randn(total_nodes, 3, device=device).clamp(
        -prior_pos_clamp, prior_pos_clamp
    )
    x = sample_atom_dirichlet_noise(total_nodes, num_atom_types, device=device)[0]

    batch_vec = torch.repeat_interleave(
        torch.arange(n_molecules, device=device),
        torch.tensor(atom_counts, dtype=torch.long, device=device),
    )

    parts, offset = [], 0
    for n in atom_counts:
        n = int(n)
        parts.append(get_dense_edge_index(n, device) + offset)
        offset += n
    dense_edge_index = torch.cat(parts, dim=1)

    return x, pos, batch_vec, dense_edge_index, atom_counts
