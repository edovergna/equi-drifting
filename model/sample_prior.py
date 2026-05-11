import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Dense edge-index cache
# ---------------------------------------------------------------------------

_dense_edge_index_cache: dict[int, torch.Tensor] = {}


def get_dense_edge_index(n: int, device: torch.device) -> torch.Tensor:
    """Return a cached fully-connected no-self-loop edge index for n nodes."""
    if n not in _dense_edge_index_cache:
        row = torch.arange(n).repeat_interleave(n)
        col = torch.arange(n).repeat(n)
        mask = row != col
        _dense_edge_index_cache[n] = torch.stack([row[mask], col[mask]], dim=0)
    return _dense_edge_index_cache[n].to(device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_underlying_dataset(dataset):
    """Handle torch_geometric Subset-style wrappers."""
    return dataset.dataset if hasattr(dataset, "dataset") else dataset


def _get_data_item(dataset, idx: int):
    """
    Supports either a plain dataset or a torch.utils.data.Subset-like wrapper.

    If `dataset` is a Subset, dataset[i] already returns the correct item.
    """
    return dataset[idx]


def _atom_types_from_x(
    x: torch.Tensor,
    num_atom_types: int | None = None,
) -> torch.Tensor:
    """
    Convert node features to categorical atom types.

    Assumes either:
        x: [N, num_atom_types] one-hot / probabilities
    or:
        x: [N] integer atom types
    """
    if x.ndim == 2:
        if num_atom_types is not None and x.shape[-1] > num_atom_types:
            x = x[:, :num_atom_types]
        return x.argmax(dim=-1).long()
    return x.long()


def _atom_types_from_data(data, num_atom_types: int | None = None) -> torch.Tensor:
    """Prefer explicit transformed QM9 labels over raw feature matrices."""
    if hasattr(data, "real_atom_types"):
        return _atom_types_from_x(data.real_atom_types, num_atom_types)
    return _atom_types_from_x(data.x, num_atom_types)


def radius_of_gyration(pos: torch.Tensor) -> torch.Tensor:
    """Compute radius of gyration for one molecule."""
    pos = pos.float()
    pos = pos - pos.mean(dim=0, keepdim=True)
    return torch.sqrt(pos.pow(2).sum(dim=-1).mean().clamp_min(1e-12))


# ---------------------------------------------------------------------------
# Empirical QM9 size distribution
# ---------------------------------------------------------------------------

def compute_size_distribution(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical atom-count distribution over the dataset."""
    underlying = _get_underlying_dataset(dataset)

    # Fast path for full PyG InMemoryDataset. For Subset-like wrappers, iterate
    # through the wrapper so the distribution matches the selected split.
    if (
        underlying is dataset
        and hasattr(underlying, "slices")
        and "pos" in underlying.slices
    ):
        sizes = torch.diff(underlying.slices["pos"])
    else:
        sizes = torch.tensor(
            [_get_data_item(dataset, i).num_nodes for i in range(len(dataset))]
        )

    counts = torch.bincount(sizes.long())
    mask = counts > 0

    unique = torch.where(mask)[0].cpu().numpy().astype(int)
    probs = (counts[mask].float() / counts[mask].sum()).cpu().numpy()

    return unique, probs


# ---------------------------------------------------------------------------
# Empirical QM9 composition distribution
# ---------------------------------------------------------------------------

def compute_composition_distribution(
    dataset,
    num_atom_types: int = 5,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """
    Compute empirical atom-composition distribution.

    Each composition is a tuple of length `num_atom_types`, for example:

        (n_H, n_C, n_N, n_O, n_F)

    depending on your atom-type ordering in `data.x`.

    Returns:
        compositions: list of tuples
        probs:        empirical probabilities over those tuples
    """
    counter = Counter()

    for i in range(len(dataset)):
        data = _get_data_item(dataset, i)
        atom_types = _atom_types_from_data(data, num_atom_types)

        counts = torch.bincount(atom_types, minlength=num_atom_types)
        comp = tuple(int(c) for c in counts[:num_atom_types].cpu().tolist())

        counter[comp] += 1

    compositions = list(counter.keys())
    probs = np.array([counter[c] for c in compositions], dtype=np.float64)
    probs = probs / probs.sum()

    return compositions, probs


def compute_rg_by_composition(
    dataset,
    num_atom_types: int = 5,
) -> dict[tuple[int, ...], np.ndarray]:
    """
    Compute empirical radius-of-gyration samples per composition.

    Used to scale Gaussian position noise so that generated prior samples have
    roughly QM9-like spatial extent for their composition.
    """
    rg_by_comp = defaultdict(list)

    for i in range(len(dataset)):
        data = _get_data_item(dataset, i)
        atom_types = _atom_types_from_data(data, num_atom_types)

        counts = torch.bincount(atom_types, minlength=num_atom_types)
        comp = tuple(int(c) for c in counts[:num_atom_types].cpu().tolist())

        rg = radius_of_gyration(data.pos).item()
        rg_by_comp[comp].append(rg)

    return {
        comp: np.asarray(values, dtype=np.float32)
        for comp, values in rg_by_comp.items()
    }


def compute_rg_by_size(dataset) -> dict[int, np.ndarray]:
    """
    Fallback empirical radius-of-gyration samples per atom count.
    Useful when a sampled composition has no specific Rg statistics.
    """
    rg_by_size = defaultdict(list)

    for i in range(len(dataset)):
        data = _get_data_item(dataset, i)
        n = int(data.num_nodes)
        rg = radius_of_gyration(data.pos).item()
        rg_by_size[n].append(rg)

    return {
        n: np.asarray(values, dtype=np.float32)
        for n, values in rg_by_size.items()
    }


# ---------------------------------------------------------------------------
# Atom-feature samplers
# ---------------------------------------------------------------------------

def sample_atom_dirichlet_noise(
    total_nodes: int,
    num_atom_types: int = 5,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Original unconstrained flat Dirichlet atom sampler.

    Kept for ablations / compatibility.
    """
    alpha_atom = torch.ones(num_atom_types, device=device, dtype=dtype)
    dist = torch.distributions.Dirichlet(alpha_atom)
    A_prob = dist.sample((total_nodes,))
    S_sqrt = torch.sqrt(A_prob.clamp_min(1e-12))
    return A_prob, S_sqrt


def sample_atom_features_from_compositions(
    compositions: list[tuple[int, ...]],
    num_atom_types: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    dirichlet_strength: float | None = None,
    dirichlet_low: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build atom features from exact empirical compositions.

    If dirichlet_strength is None:
        returns exact one-hot atom features.

    If dirichlet_strength is set:
        samples soft Dirichlet features centered around the chosen atom type.
        Example: dirichlet_strength=50.0 gives sharp atom probabilities.

    Returns:
        x:          [total_nodes, num_atom_types]
        atom_types: [total_nodes]
    """
    atom_type_chunks = []

    for comp in compositions:
        types = []
        for atom_type, count in enumerate(comp):
            types.extend([atom_type] * int(count))

        types = torch.tensor(types, dtype=torch.long, device=device)

        # Randomize node order within each molecule.
        if types.numel() > 0:
            types = types[torch.randperm(types.numel(), device=device)]

        atom_type_chunks.append(types)

    atom_types = torch.cat(atom_type_chunks, dim=0)
    total_nodes = int(atom_types.numel())

    if dirichlet_strength is None:
        x = F.one_hot(atom_types, num_classes=num_atom_types).to(dtype=dtype)
        return x, atom_types

    alpha = torch.full(
        (total_nodes, num_atom_types),
        float(dirichlet_low),
        device=device,
        dtype=dtype,
    )
    alpha[torch.arange(total_nodes, device=device), atom_types] = float(
        dirichlet_strength
    )

    x = torch.distributions.Dirichlet(alpha).sample()
    return x, atom_types


# ---------------------------------------------------------------------------
# Position samplers
# ---------------------------------------------------------------------------

def sample_centered_scaled_gaussian_positions(
    compositions: list[tuple[int, ...]],
    rg_by_composition: dict[tuple[int, ...], np.ndarray] | None,
    rg_by_size: dict[int, np.ndarray] | None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    prior_pos_clamp: float | None = None,
    default_rg: float = 1.0,
) -> torch.Tensor:
    """
    Sample per-molecule centered Gaussian positions and scale them using
    empirical radius of gyration.

    Priority:
        1. empirical Rg for exact composition
        2. empirical Rg for same atom count
        3. default_rg
    """
    pos_chunks = []

    for comp in compositions:
        n = int(sum(comp))

        if rg_by_composition is not None and comp in rg_by_composition:
            rg = float(np.random.choice(rg_by_composition[comp]))
        elif rg_by_size is not None and n in rg_by_size:
            rg = float(np.random.choice(rg_by_size[n]))
        else:
            rg = float(default_rg)

        z = torch.randn(n, 3, device=device, dtype=dtype)
        z = z - z.mean(dim=0, keepdim=True)

        z_rg = torch.sqrt(z.pow(2).sum(dim=-1).mean().clamp_min(1e-12))
        z = z / z_rg * rg

        if prior_pos_clamp is not None:
            z = z.clamp(-prior_pos_clamp, prior_pos_clamp)

        pos_chunks.append(z)

    return torch.cat(pos_chunks, dim=0)


# ---------------------------------------------------------------------------
# Batch samplers
# ---------------------------------------------------------------------------

def sample_prior_batch(
    n_molecules: int,
    size_values: np.ndarray,
    size_probs: np.ndarray,
    num_atom_types: int,
    prior_pos_clamp: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Original size-only prior sampler.

    Kept for compatibility / ablations.

    This is NOT composition-constrained.
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


def sample_composition_prior_batch(
    n_molecules: int,
    compositions: list[tuple[int, ...]],
    composition_probs: np.ndarray,
    num_atom_types: int,
    device: torch.device,
    prior_pos_clamp: float | None = None,
    rg_by_composition: dict[tuple[int, ...], np.ndarray] | None = None,
    rg_by_size: dict[int, np.ndarray] | None = None,
    dirichlet_strength: float | None = None,
    dirichlet_low: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    list[tuple[int, ...]],
    torch.Tensor,
]:
    """
    Composition-constrained QM9 prior sampler.

    Steps:
        1. Sample empirical QM9 atom compositions.
        2. Build atom features with exactly those compositions.
        3. Sample centered Gaussian positions scaled by empirical Rg.
        4. Build dense fully-connected no-self-loop edges.

    Args:
        n_molecules:
            Number of molecules in the batch.

        compositions:
            List of empirical composition tuples from `compute_composition_distribution`.

        composition_probs:
            Empirical probability for each composition.

        num_atom_types:
            Number of atom types in x.

        device:
            Torch device.

        prior_pos_clamp:
            Optional clamp for positions. Set None to disable.

        rg_by_composition:
            Optional dict from `compute_rg_by_composition`.

        rg_by_size:
            Optional fallback dict from `compute_rg_by_size`.

        dirichlet_strength:
            If None, atom features are exact one-hot.
            If float, atom features are soft Dirichlet samples centered around
            their chosen atom type.

            Suggested first values:
                None   -> exact one-hot
                50.0   -> sharp soft atom features
                100.0  -> very sharp soft atom features

        dirichlet_low:
            Concentration for non-selected atom types when using soft Dirichlet.

    Returns:
        x:                     [total_nodes, num_atom_types]
        pos:                   [total_nodes, 3]
        batch_vec:             [total_nodes]
        dense_edge_index:      [2, total_edges]
        atom_counts:           [n_molecules] numpy array
        sampled_compositions:  list of composition tuples
        atom_types:            [total_nodes] categorical atom type IDs
    """
    comp_indices = np.random.choice(
        len(compositions),
        size=n_molecules,
        p=composition_probs,
    )

    sampled_compositions = [compositions[int(i)] for i in comp_indices]
    atom_counts = np.asarray(
        [sum(comp) for comp in sampled_compositions],
        dtype=np.int64,
    )

    x, atom_types = sample_atom_features_from_compositions(
        sampled_compositions,
        num_atom_types=num_atom_types,
        device=device,
        dtype=dtype,
        dirichlet_strength=dirichlet_strength,
        dirichlet_low=dirichlet_low,
    )

    pos = sample_centered_scaled_gaussian_positions(
        sampled_compositions,
        rg_by_composition=rg_by_composition,
        rg_by_size=rg_by_size,
        device=device,
        dtype=dtype,
        prior_pos_clamp=prior_pos_clamp,
    )

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

    return (
        x,
        pos,
        batch_vec,
        dense_edge_index,
        atom_counts,
        sampled_compositions,
        atom_types,
    )
