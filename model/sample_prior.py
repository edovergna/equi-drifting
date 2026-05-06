import torch


def sample_coordinate_noise(
    total_nodes: int,
    clamp_range: float,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """
    Sample flat 3D coordinate noise for all nodes.

    Returns:
        pos: [total_nodes, 3]
    """
    pos = torch.randn(total_nodes, 3, device=device, dtype=dtype)
    return pos.clamp(min=-float(clamp_range), max=float(clamp_range))


def sample_atom_dirichlet_noise(
    total_nodes: int,
    num_atom_types: int = 5,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample flat simplex-valued atom probabilities and their square-root map.

    Returns:
        A_prob: [total_nodes, num_atom_types]
        S_sqrt: [total_nodes, num_atom_types]
    """
    alpha_atom = torch.ones(num_atom_types, device=device, dtype=dtype)
    dist = torch.distributions.Dirichlet(alpha_atom)
    A_prob = dist.sample((total_nodes,))
    S_sqrt = torch.sqrt(A_prob.clamp_min(1e-12))
    return A_prob, S_sqrt


def batch_vector_from_node_counts(node_counts, device=None) -> torch.Tensor:
    """
    node_counts: list/1D tensor of length B with number of real nodes per graph.

    Returns:
        batch: [sum(node_counts)], PyG graph id per node.
    """
    node_counts = torch.as_tensor(node_counts, device=device, dtype=torch.long)
    if node_counts.dim() != 1:
        raise ValueError("node_counts must be a 1D list or tensor")
    if node_counts.numel() == 0:
        raise ValueError("node_counts must contain at least one graph")
    if torch.any(node_counts <= 0):
        raise ValueError("all node counts must be positive")

    return torch.repeat_interleave(
        torch.arange(node_counts.numel(), device=node_counts.device),
        node_counts,
    )


def sample_egnn_molecule_batch(
    node_counts,
    clamp_range: float,
    num_atom_types: int = 5,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> dict[str, torch.Tensor]:
    """
    Sample flat molecule-shaped prior noise for the EGNN.

    node_counts: list/1D tensor of length B. node_counts[b] is the number of
        nodes in graph b.

    Returns:
        x: [sum(node_counts), num_atom_types]
        pos: [sum(node_counts), 3]
        batch: [sum(node_counts)]
    """
    node_counts = torch.as_tensor(node_counts, device=device, dtype=torch.long)
    batch = batch_vector_from_node_counts(node_counts, device=device)
    total_nodes = int(node_counts.sum().item())

    pos = sample_coordinate_noise(
        total_nodes=total_nodes,
        clamp_range=clamp_range,
        dtype=dtype,
        device=device,
    )
    A_prob, S_sqrt = sample_atom_dirichlet_noise(
        total_nodes=total_nodes,
        num_atom_types=num_atom_types,
        dtype=dtype,
        device=device,
    )

    return {
        "x": S_sqrt,
        "pos": pos,
        "batch": batch,
        "A_prob": A_prob,
        "S_sqrt": S_sqrt,
    }


def check_simplex(A_prob, atol=1e-5):
    row_sums = A_prob.sum(dim=-1)
    return torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol)


def check_sqrt_simplex(S_sqrt, atol=1e-5):
    row_sums = (S_sqrt**2).sum(dim=-1)
    return torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol)
