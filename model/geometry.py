"""Geometry utilities for batched molecular graph data.

This module provides helpers to center coordinates and compute per-graph norms
for batched PyTorch Geometric data objects.
"""

import torch
from torch_geometric.nn import global_mean_pool


def center_positions_per_graph(
    pos: torch.Tensor, batch_vec: torch.Tensor
) -> torch.Tensor:
    """Zero-center coordinates independently for each graph in a batch.

    Args:
        pos: Node positions [total_nodes, 3].
        batch_vec: Batch indices [total_nodes].

    Returns:
        Centered positions [total_nodes, 3].
    """
    if batch_vec is None or batch_vec.numel() == 0:
        return pos - pos.mean(dim=0, keepdim=True)
    com = global_mean_pool(pos, batch_vec)  # [G, 3]
    return pos - com[batch_vec]


def per_graph_center_norms(pos: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
    """Compute per-graph center L2 norms (should be ~0 if centered).

    Args:
        pos: Node positions [total_nodes, 3].
        batch_vec: Batch indices [total_nodes].

    Returns:
        L2 norms of per-graph centers, shape [G].
    """
    com = global_mean_pool(pos, batch_vec)  # [G, 3]
    return com.norm(dim=-1)


def batch_size_for_logging(batch) -> int:
    """Return the effective batch size for display and logging.

    Handles both PyTorch Geometric batch objects and single graph inputs.

    Returns:
        Integer batch size.
    """
    if hasattr(batch, "num_graphs") and batch.num_graphs is not None:
        return int(batch.num_graphs)
    if hasattr(batch, "batch") and batch.batch is not None:
        return int(batch.batch.max().item()) + 1
    return 1
