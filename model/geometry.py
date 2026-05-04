import torch
from torch_geometric.nn import global_mean_pool


def center_positions_per_graph(
    pos: torch.Tensor, batch_vec: torch.Tensor
) -> torch.Tensor:
    """Zero-center coordinates independently for each graph in a batch."""
    if batch_vec is None or batch_vec.numel() == 0:
        return pos - pos.mean(dim=0, keepdim=True)
    com = global_mean_pool(pos, batch_vec)  # [G, 3]
    return pos - com[batch_vec]


def per_graph_center_norms(
    pos: torch.Tensor, batch_vec: torch.Tensor
) -> torch.Tensor:
    """Returns a [G] tensor of per-graph center L2 norms. Should be ~0 if centered."""
    com = global_mean_pool(pos, batch_vec)  # [G, 3]
    return com.norm(dim=-1)


def batch_size_for_logging(batch) -> int:
    if hasattr(batch, "num_graphs") and batch.num_graphs is not None:
        return int(batch.num_graphs)
    if hasattr(batch, "batch") and batch.batch is not None:
        return int(batch.batch.max().item()) + 1
    return 1
