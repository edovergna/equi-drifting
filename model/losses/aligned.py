from typing import Tuple

import torch
import torch.nn.functional as F
import wandb

from . import TrainingDivergedException
from ..spherical_utils import product_tangent_norm, sphere_exp, geodesic_distance


def compute_aligning_drift_loss(
    pos_gen: torch.Tensor,
    pos_real: torch.Tensor,
    x_gen_sphere: torch.Tensor,
    x_real: torch.Tensor,
    gen_batch_vec: torch.Tensor,
    real_batch_vec: torch.Tensor,
    temperatures: tuple[float, ...] = (0.02, 0.05, 0.02),
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Drifting field loss directly on 3D molecules by aligning the generated molecules with the real ones.
    Assumes that x_gen_sphere is already mapped to the spherical space

    Returns (loss, stats) where stats is a flat dict of float diagnostics safe to
    pass directly to self.log(). Raises TrainingDivergedException on non-finite loss.
    """

    max_nodes = max(
        _max_nodes_per_graph(gen_batch_vec),
        _max_nodes_per_graph(real_batch_vec),
    )

    mol_pos_gen, mol_x_gen_sphere, gen_mask = _molecules_to_padded(pos_gen, x_gen_sphere, gen_batch_vec, max_nodes)
    mol_pos_real, mol_x_real, real_mask = _molecules_to_padded(pos_real, x_real, real_batch_vec, max_nodes)

    # TODO: add aligning here

    # Assume for now that the molecules are aligned, so that the atoms are ordered in a way that they correspond
    # and the 3D positions are rotated/reflected accordingly.

    N_gen = mol_pos_gen.shape[0]           # number of generated molecules
    N_real = mol_pos_real.shape[0]         # number of real molecules

    old_pos_gen = mol_pos_gen.detach()
    old_x_gen_sphere = mol_x_gen_sphere.detach()

    # Obtain actual pairs mask
    pair_mask_pos = gen_mask[:, None, :] * real_mask[None, :, :]                # shape: (N_gen, N_real, max_nodes)
    pair_mask_neg = gen_mask[:, None, :] * gen_mask[None, :, :]                 # shape: (N_gen, N_gen, max_nodes)

    # Calculate pairwise distances for positions of molecules
    mol_dist_pos = _pairwise_geodesic_distance(mol_pos_gen, mol_pos_real, pair_mask_pos, "euclidean")  # shape: (N_gen, N_real)
    mol_dist_neg = _pairwise_geodesic_distance(mol_pos_gen, old_pos_gen, pair_mask_neg, "euclidean")   # shape: (N_gen, N_real)

    # Mask self connections in negative with high value
    mol_dist_neg.fill_diagonal_(1e8)

    return None


def _pairwise_geodesic_distance(
        x: torch.Tensor,
        y: torch.Tensor,
        pair_mask: torch.Tensor,
        manifold: str = "euclidean"
) -> torch.Tensor:
    """
    Calculates the pairwise distance between atoms for the molecule attributes depending on 
    the specified geodesic distance. Assumed that molecules are aligned.
    Args:
        x: [N_x, max_nodes, z]
        y: [N_y, max_nodes, z]
        pair_mask: [N_x, N_y, max_nodes]
    """

    if manifold == "euclidean":
        diff = x[:, None, :, :] - y[None, :, :, :]                              # shape: (N_x, N_y, max_nodes, z)
        sq_dist_per_atom = (diff ** 2).sum(dim=-1)                              # shape: (N_x, N_y, max_nodes)
        masked_sq_dist = sq_dist_per_atom * pair_mask   

        # Scale by num of atoms shared
        num_atoms_shared = pair_mask.sum(dim=-1).clamp_min(1.0)                 # shape: (N_x, N_y)
        distances = torch.sqrt(masked_sq_dist.sum(dim=-1) / num_atoms_shared)   # shape: (N_x, N_y)
    elif manifold == "spherical":
        distances = ...
    else:
        raise ValueError("Undefined geodesic distance queried.")
    return distances


def _molecules_to_padded(
    pos: torch.Tensor,
    atom_types: torch.Tensor,
    batch_vec: torch.Tensor,
    max_nodes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    graph_ids = torch.unique(batch_vec, sorted=True)
    counts = torch.stack([(batch_vec == graph_id).sum() for graph_id in graph_ids])
    if max_nodes is None:
        max_nodes = int(counts.max().item())

    padded_pos = pos.new_zeros((graph_ids.numel(), max_nodes, pos.shape[-1]))
    padded_atom_types = atom_types.new_zeros(
        (graph_ids.numel(), max_nodes, *atom_types.shape[1:])
    )
    mask = torch.zeros(
        (graph_ids.numel(), max_nodes), device=pos.device
    )

    for i, graph_id in enumerate(graph_ids):
        graph_mask = batch_vec == graph_id

        graph_pos = pos[graph_mask]
        graph_atom_types = atom_types[graph_mask]

        n = min(graph_pos.shape[0], max_nodes)
        padded_pos[i, :n] = graph_pos[:n]
        padded_atom_types[i, :n] = graph_atom_types[:n]
        mask[i, :n] = 1.0

    return padded_pos, padded_atom_types, mask


def _max_nodes_per_graph(batch_vec: torch.Tensor) -> int:
    graph_ids = torch.unique(batch_vec, sorted=True)
    counts = torch.stack([(batch_vec == graph_id).sum() for graph_id in graph_ids])
    return int(counts.max().item())
