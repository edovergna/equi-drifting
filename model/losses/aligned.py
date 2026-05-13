import torch
import torch.nn.functional as F

from . import TrainingDivergedException
from ..spherical_utils import product_tangent_norm, sphere_exp, geodesic_distance, sphere_normalize, sphere_project_tangent


def compute_aligning_drift_loss(
    pos_gen: torch.Tensor,
    pos_real: torch.Tensor,
    x_gen_sphere: torch.Tensor,
    x_real: torch.Tensor,
    gen_batch_vec: torch.Tensor,
    real_batch_vec: torch.Tensor,
    temperatures: tuple[float, ...] = (0.02, 0.05, 0.02),
    eps: float = 1e-8
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Drifting field loss directly on 3D molecules by aligning the generated molecules with the real ones.
    Assumes that x_gen_sphere is already mapped to the spherical space

    Returns (loss, stats) where stats is a flat dict of float diagnostics safe to
    pass directly to self.log(). Raises TrainingDivergedException on non-finite loss.
    """
    # TO ASK: currently when there is an atom to many, which is not paired, I just ignore it so that it does not 
    # play a role in the loss. How should we incorporate this? -> conditional on number of atoms

    x_real = x_real.float()

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
    posit_dist_pos, posit_diff_pos = _pairwise_geodesic_distance(mol_pos_gen, mol_pos_real, pair_mask_pos, "euclidean")  
    posit_dist_neg, posit_diff_neg = _pairwise_geodesic_distance(mol_pos_gen, old_pos_gen, pair_mask_neg, "euclidean")   

    # Calculate pairwise distances for types of molecules in spherical space
    types_dist_pos, types_diff_pos = _pairwise_geodesic_distance(mol_x_gen_sphere, mol_x_real, pair_mask_pos, "spherical")
    types_dist_neg, types_diff_neg = _pairwise_geodesic_distance(mol_x_gen_sphere, old_x_gen_sphere, pair_mask_neg, "spherical")    
    # TO ASK: should we rescale these distances

    # Mask self connections in negative with high value
    posit_dist_neg.fill_diagonal_(1e8)
    types_dist_neg.fill_diagonal_(1e8)

    v_pos_across_tau = torch.zeros_like(old_pos_gen)
    v_type_across_tau = torch.zeros_like(old_x_gen_sphere)

    for tau in temperatures:
        v_posit_tau = _compute_V_at_temp(tau, posit_dist_pos, posit_dist_neg, posit_diff_pos, posit_diff_neg)
        v_types_tau = _compute_V_at_temp(tau, types_dist_pos, types_dist_neg, types_diff_pos, types_diff_neg)

        # Rescale per temp
        lambda_posit_t = v_posit_tau.pow(2).mean().sqrt().clamp_min(eps)
        v_posit_tau = v_posit_tau / lambda_posit_t
        
        lambda_types_t = v_types_tau.pow(2).mean().sqrt().clamp_min(eps)
        v_types_tau = v_types_tau / lambda_types_t

        v_pos_across_tau += v_posit_tau
        v_type_across_tau += v_types_tau

    # TODO: add rescaling of V, PLUS ASK IF NECESSARY

    v_type_across_tau = sphere_project_tangent(old_x_gen_sphere, v_type_across_tau)

    # Calculate target for positions
    target_positions = (old_pos_gen + v_pos_across_tau).detach()            # shape [N_mol, max_atoms, 3]

    # Calculate target for atom types via the exponential mapping of the spherical space
    target_types = sphere_exp(old_x_gen_sphere, v_type_across_tau, eps).detach()     # shape: [N_mol, max_atoms, 5]

    # Distance metric for the euclidean space
    molecule_position_dist = ((mol_pos_gen - target_positions).pow(2).sum(dim=-1) * gen_mask).sum(dim=-1) / gen_mask.sum(dim=-1).clamp_min(1.0)

    # Geodesic distances for atom types on the sphere
    molecule_types_dist = (geodesic_distance(mol_x_gen_sphere, target_types, eps).pow(2) * gen_mask).sum(dim=-1) / gen_mask.sum(dim=-1).clamp_min(1.0)

    # Combine the distances per molecule, then loss is average distance
    combined_dist = molecule_position_dist + molecule_types_dist
    loss = combined_dist.mean()

    if not torch.isfinite(loss):
        raise TrainingDivergedException(f"Non-finite loss ({loss.item()!r}). ")
    # TODO: check if rescaling combined distance is needed
    stats: dict[str, float] = {}
    with torch.no_grad():
        stats["average_position_distance"] = molecule_position_dist.mean().item()
        stats["average_types_distance"] = molecule_types_dist.mean().item()
        stats["std_position_distance"] = molecule_position_dist.std().item()
        stats["std_types_distance"] = molecule_types_dist.std().item()

    return loss, stats


def _pairwise_geodesic_distance(
        x: torch.Tensor,
        y: torch.Tensor,
        pair_mask: torch.Tensor,
        manifold: str = "euclidean",
        eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        x = sphere_normalize(x, eps)
        y = sphere_normalize(y, eps)

        dot = torch.einsum("blc,nlc->bnl", x, y).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(dot)  # [N_x, N_y, max_nodes]
        masked_theta_sq = theta.pow(2) * pair_mask

        distances = torch.sqrt(masked_theta_sq.pow(2).sum(dim=-1).clamp_min(eps))  # [N_x, N_y]

        u = y.unsqueeze(0) - dot.unsqueeze(-1) * x.unsqueeze(1)  # [N_x, N_y, max_nodes, z]
        u_norm = u.norm(dim=-1, keepdim=True)

        scale = theta.unsqueeze(-1) / u_norm.clamp_min(eps)
        out = scale * u

        small = theta.unsqueeze(-1) < 1e-5
        first_order = sphere_project_tangent(
            x.unsqueeze(1), y.unsqueeze(0) - x.unsqueeze(1)
        )
        out = torch.where(small, first_order, out)
        diff = sphere_project_tangent(x.unsqueeze(1), out)  # shape: (N_x, N_y, max_nodes, z)

        diff = diff * pair_mask.unsqueeze(-1)
    else:
        raise ValueError("Undefined manifold.")
    return distances, diff


def _compute_V_at_temp(
    temp: float,
    dist_pos: torch.Tensor,
    dist_neg: torch.Tensor,
    diff_pos: torch.Tensor,
    diff_neg: torch.Tensor,
    manifold: str = "euclidean",
) -> torch.Tensor:  
    
    if manifold == "euclidean":
        log_k_pos = - (dist_pos / temp)
        log_k_neg = - (dist_neg / temp)

        W_pos = F.softmax(log_k_pos, dim=-1)
        W_neg = F.softmax(log_k_neg, dim=-1)

        min_laplace_dist = 0.01
        inv_dist_pos = 1.0 / dist_pos.clamp_min(min_laplace_dist)

        dist_neg_safe = dist_neg.clone()
        dist_neg_safe.fill_diagonal_(1.0)
        inv_dist_neg = 1.0 / dist_neg_safe.clamp_min(min_laplace_dist)

        # TO ASK: ask about sqrt dim in floor's code
        term_pos = diff_pos * inv_dist_pos.unsqueeze(-1).unsqueeze(-1) / temp
        term_neg = diff_neg * inv_dist_neg.unsqueeze(-1).unsqueeze(-1) / temp

        V_pos_at_temp = (W_pos.unsqueeze(-1).unsqueeze(-1) * term_pos).sum(dim=1)
        V_neg_at_temp = (W_neg.unsqueeze(-1).unsqueeze(-1) * term_neg).sum(dim=1)

        V_at_temp = V_pos_at_temp - V_neg_at_temp
    elif manifold == "spherical":
        # TO ASK: ask about whtether the gradient version behaves differently under spherical manifold
        V_at_temp = ...
    else:
        raise ValueError("Undefined manifold.")
    return V_at_temp


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
