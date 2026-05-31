"""Molecular alignment using Hungarian matching.

Aligns generated molecules to real molecules by finding optimal atom permutations
through the Hungarian algorithm for atom matching.
"""

# Adding Kabsch algorithm and Hungarian method
import torch
from torch_linear_assignment import batch_linear_assignment
from ..spherical_utils import sphere_normalize
from contextlib import nullcontext


def _hungarian_method_batched(
    gen_types,
    real_types,
    eps,
    weight=1.0,
):
    """Find optimal atom permutations using the Hungarian (linear assignment) algorithm.

    Args:
        gen_types: Generated atom type embeddings, either [N_gen, N_atoms, D] or [N_gen, N_real, N_atoms, D].
        real_types: Real atom types [N_real, N_atoms, D].
        gen_pos: Generated positions, either [N_gen, N_atoms, 3] or [N_gen, N_real, N_atoms, 3].
        real_pos: Real positions [N_real, N_atoms, 3].
        eps: Small constant for numerical stability.
        t_weight: Weight for type distance in cost matrix.
        p_weight: Weight for position distance in cost matrix.

    Returns:
        Assignment tensor [N_gen, N_real, N_atoms] where assignment[g, r, j] = i means
        generated atom i is matched to real atom j.
    """
    cost_matrix = _build_cost_matrix(
        gen_types=gen_types,
        real_types=real_types,
        eps=eps,
        weight=weight,
    )

    N_gen = cost_matrix.shape[0]
    N_real = cost_matrix.shape[1]
    N_atoms = cost_matrix.shape[2]

    cost_flat = cost_matrix.reshape(N_gen * N_real, N_atoms, N_atoms).contiguous()

    with torch.no_grad():
        assignment_flat = batch_linear_assignment(cost_flat)

    assignment = assignment_flat.reshape(N_gen, N_real, N_atoms)

    return assignment


def _build_cost_matrix(
    gen_types,
    real_types,
    eps,
    weight=1.0,
):
    """Build cost matrix for Hungarian algorithm based on type differences.

    Cost combines spherical distance on atom type embeddings, weighted by weight.

    cost[g, r, j_real, i_gen] =
        type_weight * spherical_distance(type_real_j, type_gen_i)^2

    Args:
        gen_types: Generated types [N_gen, N_atoms, D] or [N_gen, N_real, N_atoms, D].
        real_types: Real types [N_real, N_atoms, D].
        eps: Numerical stability constant.
        weight: Weight for type cost.

    Returns:
        Cost matrix [N_gen, N_real, N_atoms, N_atoms_gen] for assignment.
    """
    assert real_types.ndim == 3

    gen_types = sphere_normalize(gen_types, eps)
    real_types = sphere_normalize(real_types, eps)

    if gen_types.ndim == 3:
        gen_types_exp = gen_types[:, None, None, :, :]
    elif gen_types.ndim == 4:
        gen_types_exp = gen_types[:, :, None, :, :]
    else:
        raise ValueError(f"Expected gen_types 3D or 4D, got {gen_types.shape}")

    real_types_exp = real_types[None, :, :, None, :]

    dot = (gen_types_exp * real_types_exp).sum(dim=-1)
    dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    type_dist = torch.acos(dot)
    type_cost = type_dist.pow(2)

    cost_matrix = weight * type_cost

    return cost_matrix


def _to_pairwise(gen, n_real):
    """Expand tensor to pairwise form for all real molecules.

    Args:
        gen: Tensor [N_gen, N_atoms, D] to expand.
        n_real: Number of real molecules.

    Returns:
        Pairwise tensor [N_gen, N_real, N_atoms, D].
    """
    if gen.ndim == 3:
        return gen[:, None, :, :].expand(-1, n_real, -1, -1)
    if gen.ndim == 4:
        return gen
    raise ValueError(f"Expected 3D or 4D tensor, got {gen.shape}")


def permute_generated_to_real_order(gen, assignment):
    """Reorder generated atoms according to assignment to real atoms.

    Args:
        gen: Generated tensor [N_gen, N_atoms, D] or [N_gen, N_real, N_atoms, D].
        assignment: Assignment indices [N_gen, N_real, N_atoms] where
            assignment[g, r, j] = i means gen atom i -> real atom j.

    Returns:
        Permuted tensor [N_gen, N_real, N_atoms, D].
    """
    D = gen.shape[-1]

    if gen.ndim == 3:
        gen_pairwise = gen[:, None, :, :].expand(-1, assignment.shape[1], -1, -1)

    elif gen.ndim == 4:
        gen_pairwise = gen

    else:
        raise ValueError(f"Expected gen to be 3D or 4D, got shape {gen.shape}")

    idx = assignment[..., None].expand(-1, -1, -1, D)

    gen_perm = gen_pairwise.gather(dim=2, index=idx)

    return gen_perm


def unpermute_real_order_to_gen_order(x_perm, assignment):
    """Reverse permutation to restore generation atom order.

    Args:
        x_perm: Permuted tensor [N_gen, N_real, N_atoms, D] in real atom order.
        assignment: Assignment [N_gen, N_real, N_atoms] where
            assignment[g, r, j_real] = i_gen.

    Returns:
        Tensor in original generated atom order [N_gen, N_real, N_atoms, D].
    """
    x = torch.empty_like(x_perm)

    idx = assignment[..., None].expand_as(x_perm)

    x.scatter_(dim=2, index=idx, src=x_perm)

    return x


@torch.no_grad()
def find_permutation(gen_types, real_types, cfg):
    """Iteratively find optimal rotation and atom permutation via Kabsch + Hungarian.

    Alternates between finding optimal atom assignment via Hungarian algorithm and
    optimal rotation via Kabsch algorithm until convergence (position tolerance).

    Args:
        gen_pos: Generated positions [N_gen, N_atoms, 3].
        real_pos: Real positions [N_real, N_atoms, 3].
        gen_types: Generated atom types [N_gen, N_atoms, D].
        real_types: Real atom types [N_real, N_atoms, D].
        cfg: Config dict with eps, max_iter, p_tol, p_weight, t_weight.

    Returns:
        Tuple of (assignment, total_R, final_pos, final_types).
    """

    eps = cfg["eps"]
    weight = cfg["t_weight"]

    g_types = gen_types.clone()

    N_gen = gen_types.shape[0]
    N_real = real_types.shape[0]
    N_atoms = gen_types.shape[1]

    old_g_types = _to_pairwise(g_types, N_real)

    assignment = _hungarian_method_batched(
        g_types,
        real_types,
        eps=eps,
        weight=weight,
    )

    return assignment