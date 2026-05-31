"""Molecular alignment using Kabsch algorithm and Hungarian matching.

Aligns generated molecules to real molecules by finding optimal atom permutations
and rotations through iterative refinement using the Kabsch algorithm for rigid
registration and Hungarian algorithm for atom matching.
"""

# Adding Kabsch algorithm and Hungarian method
import torch
from torch_linear_assignment import batch_linear_assignment
from ..spherical_utils import sphere_normalize
from contextlib import nullcontext


def _kabsch_rotations(gen_pos, real_pos):
    """Compute optimal rotation matrices aligning generated to real molecules.

    Uses Singular Value Decomposition (SVD) to find rotation matrices minimizing
    squared distance between centered point clouds.

    Args:
        gen_pos: Generated positions, either [N_gen, N_atoms, 3] or [N_gen, N_real, N_atoms, 3].
        real_pos: Real positions [N_real, N_atoms, 3].

    Returns:
        Rotation matrices [N_gen, N_real, 3, 3] (or [N_gen, N_gen, 3, 3] if pairwise).
    """
    out_dtype = gen_pos.dtype
    device_type = gen_pos.device.type
    autocast_ctx = (
        torch.autocast(device_type=device_type, enabled=False)
        if device_type in {"cuda", "cpu"}
        else nullcontext()
    )

    with autocast_ctx:
        gen_pos = gen_pos.float()
        real_pos = real_pos.float()

        if real_pos.ndim != 3:
            raise ValueError(
                f"real_pos must be [N_real, N_atoms, 3], got {real_pos.shape}"
            )

        if gen_pos.ndim == 3:
            if gen_pos.shape[1:] != real_pos.shape[1:]:
                raise ValueError(
                    f"Shape mismatch: gen_pos {gen_pos.shape}, real_pos {real_pos.shape}"
                )

            gen_c = gen_pos - gen_pos.mean(dim=1, keepdim=True)
            real_c = real_pos - real_pos.mean(dim=1, keepdim=True)
            H = torch.einsum("gni,rnj->grij", gen_c, real_c)

        elif gen_pos.ndim == 4:
            if gen_pos.shape[1] != real_pos.shape[0]:
                raise ValueError(
                    f"Pairwise gen_pos has N_real={gen_pos.shape[1]}, "
                    f"but real_pos has N_real={real_pos.shape[0]}"
                )
            if gen_pos.shape[2:] != real_pos.shape[1:]:
                raise ValueError(
                    f"Shape mismatch: gen_pos {gen_pos.shape}, real_pos {real_pos.shape}"
                )

            gen_c = gen_pos - gen_pos.mean(dim=2, keepdim=True)
            real_c = real_pos - real_pos.mean(dim=1, keepdim=True)
            H = torch.einsum("grni,rnj->grij", gen_c, real_c)

        else:
            raise ValueError(f"gen_pos must be 3D or 4D, got {gen_pos.shape}")

        H_flat = H.reshape(-1, 3, 3)

        U, S, Vh = torch.linalg.svd(H_flat)

        V = Vh.transpose(-2, -1)
        Ut = U.transpose(-2, -1)

        R = V @ Ut

        return R.reshape(H.shape[0], H.shape[1], 3, 3).to(out_dtype)


def _hungarian_method_batched(
    gen_types,
    real_types,
    gen_pos,
    real_pos,
    eps,
    t_weight=1.0,
    p_weight=1.0,
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
        gen_pos=gen_pos,
        real_pos=real_pos,
        eps=eps,
        t_weight=t_weight,
        p_weight=p_weight,
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
    gen_pos,
    real_pos,
    eps,
    t_weight=1.0,
    p_weight=1.0,
):
    """Build cost matrix for Hungarian algorithm based on type and position differences.

    Cost combines spherical distance on atom type embeddings and Euclidean distance
    on atomic positions, weighted by t_weight and p_weight respectively.

    cost[g, r, j_real, i_gen] =
        type_weight * spherical_distance(type_real_j, type_gen_i)^2
        + pos_weight * euclidean_distance(pos_real_j, pos_gen_i)^2

    Args:
        gen_types: Generated types [N_gen, N_atoms, D] or [N_gen, N_real, N_atoms, D].
        real_types: Real types [N_real, N_atoms, D].
        gen_pos: Generated positions [N_gen, N_atoms, 3] or [N_gen, N_real, N_atoms, 3].
        real_pos: Real positions [N_real, N_atoms, 3].
        eps: Numerical stability constant.
        t_weight: Weight for type cost.
        p_weight: Weight for position cost.

    Returns:
        Cost matrix [N_gen, N_real, N_atoms, N_atoms_gen] for assignment.
    """
    assert real_types.ndim == 3
    assert real_pos.ndim == 3

    gen_types = sphere_normalize(gen_types, eps)
    real_types = sphere_normalize(real_types, eps)

    if gen_types.ndim == 3:
        gen_types_exp = gen_types[:, None, None, :, :]
        gen_pos_exp = gen_pos[:, None, None, :, :]

    elif gen_types.ndim == 4:
        gen_types_exp = gen_types[:, :, None, :, :]
        gen_pos_exp = gen_pos[:, :, None, :, :]

    else:
        raise ValueError(f"Expected gen_types 3D or 4D, got {gen_types.shape}")

    real_types_exp = real_types[None, :, :, None, :]
    real_pos_exp = real_pos[None, :, :, None, :]

    dot = (gen_types_exp * real_types_exp).sum(dim=-1)
    dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    type_dist = torch.acos(dot)
    type_cost = type_dist.pow(2)

    pos_cost = (gen_pos_exp - real_pos_exp).pow(2).sum(dim=-1)

    cost_matrix = t_weight * type_cost + p_weight * pos_cost

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


def _pairwise_position_rmse(gen_pos_pairwise, real_pos):
    """Compute RMSE for pairwise position alignment.

    Args:
        gen_pos_pairwise: [N_gen, N_real, N_atoms, 3]
        real_pos: [N_real, N_atoms, 3]

    Returns:
        RMSE values [N_gen, N_real].
    """
    diff = gen_pos_pairwise - real_pos[None, :, :, :]
    return diff.pow(2).sum(dim=-1).mean(dim=-1).sqrt()


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


def apply_pairwise_rotation(gen_pos, R):
    """Apply rotation matrices to generated positions.

    Args:
        gen_pos: Positions [N_gen, N_atoms, 3] or [N_gen, N_real, N_atoms, 3].
        R: Rotation matrices [N_gen, N_real, 3, 3].

    Returns:
        Rotated positions [N_gen, N_real, N_atoms, 3].
    """
    if gen_pos.ndim == 3:
        gen_pairwise = gen_pos[:, None, :, :].expand(-1, R.shape[1], -1, -1)
    elif gen_pos.ndim == 4:
        gen_pairwise = gen_pos
    else:
        raise ValueError(f"gen_pos must be 3D or 4D, got {gen_pos.shape}")

    return gen_pairwise @ R


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
def find_rotation_and_permutation(gen_pos, real_pos, gen_types, real_types, cfg):
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
    max_iter = cfg["max_iter"]
    pos_tol = cfg["p_tol"]
    p_weight = cfg["p_weight"]
    t_weight = cfg["t_weight"]

    g_types = gen_types.clone()
    g_pos = gen_pos.clone()

    N_gen = gen_pos.shape[0]
    N_real = real_pos.shape[0]
    N_atoms = gen_pos.shape[1]

    active = torch.ones(N_gen, N_real, device=gen_pos.device, dtype=torch.bool)

    total_assignment = (
        torch.arange(
            N_atoms,
            device=gen_pos.device,
        )
        .view(1, 1, N_atoms)
        .expand(N_gen, N_real, N_atoms)
        .clone()
    )

    total_R = (
        torch.eye(
            3,
            device=gen_pos.device,
            dtype=gen_pos.dtype,
        )
        .view(1, 1, 3, 3)
        .expand(N_gen, N_real, 3, 3)
        .clone()
    )

    for step in range(max_iter):
        pos_weight = 0.0 if step == 0 else p_weight
        old_g_pos = _to_pairwise(g_pos, N_real)
        old_g_types = _to_pairwise(g_types, N_real)

        step_assignment = _hungarian_method_batched(
            g_types,
            real_types,
            g_pos,
            real_pos,
            eps=eps,
            t_weight=t_weight,
            p_weight=pos_weight,
        )

        cand_g_pos = permute_generated_to_real_order(g_pos, step_assignment)
        cand_g_types = permute_generated_to_real_order(g_types, step_assignment)

        step_R = _kabsch_rotations(cand_g_pos, real_pos)
        cand_g_pos = apply_pairwise_rotation(cand_g_pos, step_R)

        cand_total_assignment = total_assignment.gather(
            dim=2,
            index=step_assignment,
        )

        cand_total_R = total_R @ step_R

        pair_mask = active[..., None, None]
        assign_mask = active[..., None]

        g_pos = torch.where(pair_mask, cand_g_pos, old_g_pos)
        g_types = torch.where(pair_mask, cand_g_types, old_g_types)

        total_assignment = torch.where(
            assign_mask,
            cand_total_assignment,
            total_assignment,
        )

        total_R = torch.where(
            pair_mask,
            cand_total_R,
            total_R,
        )

        rmse = _pairwise_position_rmse(g_pos, real_pos)
        done = rmse <= pos_tol

        active = ~done

        if not active.any():
            break

    return total_assignment, total_R, g_pos, g_types
