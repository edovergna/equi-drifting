import torch
from .align import kabsch_align_pairwise


def compute_pairwise_drifting_field(
    gen_pos: torch.Tensor,
    target_pos: torch.Tensor,
    sigma: float = 1.0,
    exclude_self: bool = False,
) -> torch.Tensor:
    """Drift field with a separate Kabsch alignment per (gen, target) pair.

    For each pair (i, j):
      1. Rotate gen[i] to target[j]'s frame via R[i,j]
      2. Compute diff = target[j] - gen[i] @ R[i,j]  (in aligned frame)
      3. Rotate diff back: diff @ R[i,j]^T          (in original gen frame)
      4. Weight by Gaussian kernel of the per-pair aligned distance

    Both gen_pos and target_pos must be zero-centered (CoM = 0) before calling.

    Args:
        gen_pos: [N_gen, N_atoms, 3]
        target_pos: [N_target, N_atoms, 3]
        sigma: Gaussian kernel bandwidth (acts on sum-of-squared atom distances)
        exclude_self: mask the diagonal; requires N_gen == N_target (for gen-vs-gen)

    Returns:
        field: [N_gen, N_atoms, 3] drift vectors in the original gen frame
    """
    N_gen = gen_pos.shape[0]

    with torch.no_grad():
        R, aligned = kabsch_align_pairwise(gen_pos, target_pos)

    # diff[g, r] = target[r] - gen[g] @ R[g,r]  (in aligned frame)
    # [N_gen, N_target, N_atoms, 3]
    diff = target_pos[None, :, :, :] - aligned

    # [N_gen, N_target]
    sq_dist = (diff ** 2).sum(dim=-1).sum(dim=-1)
    kernel = torch.exp(-sq_dist / (2 * sigma ** 2))

    if exclude_self:
        assert gen_pos.shape[0] == target_pos.shape[0], "exclude_self requires N_gen == N_target"
        eye = torch.eye(N_gen, device=gen_pos.device, dtype=torch.bool)
        kernel = kernel.masked_fill(eye, 0.0)

    # Rotate each per-pair contribution back to the original gen frame.
    # diff[g,r] is in the frame where gen[g] was rotated by R[g,r], so the
    # inverse is R[g,r]^T.
    
    # [N_gen, N_target, 3, 3]
    inv_R = R.transpose(-2, -1)
    # [N_gen, N_target, N_atoms, 3]
    weighted_diff = diff * kernel[:, :, None, None]
    # [N_gen, N_target, N_atoms, 3]
    field_back = weighted_diff @ inv_R

    # [N_gen]
    Z = kernel.sum(dim=1).clamp_min(1e-8)
    
    # [N_gen, N_atoms, 3]
    return field_back.sum(dim=1) / Z[:, None, None]


def compute_individual_drifting_field(
    gen_mol: torch.Tensor,
    target_mol: torch.Tensor,
    sigma: float = 1.0,
    R: torch.Tensor = None,
    casadeval: bool = False
) -> torch.Tensor:
    """Computes the drifting field between real and generated molecules
    
    Keyword arguments:
        gen_mol -- [N_gen, D] tensor of generated molecule positions
        target_mol -- [N_target, D] tensor of target molecule positions
        sigma -- bandwidth parameter for the Gaussian kernel
        R -- Optional [N_total, 3, 3] tensor of rotation matrices to apply to the field
        casadeval -- Whether to scale the field by 1/sigma^2 to match Esteban-Casadeval's definition.
    Return:
        field -- [N_gen, D] tensor of the drifting field for each generated molecule
    """

    n_gen = gen_mol.size(0)

    # [N_gen, N_targets, D]
    diffs = target_mol.unsqueeze(0) - gen_mol.unsqueeze(1)

    # [N_gen, N_targets]
    sq_dists = (diffs * diffs).sum(dim=-1)

    # [N_gen, N_targets]
    kernel = torch.exp(-sq_dists / (2 * sigma**2))

    # [N_gen, N_targets, D]
    weighted_diffs = kernel.unsqueeze(-1) * diffs 

    # We add the division by sigma^2 here to match the
    # gradient of the Gaussian kernel, matching
    # Esteban-Casadeval's definition.
    if casadeval:
        # NOTE: This seems to be scaling the field too aggressively
        # for the positions. It isn't even able to overfit to a single
        # sample.
        weighted_diffs = weighted_diffs / (sigma**2)
        
    # [N_gen, 1]
    Z = kernel.sum(dim=1, keepdim=True)
    
    # [N_gen, D]
    field = weighted_diffs.sum(dim=1) / (Z + 1e-8)

    if R is not None:
        # When a rotation matrix is provided then we rotate the field
        # R is [N_total, 3, 3], we need to reshape the field to apply the rotation per node
        field = torch.einsum('nji,nj->ni', R, field.view(-1, 3)).view(n_gen, -1)
    
    return field

def compute_euclidean_drifting_field(
    gen_mol: torch.Tensor,
    real_mol: torch.Tensor,
    sigma: float = 1.0,
    R: torch.Tensor = None,
    casadeval: bool = False
) -> torch.Tensor:
    """Computes the drifting field between real and generated molecules
    
    Keyword arguments:
        gen_mol -- [N_gen, D] tensor of generated molecule positions
        real_mol -- [N_real, D] tensor of real molecule positions
        sigma -- bandwidth parameter for the Gaussian kernel
        R -- Optional [N_total, 3, 3] tensor of rotation matrices to apply to the field
        casadeval -- Whether to scale the field by 1/sigma^2 to match Esteban-Casadeval's definition.
    Return:
        field -- [N_gen, D] tensor of the drifting field for each generated molecule
    """

    n_gen = gen_mol.size(0)
    n_real = real_mol.size(0)

    targets = torch.cat([gen_mol, real_mol], dim=0)

    # [N_gen, N_targets, D]
    diffs = targets.unsqueeze(0) - gen_mol.unsqueeze(1)

    # [N_gen, N_targets]
    sq_dists = (diffs * diffs).sum(dim=-1)

    # [N_gen, N_targets]
    kernel = torch.exp(-sq_dists / (2 * sigma**2))

    # [N_gen, N_gen], [N_gen, N_real]
    kernel_neg, kernel_pos = torch.split(kernel, [n_gen, n_real], dim=1)
    # [N_gen, N_gen, D], [N_gen, N_real, D]
    diffs_neg, diffs_pos = torch.split(diffs, [n_gen, n_real], dim=1)

    # [N_gen, N_gen, D]
    weighted_neg = kernel_neg.unsqueeze(-1) * diffs_neg 
    weighted_pos = kernel_pos.unsqueeze(-1) * diffs_pos

    # We add the division by sigma^2 here to match the
    # gradient of the Gaussian kernel, matching
    # Esteban-Casadeval's definition.
    if casadeval:
        # NOTE: This seems to be scaling the field too aggressively
        # for the positions. It isn't even able to overfit to a single
        # sample.
        weighted_neg = weighted_neg / (sigma**2)
        weighted_pos = weighted_pos / (sigma**2)
        
    # [N_gen, 1]
    Z_neg = kernel_neg.sum(dim=1, keepdim=True)
    Z_pos = kernel_pos.sum(dim=1, keepdim=True)
    
    # [N_gen, D]
    v_neg = weighted_neg.sum(dim=1) / (Z_neg + 1e-8)
    v_pos = weighted_pos.sum(dim=1) / (Z_pos + 1e-8)

    field = v_pos - v_neg
    if R is not None:
        # When a rotation matrix is provided then we rotate the field
        # R is [N_total, 3, 3], we need to reshape the field to apply the rotation per node
        field = torch.einsum('nji,nj->ni', R, field.view(-1, 3)).view(n_gen, -1)
    
    return field
