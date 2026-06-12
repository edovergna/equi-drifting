import torch

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
