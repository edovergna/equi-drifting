import torch
import torch.nn.functional as F


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""


def compute_normalized_drift_loss(
    phi_gen: torch.Tensor,
    phi_real: torch.Tensor,
    temperatures: list[float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Normalized drifting field loss (Algorithm 2).

    Returns (loss, stats) where stats is a flat dict of float diagnostics safe to
    pass directly to self.log(). Raises TrainingDivergedException on non-finite loss.
    """
    phi_gen = torch.nan_to_num(phi_gen.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    phi_real = torch.nan_to_num(phi_real.float(), nan=0.0, posinf=1e4, neginf=-1e4)

    D = phi_gen.shape[-1]

    dist_pos = torch.cdist(phi_gen, phi_real)
    dist_neg = torch.cdist(phi_gen, phi_gen)
    dist_neg.fill_diagonal_(1e6)

    all_dists = torch.cat([dist_pos.flatten(), dist_neg.flatten()])
    valid_dists = all_dists[torch.isfinite(all_dists) & (all_dists < 1e5)]

    if valid_dists.numel() == 0:
        S = torch.tensor(1.0, device=phi_gen.device, dtype=phi_gen.dtype)
    else:
        S = (valid_dists.mean() / (D**0.5)).detach()
    S = S.clamp(min=1e-5, max=1e3)

    phi_gen_norm = phi_gen / S
    phi_real_norm = phi_real / S
    norm_dist_pos = dist_pos / S
    norm_dist_neg = dist_neg / S

    aggregated_v_norm = torch.zeros_like(phi_gen_norm)
    stats: dict[str, float] = {}

    with torch.no_grad():
        stats["scale_S"] = S.item()
        gen_norms = phi_gen_norm.norm(dim=-1)
        real_norms = phi_real_norm.norm(dim=-1)
        stats["phi_gen_norm_mean"] = gen_norms.mean().item()
        stats["phi_gen_norm_std"] = gen_norms.std().item()
        stats["phi_real_norm_mean"] = real_norms.mean().item()

        # Angular and distance proximity to nearest real neighbour
        phi_gen_unit = F.normalize(phi_gen, dim=-1)
        phi_real_unit = F.normalize(phi_real, dim=-1)
        cos_sim_matrix = phi_gen_unit @ phi_real_unit.T  # [N_gen, N_real]
        stats["cosine_sim_to_nn"] = cos_sim_matrix.max(dim=1).values.mean().item()
        stats["nn_l2_distance"] = dist_pos.min(dim=1).values.mean().item()

    for tau in temperatures:
        tau_key = str(tau).replace(".", "_")
        tau_tilde = tau * (D**0.5)

        logit = torch.clamp(
            torch.cat([-norm_dist_pos / tau_tilde, -norm_dist_neg / tau_tilde], dim=1),
            min=-100.0,
            max=50.0,
        )

        A_row = F.softmax(logit, dim=-1)
        A_col = F.softmax(logit, dim=-2)
        A = torch.sqrt(torch.clamp(A_row * A_col, min=1e-30))

        A_pos, A_neg = torch.split(A, [phi_real.size(0), phi_gen.size(0)], dim=1)
        W_pos = A_pos / A_pos.sum(dim=1, keepdim=True).clamp_min(1e-5)
        W_neg = A_neg / A_neg.sum(dim=1, keepdim=True).clamp_min(1e-5)

        V_tau = W_pos @ phi_real_norm - W_neg @ phi_gen_norm

        lambda_tau = (
            torch.sqrt(((V_tau**2).sum(dim=-1).mean() / D).clamp(min=1e-10))
            .detach()
            .clamp(min=1e-5, max=1e3)
        )

        V_tau_norm = V_tau / lambda_tau
        aggregated_v_norm = aggregated_v_norm + V_tau_norm

        with torch.no_grad():
            # Entropy of row-softmax: 0 = peaked on single neighbour, log(N) = uniform
            row_entropy = -(A_row * (A_row + 1e-30).log()).sum(dim=-1).mean()
            stats[f"attn_entropy_{tau_key}"] = row_entropy.item()
            stats[f"lambda_{tau_key}"] = lambda_tau.item()
            stats[f"v_norm_{tau_key}"] = V_tau_norm.norm(dim=-1).mean().item()

            # Fraction of attention mass on real samples; ~0.5 at convergence (equal pull)
            pos_mass = A_pos.sum(dim=1)
            neg_mass = A_neg.sum(dim=1)
            stats[f"attn_pos_mass_frac_{tau_key}"] = (
                (pos_mass / (pos_mass + neg_mass).clamp_min(1e-8)).mean().item()
            )

    target = (phi_gen_norm + aggregated_v_norm).detach()
    loss = F.mse_loss(phi_gen_norm, target)

    if not torch.isfinite(loss):
        raise TrainingDivergedException(
            f"Non-finite loss ({loss.item()!r}). "
            f"phi_gen range: [{phi_gen.min().item():.3g}, {phi_gen.max().item():.3g}], "
            f"phi_real range: [{phi_real.min().item():.3g}, {phi_real.max().item():.3g}], "
            f"S={S.item():.3g}"
        )

    return loss, stats


def compute_molecule_based_drift_loss(
        pos_gen: torch.Tensor,
        x_gen: torch.Tensor,
        pos_real: torch.Tensor,
        x_real: torch.Tensor,
        index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Drifting field loss directly on 3D molecules.
    Assumes that the x_gen and x_real are already mapped to the spherical space.

    Returns (loss, stats) where stats is a flat dict of float diagnostics safe to
    pass directly to self.log(). Raises TrainingDivergedException on non-finite loss.
    """
    # TODO

    gen_distances, gen_angles = prep_batch_for_kernel(pos_gen, index)
    real_distances, real_angles = prep_batch_for_kernel(pos_real, index)

    kernel_pos = molecule_kernel(gen_distances, gen_angles, x_gen, 
                                 real_distances, real_angles, 
                                 x_real, index)                                 # shape: [Num_Gen_Mol, Num_Real_Mol]
    kernel_neg = molecule_kernel(gen_distances, gen_angles, x_gen, 
                                 real_distances=gen_distances, real_angles=gen_angles,
                                 x_real=x_gen, index=index)                     # shape: [Num_Gen_Mol, Num_Gen_Mol]
    
    N_pos, N_neg = kernel_pos.shape[1], kernel_neg.shape[1]
    exp_pos = torch.clamp(kernel_pos.sum(dim=1) / N_pos, min=1e-8)
    exp_neg = torch.clamp(kernel_neg.sum(dim=1) / N_neg, min=1e-8)

    log_exp_pos = torch.log(exp_pos)
    log_exp_neg = torch.log(exp_neg)
    
    # Automatically obtain relevant gradients with regards to the inputs separately
    # Can check whether to obtain an analytical gradient instead
    grad_pos_pos = torch.autograd.grad(
                        outputs=log_exp_pos.sum(),
                        inputs=pos_gen)[0]
    grad_pos_neg = torch.autograd.grad(
                        outputs=log_exp_neg.sum(),
                        inputs=pos_gen)[0]
    
    grad_types_pos = torch.autograd.grad(
                        outputs=log_exp_pos.sum(),
                        inputs=x_gen)[0]
    grad_types_neg = torch.autograd.grad(
                        outputs=log_exp_neg.sum(),
                        inputs=x_gen)[0]
    
    # Obtain drift field by subtracting
    v_positions = grad_pos_pos - grad_pos_neg
    v_types = grad_types_pos - grad_types_neg

    # Can now obtain the targets
    target_positions = (pos_gen + v_positions).detach()
    
    # Target for atom types is through the exponential mapping of the spherical space:
    v_types = v_types - ((x_gen * v_types).sum(dim=-1, keepdim=True)) * x_gen       # Tangent projection of drifting field
    v_types_norm = torch.clamp(torch.norm(v_types, dim=-1, keepdim=True), min=1e-8)
    target_types = (torch.cos(v_types_norm) * x_gen + torch.sin(v_types_norm) * (v_types / v_types_norm)).detach()

    # Distance metric for the euclidean space
    euclidean_distances = ((pos_gen - target_positions) ** 2).sum(-1)
    
    # Calculating distances for atom types on the sphere
    cos_sim = (x_gen * target_types).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    spherical_distances = torch.acos(cos_sim) ** 2

    # Combine the distances 
    # TODO: check whether it would be better to use weights by looking at the distances from the stats
    combined_distances = euclidean_distances + spherical_distances

    # Calculate final loss as the expectation over the distances per molecule
    num_molecules = int(index.max().item()) + 1

    sum_per_molecule = torch.zeros(
        num_molecules,
        device=combined_distances.device,
        dtype=combined_distances.dtype,
    )

    sum_per_molecule.scatter_add_(0, index, combined_distances)
    loss = sum_per_molecule.mean()

    if not torch.isfinite(loss):
        raise TrainingDivergedException(
            f"Non-finite loss ({loss.item()!r}). "
        )
    
    stats: dict[str, float] = {}
    
    with torch.no_grad():
        # Size of euclidean and spherical distances
        stats["average_euclidean_distance"] = euclidean_distances.mean().item()
        stats["average_spherical_distance"] = spherical_distances.mean().item()
        stats["std_euclidean_distance"] = euclidean_distances.std().item()
        stats["std_spherical_distance"] = spherical_distances.std().item()

    return loss, stats

# TODO
def prep_batch_for_kernel(
        positions: torch.Tensor,
        index: torch.Tensor
):
    """
    Returns for a batch of molecules (per molecule); the pairwise distances between atoms, 
    the angles between the connections of atoms.
    """
    # For a batch [Total_Atoms, 3], it should return a tensor that stores the distances between atoms 
    # within molecules, and a tensor that stores the angles between connections of atoms within molcules. 
    # Shape for distance tensor should be ig [Total_Atoms, Total_Atoms], where we skip the calculation based
    # on the indexing for which atoms belong to which molecule. 
    # Shape for angle tensor should be ig . Again skip based on molecules,
    # and the angle stored is for the first index of the tensor as "the middle atom", so that the angle between the 
    # connections of this atom with the other atoms are checked. It stores cosine angles.

# TODO
def molecule_kernel(
        gen_distances: torch.Tensor,
        gen_angles: torch.Tensor,
        x_gen: torch.Tensor,
        real_distances: torch.Tensor,
        real_angles: torch.Tensor,
        x_real: torch.Tensor,
        index: torch.Tensor    
):
    """
    Calculates the molecule kernel between each combination of real versus generated molecules
    """
