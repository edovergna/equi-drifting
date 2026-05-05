from typing import Tuple

import torch
import torch.nn.functional as F


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""


def compute_drift_loss(
    phi_gen: torch.Tensor,
    phi_real: torch.Tensor,
    temperatures: Tuple[float, ...] = (0.02, 0.05, 0.2),
) -> Tuple[torch.Tensor, dict]:
    """
    Faithful PyTorch translation of the original JAX drift_loss.

    Key design choices that match the original:
    - targets = [stop_grad(gen), phi_real]; gen acts as its own negatives
    - diagonal mask only blocks gen[i]->gen[i] self-connections
    - scale = dist.mean(); scale_inputs = scale / sqrt(D) normalises coords to O(1)
    - per-R force normalisation: force_scale = sqrt(mean(force^2))
    - loss is computed in scaled-coordinate space: mse(gen/scale_inputs, goal_scaled)

    Args:
        phi_gen:  [N_gen, D]
        phi_real: [N_real, D]
        R_list:   kernel bandwidths (equivalent to tau in the paper)

    Returns:
        loss, stats
    """
    phi_gen = phi_gen.float()
    phi_real = phi_real.float()

    N_gen, D = phi_gen.shape
    N_real = phi_real.shape[0]
    N_targets = N_gen + N_real

    old_gen = phi_gen.detach()
    targets = torch.cat([old_gen, phi_real], dim=0)  # [N_targets, D]

    # Distances from each gen sample to all targets
    dist = torch.cdist(old_gen, targets)  # [N_gen, N_targets]

    # Scale: mean pairwise distance (uniform weights)
    all_dists = dist.flatten()
    valid_mask = torch.isfinite(all_dists) & (all_dists < 1e5)
    valid = all_dists[valid_mask]

    if valid.numel() == 0:
        raise TrainingDivergedException("No valid distances found; loss is unstable.")

    scale = valid.mean().detach().clamp(min=1e-3)
    scale_inputs = (scale / (D ** 0.5)).clamp(min=1e-3)

    old_gen_scaled = old_gen / scale_inputs       # [N_gen, D]
    targets_scaled = targets / scale_inputs        # [N_targets, D]
    dist_normed = dist / scale                     # [N_gen, N_targets]

    # Mask self-connections: gen[i] -> target[i] (gen block only)
    diag_mask = torch.zeros(N_gen, N_targets, device=phi_gen.device)
    diag_mask[:, :N_gen] = torch.eye(N_gen, device=phi_gen.device)
    dist_normed = dist_normed + diag_mask * 100.0

    stats = {}
    with torch.no_grad():
        stats["scale_S"] = scale_inputs.item()

        gen_norms = old_gen_scaled.norm(dim=-1)
        real_norms = (phi_real / scale_inputs).norm(dim=-1)
        stats["phi_gen_norm_mean"] = gen_norms.mean().item()
        stats["phi_gen_norm_std"] = gen_norms.std().item()
        stats["phi_real_norm_mean"] = real_norms.mean().item()

        phi_gen_unit = F.normalize(old_gen, dim=-1)
        phi_real_unit = F.normalize(phi_real, dim=-1)
        cos_sim_matrix = phi_gen_unit @ phi_real_unit.T  # [N_gen, N_real]
        stats["cosine_sim_to_nn"] = cos_sim_matrix.max(dim=1).values.mean().item()
        stats["nn_l2_distance"] = dist[:, N_gen:].min(dim=1).values.mean().item()
        stats["invalid_dist_frac"] = (~valid_mask).float().mean().item()

    force_across_R = torch.zeros_like(old_gen_scaled)

    for R in temperatures:
        R_key = str(R).replace(".", "_")
        logits = -dist_normed / R  # [N_gen, N_targets]

        A_row = F.softmax(logits, dim=-1)   # [N_gen, N_targets]
        A_col = F.softmax(logits, dim=-2)   # [N_gen, N_targets]
        A = torch.sqrt(torch.clamp(A_row * A_col, min=1e-6))

        # neg block = gen columns, pos block = real columns
        aff_neg = A[:, :N_gen]   # [N_gen, N_gen]
        aff_pos = A[:, N_gen:]   # [N_gen, N_real]

        sum_pos = aff_pos.sum(dim=-1, keepdim=True)  # [N_gen, 1]
        sum_neg = aff_neg.sum(dim=-1, keepdim=True)  # [N_gen, 1]

        r_coeff_neg = -aff_neg * sum_pos  # [N_gen, N_gen]  repulsion
        r_coeff_pos = aff_pos * sum_neg   # [N_gen, N_real] attraction

        R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=1)  # [N_gen, N_targets]

        total_force_R = R_coeff @ targets_scaled  # [N_gen, D]

        # Centering correction (always ~0 with uniform weights, kept for faithfulness)
        total_coeffs = R_coeff.sum(dim=-1)  # [N_gen]
        total_force_R = total_force_R - total_coeffs.unsqueeze(-1) * old_gen_scaled

        f_norm_val = (total_force_R ** 2).mean()
        force_scale = torch.sqrt(f_norm_val.clamp(min=1e-8)).detach()

        force_across_R = force_across_R + total_force_R / force_scale

        with torch.no_grad():
            row_entropy = -(A_row * (A_row + 1e-30).log()).sum(dim=-1).mean()
            row_entropy_uniform = torch.log(
                torch.tensor(N_targets, device=A_row.device, dtype=torch.float)
            )
            stats[f"attn_entropy_{R_key}"] = row_entropy.item()
            stats[f"attn_entropy_rel_{R_key}"] = (row_entropy / row_entropy_uniform).item()
            stats[f"lambda_{R_key}"] = force_scale.item()
            stats[f"loss_{R_key}"] = f_norm_val.item()
            stats[f"v_norm_{R_key}"] = (total_force_R / force_scale).norm(dim=-1).mean().item()
            stats[f"frac_zero_dists_{R_key}"] = (A_row == 0).float().mean().item()

            pos_mass = aff_pos.sum(dim=-1)
            neg_mass = aff_neg.sum(dim=-1)
            stats[f"attn_pos_mass_frac_{R_key}"] = (
                (pos_mass / (pos_mass + neg_mass).clamp_min(1e-8)).mean().item()
            )
            stats[f"total_coeffs_abs_mean_{R_key}"] = total_coeffs.abs().mean().item()

    goal_scaled = (old_gen_scaled + force_across_R).detach()
    gen_scaled = phi_gen / scale_inputs

    loss = F.mse_loss(gen_scaled, goal_scaled)

    if not torch.isfinite(loss):
        raise TrainingDivergedException(
            f"Non-finite loss ({loss.item()!r}). "
            f"phi_gen range: [{phi_gen.min().item():.3g}, {phi_gen.max().item():.3g}], "
            f"phi_real range: [{phi_real.min().item():.3g}, {phi_real.max().item():.3g}], "
            f"scale={scale_inputs.item():.3g}"
        )

    return loss, stats
