from typing import Tuple

import torch
import torch.nn.functional as F
import wandb


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""


def _finite_wandb_histogram(values: torch.Tensor) -> wandb.Histogram | None:
    values = values.detach().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return wandb.Histogram(values.cpu().numpy())


def compute_drift_loss(
    phi_gen: torch.Tensor,
    phi_real: torch.Tensor,
    temperatures: Tuple[float, ...] = (0.02, 0.05, 0.2),
) -> Tuple[torch.Tensor, dict]:
    """
    Norm-based kernel drifting field.

    k(x, y) = exp(-1/tau * (||x||^2 - ||y||^2)^2)

    - gen acts as its own negatives; diagonal masked to block gen[i]->gen[i]
    - scale = dist.mean(); scale_inputs = scale / sqrt(D) normalises coords to O(1)
    - per-tau force normalisation: force_scale = sqrt(mean(force^2))
    - loss is computed in scaled-coordinate space: mse(gen/scale_inputs, goal_scaled)

    Args:
        phi_gen:      [N_gen, D]
        phi_real:     [N_real, D]
        temperatures: kernel bandwidths (tau)

    Returns:
        loss, stats
    """
    phi_gen = phi_gen.float()
    phi_real = phi_real.float()

    if not torch.isfinite(phi_gen).all() or not torch.isfinite(phi_real).all():
        bad_gen = (~torch.isfinite(phi_gen).all(dim=-1)).sum().item()
        bad_real = (~torch.isfinite(phi_real).all(dim=-1)).sum().item()
        raise TrainingDivergedException(
            f"Non-finite embeddings ({bad_gen} gen, {bad_real} real)."
        )

    N_gen, D = phi_gen.shape

    old_gen = phi_gen.clone()
    targets = torch.cat([old_gen, phi_real], dim=0)  # [N_gen + N_real, D]

    # Distances from each gen sample to all targets (used for scale + stats)
    dist = torch.cdist(old_gen, targets)  # [N_gen, N_gen + N_real]

    # Scale: mean pairwise distance (uniform weights)
    all_dists = dist.flatten()
    valid_mask = torch.isfinite(all_dists) & (all_dists < 1e5)
    valid = all_dists[valid_mask]

    if valid.numel() == 0:
        raise TrainingDivergedException("No valid distances found; loss is unstable.")

    scale = valid.mean().detach().clamp(min=1e-3)
    scale_inputs = (scale / (D**0.5)).clamp(min=1e-3)

    old_gen_scaled = old_gen / scale_inputs  # [N_gen, D]
    phi_real_scaled = phi_real / scale_inputs  # [N_real, D]

    # Squared norms in scaled space
    gen_norms = (old_gen_scaled**2).sum(dim=1)  # [N_gen]
    real_norms = (phi_real_scaled**2).sum(dim=1)  # [N_real]

    # Norm differences for attraction (gen vs real) and repulsion (gen vs gen)
    diff_pos = gen_norms[:, None] - real_norms[None, :]  # [N_gen, N_real]
    diff_neg = gen_norms[:, None] - gen_norms[None, :]  # [N_gen, N_gen]

    # Mask self-connections: gen[i] -> gen[i]
    diff_neg = diff_neg + torch.eye(N_gen, device=phi_gen.device) * 1e6

    stats = {}
    with torch.no_grad():
        stats["scale_S"] = scale_inputs.item()

        gen_norms_l2 = old_gen_scaled.norm(dim=-1)
        real_norms_l2 = phi_real_scaled.norm(dim=-1)
        stats["phi_gen_norm_mean"] = gen_norms_l2.mean().item()
        stats["phi_gen_norm_std"] = gen_norms_l2.std().item()
        stats["phi_real_norm_mean"] = real_norms_l2.mean().item()

        phi_gen_unit = F.normalize(old_gen, dim=-1)
        phi_real_unit = F.normalize(phi_real, dim=-1)
        cos_sim_matrix = phi_gen_unit @ phi_real_unit.T  # [N_gen, N_real]
        stats["cosine_sim_to_nn"] = cos_sim_matrix.max(dim=1).values.mean().item()
        stats["nn_l2_distance"] = dist[:, N_gen:].min(dim=1).values.mean().item()
        stats["invalid_dist_frac"] = (~valid_mask).float().mean().item()

        # Pairwise stats among generated embeddings (upper triangle only — one entry per pair)
        triu_idx = torch.triu_indices(N_gen, N_gen, offset=1, device=phi_gen.device)
        gen_cos_sim = (phi_gen_unit @ phi_gen_unit.T)[triu_idx[0], triu_idx[1]]
        gen_l2_dist = dist[:, :N_gen][triu_idx[0], triu_idx[1]]
        gen_cos_sim_hist = _finite_wandb_histogram(gen_cos_sim)
        gen_l2_dist_hist = _finite_wandb_histogram(gen_l2_dist)
        if gen_cos_sim_hist is not None:
            stats["gen_pairwise_cos_sim_hist"] = gen_cos_sim_hist
        if gen_l2_dist_hist is not None:
            stats["gen_pairwise_l2_dist_hist"] = gen_l2_dist_hist

    V_across_taus = torch.zeros_like(old_gen_scaled)

    for tau in temperatures:
        tau_key = str(tau).replace(".", "_")

        kernel_pos = torch.exp(-(diff_pos**2) / tau)  # [N_gen, N_real]
        kernel_neg = torch.exp(-(diff_neg**2) / tau)  # [N_gen, N_gen]

        Z_p = kernel_pos.sum(dim=1).clamp(min=1e-8)  # [N_gen]
        Z_q = kernel_neg.sum(dim=1).clamp(min=1e-8)  # [N_gen]

        # Gradient of k(x, y) w.r.t. x, summed over attraction/repulsion targets
        grad_pos = (-1 *kernel_pos * diff_pos).sum(dim=1)  # [N_gen]
        grad_neg = (-1 * kernel_neg * diff_neg).sum(dim=1)  # [N_gen]

        drift_pos = (grad_pos / Z_p).unsqueeze(-1) * old_gen_scaled  # [N_gen, D]
        drift_neg = (grad_neg / Z_q).unsqueeze(-1) * old_gen_scaled  # [N_gen, D]

        total_force_R = drift_pos - drift_neg  # [N_gen, D]

        f_norm_val = (total_force_R**2).mean()
        force_scale = torch.sqrt(f_norm_val.clamp(min=1e-8)).detach()

        V_across_taus = V_across_taus + total_force_R / force_scale

        with torch.no_grad():
            stats[f"force_scale_{tau_key}"] = force_scale.item()
            stats[f"v_norm_{tau_key}"] = (
                (total_force_R / force_scale).norm(dim=-1).mean().item()
            )

    goal_scaled = (old_gen_scaled + V_across_taus).detach()
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
