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
