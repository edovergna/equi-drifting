import torch
import torch.nn.functional as F


class TrainingDivergedException(Exception):
    pass


def compute_normalized_drift_loss(
    phi_gen: torch.Tensor,
    phi_real: torch.Tensor,
    temperatures: list[float],
):
    phi_gen = torch.nan_to_num(
        phi_gen.float(), nan=0.0, posinf=1e4, neginf=-1e4
    )
    phi_real = torch.nan_to_num(
        phi_real.float(), nan=0.0, posinf=1e4, neginf=-1e4
    )

    D = phi_gen.shape[-1]

    # ------------------------------------------------------------
    # distance normalization (your stabilization)
    # ------------------------------------------------------------

    dist_pos = torch.cdist(phi_gen, phi_real)
    dist_neg = torch.cdist(phi_gen, phi_gen)

    dist_neg.fill_diagonal_(1e6)

    valid = torch.cat([dist_pos.flatten(), dist_neg.flatten()])
    valid = valid[torch.isfinite(valid) & (valid < 1e5)]

    if valid.numel() == 0:
        S = torch.tensor(1.0, device=phi_gen.device)
    else:
        S = (valid.mean() / (D**0.5)).detach()

    S = S.clamp(min=1e-5, max=1e3)

    phi_gen = phi_gen / S
    phi_real = phi_real / S

    aggregated_v = torch.zeros_like(phi_gen)

    stats = {
        "scale_S": S.item(),
    }

    # recompute normalized distances
    dist_pos = torch.cdist(phi_gen, phi_real)
    dist_neg = torch.cdist(phi_gen, phi_gen)

    dist_neg.fill_diagonal_(1e6)

    for tau in temperatures:

        tau_key = str(tau).replace(".", "_")

        tau_eff = tau * (D**0.5)

        logits_pos = -dist_pos / tau_eff
        logits_neg = -dist_neg / tau_eff

        logits = torch.cat([logits_pos, logits_neg], dim=1)
        logits = logits.clamp(min=-100, max=50)

        # ------------------------------------------------------------
        # exact Algorithm 2 normalization
        # ------------------------------------------------------------

        A_row = F.softmax(logits, dim=-1)
        A_col = F.softmax(logits, dim=-2)

        A = torch.sqrt(torch.clamp(A_row * A_col, min=1e-30))

        N_real = phi_real.size(0)

        A_pos, A_neg = torch.split(
            A,
            [N_real, phi_gen.size(0)],
            dim=1,
        )

        # ------------------------------------------------------------
        # CRITICAL: coupled normalization from paper
        # ------------------------------------------------------------

        pos_mass = A_pos.sum(dim=1, keepdim=True)
        neg_mass = A_neg.sum(dim=1, keepdim=True)

        W_pos = A_pos * neg_mass
        W_neg = A_neg * pos_mass

        drift_pos = W_pos @ phi_real
        drift_neg = W_neg @ phi_gen

        V_tau = drift_pos - drift_neg

        # optional stabilization (NOT in paper)
        # comment out for faithful implementation

        lambda_tau = torch.sqrt(
            (V_tau.pow(2).sum(dim=-1).mean() / D).clamp(min=1e-10)
        ).detach()

        V_tau = V_tau / lambda_tau.clamp(min=1e-5)

        aggregated_v = aggregated_v + V_tau

        with torch.no_grad():
            stats[f"lambda_{tau_key}"] = lambda_tau.item()
            stats[f"v_norm_{tau_key}"] = (
                V_tau.norm(dim=-1).mean().item()
            )

    target = (phi_gen + aggregated_v).detach()

    loss = F.mse_loss(phi_gen, target)

    if not torch.isfinite(loss):
        raise TrainingDivergedException(
            f"Non-finite drift loss: {loss.item()}"
        )

    return loss, stats
