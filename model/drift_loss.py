import torch
import torch.nn.functional as F


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite."""


def compute_drift_loss(
    phi_gen: torch.Tensor,
    phi_real: torch.Tensor,
    tau: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Faithful implementation of Algorithm 2 from:
    'Generative Modeling via Drifting'

    Computes:
        loss = ||V||^2

    where:
        target = stopgrad(phi_gen + V)

    Args:
        phi_gen: [N_gen, D]
        phi_real: [N_real, D]
        tau: kernel temperature

    Returns:
        loss, stats
    """

    phi_gen = phi_gen.float()
    phi_real = phi_real.float()

    N_gen, D = phi_gen.shape
    N_real = phi_real.shape[0]

    # ------------------------------------------------------------
    # Pairwise distances
    # ------------------------------------------------------------

    dist_pos = torch.cdist(phi_gen, phi_real)   # [N_gen, N_real]
    dist_neg = torch.cdist(phi_gen, phi_gen)    # [N_gen, N_gen]

    # ignore self-matches among negatives
    dist_neg.fill_diagonal_(1e6)

    # ------------------------------------------------------------
    # Kernel logits
    #
    # Paper Eq. (12):
    # k(x,y) = exp(-||x-y|| / tau)
    # ------------------------------------------------------------

    logits_pos = -dist_pos / tau
    logits_neg = -dist_neg / tau

    logits = torch.cat([logits_pos, logits_neg], dim=1)

    # ------------------------------------------------------------
    # Bidirectional normalization (Algorithm 2)
    # ------------------------------------------------------------

    A_row = F.softmax(logits, dim=-1)
    A_col = F.softmax(logits, dim=-2)

    A = torch.sqrt(A_row * A_col)

    # split positive / negative blocks
    A_pos, A_neg = torch.split(
        A,
        [N_real, N_gen],
        dim=1,
    )

    # ------------------------------------------------------------
    # Coupled weighting (CRITICAL)
    # ------------------------------------------------------------

    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    # ------------------------------------------------------------
    # Drifting field
    # ------------------------------------------------------------

    drift_pos = W_pos @ phi_real
    drift_neg = W_neg @ phi_gen

    V = drift_pos - drift_neg

    # ------------------------------------------------------------
    # Fixed-point target (Eq. 6)
    # ------------------------------------------------------------

    target = (phi_gen + V).detach()

    loss = F.mse_loss(phi_gen, target)

    if not torch.isfinite(loss):
        raise TrainingDivergedException(
            f"Non-finite drift loss: {loss.item()}"
        )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    with torch.no_grad():

        row_entropy = (
            -(A_row * (A_row + 1e-30).log())
            .sum(dim=-1)
            .mean()
        )

        stats = {
            "loss": loss.item(),
            "v_norm": V.norm(dim=-1).mean().item(),
            "attn_entropy": row_entropy.item(),
            "pos_mass_frac": (
                A_pos.sum(dim=1)
                / (A_pos.sum(dim=1) + A_neg.sum(dim=1)).clamp_min(1e-8)
            ).mean().item(),
        }

    return loss, stats