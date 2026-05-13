from typing import Tuple

import torch
import torch.nn.functional as F
import wandb

from . import TrainingDivergedException


def _attention_weighted_field(
    phi_gen_w: torch.Tensor,
    phi_real_w: torch.Tensor,
    dist_pos_w: torch.Tensor,
    dist_neg_w: torch.Tensor,
    tau_eff: float,
    weighting: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bidirectionally-normalized attention + drifting field.

    weighting="coupled":      W_pos = A_pos * ΣA_neg,  W_neg = A_neg * ΣA_pos
    weighting="inverse_attn": W_pos = A_pos / ΣA_pos,  W_neg = A_neg / ΣA_neg

    Returns (V, A_row, A_pos, A_neg).
    """
    logit = torch.cat([-dist_pos_w / tau_eff, -dist_neg_w / tau_eff], dim=1)
    A_row = F.softmax(logit, dim=-1)
    A_col = F.softmax(logit, dim=-2)
    A = torch.sqrt(torch.clamp(A_row * A_col, min=1e-30))

    N_real = phi_real_w.shape[0]
    N_gen = phi_gen_w.shape[0]
    A_pos, A_neg = torch.split(A, [N_real, N_gen], dim=1)

    if weighting == "coupled":
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)
    else:  # inverse_attn
        W_pos = A_pos / A_pos.sum(dim=1, keepdim=True).clamp_min(1e-5)
        W_neg = A_neg / A_neg.sum(dim=1, keepdim=True).clamp_min(1e-5)

    V = W_pos @ phi_real_w - W_neg @ phi_gen_w
    return V, A_row, A_pos, A_neg


def _shared_embedding_stats(
    phi_gen_w: torch.Tensor,
    phi_real_w: torch.Tensor,
    dist_pos_raw: torch.Tensor,
) -> dict[str, float]:
    """Geometric stats shared across all drift loss variants."""
    gen_norms = phi_gen_w.norm(dim=-1)
    real_norms = phi_real_w.norm(dim=-1)
    phi_gen_unit = F.normalize(phi_gen_w, dim=-1)
    phi_real_unit = F.normalize(phi_real_w, dim=-1)
    cos_sim = (phi_gen_unit @ phi_real_unit.T).max(dim=1).values
    return {
        "phi_gen_norm_mean": gen_norms.mean().item(),
        "phi_gen_norm_std": gen_norms.std().item(),
        "phi_real_norm_mean": real_norms.mean().item(),
        "cosine_sim_to_nn": cos_sim.mean().item(),
        "nn_l2_distance": dist_pos_raw.min(dim=1).values.mean().item(),
    }


def original_compute_drift_loss(
    phi_gen: torch.Tensor,
    phi_real: torch.Tensor,
    temperatures: tuple[float, ...] = (0.02, 0.05, 0.2),
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Faithful PyTorch translation of Algorithm 2 from
    'Generative Modeling via Drifting' — coupled attention weighting, multi-tau.

    - targets = [stop_grad(gen), phi_real]; gen acts as its own negatives
    - scale = dist.mean(); scale_inputs = scale / sqrt(D) normalises coords to O(1)
    - attention uses dist / scale as input (not dist / scale_inputs)
    - per-tau force normalisation: force_scale = sqrt(mean(force^2))

    Args:
        phi_gen:      [N_gen, D]
        phi_real:     [N_real, D]
        temperatures: kernel bandwidths (tau)

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

    dist = torch.cdist(old_gen, targets)  # [N_gen, N_targets]

    all_dists = dist.flatten()
    valid_mask = torch.isfinite(all_dists) & (all_dists < 1e5)
    valid = all_dists[valid_mask]

    if valid.numel() == 0:
        raise TrainingDivergedException("No valid distances found; loss is unstable.")

    scale = valid.mean().detach().clamp(min=1e-3)
    scale_inputs = (scale / (D**0.5)).clamp(min=1e-3)

    old_gen_scaled = old_gen / scale_inputs  # [N_gen, D]
    phi_real_scaled = phi_real / scale_inputs  # [N_real, D]

    # Attention distances: normalise by scale (not scale_inputs)
    dist_pos_normed = dist[:, N_gen:] / scale  # [N_gen, N_real]
    dist_neg_normed = dist[:, :N_gen].clone() / scale  # [N_gen, N_gen]
    dist_neg_normed.fill_diagonal_(1e8)  # mask self-connections

    stats: dict[str, float] = {}
    with torch.no_grad():
        stats["scale_S"] = scale_inputs.item()
        stats.update(
            _shared_embedding_stats(old_gen_scaled, phi_real_scaled, dist[:, N_gen:])
        )
        stats["invalid_dist_frac"] = (~valid_mask).float().mean().item()

    V_across_taus = torch.zeros_like(old_gen_scaled)

    for tau in temperatures:
        tau_key = str(tau).replace(".", "_")

        V_tau, A_row, A_pos, A_neg = _attention_weighted_field(
            old_gen_scaled,
            phi_real_scaled,
            dist_pos_normed,
            dist_neg_normed,
            tau,
            "coupled",
        )

        f_norm_val = (V_tau**2).mean()
        force_scale = torch.sqrt(f_norm_val.clamp(min=1e-8)).detach()
        V_tau_norm = V_tau / force_scale
        V_across_taus += V_tau_norm

        with torch.no_grad():
            row_entropy = -(A_row * (A_row + 1e-30).log()).sum(dim=-1).mean()
            row_entropy_uniform = torch.log(
                torch.tensor(N_targets, device=A_row.device, dtype=torch.float)
            )
            pos_mass = A_pos.sum(dim=1)
            neg_mass = A_neg.sum(dim=1)
            stats[f"attn_entropy_{tau_key}"] = row_entropy.item()
            stats[f"attn_entropy_rel_{tau_key}"] = (
                row_entropy / row_entropy_uniform
            ).item()
            stats[f"force_scale_{tau_key}"] = force_scale.item()
            stats[f"v_norm_{tau_key}"] = V_tau_norm.norm(dim=-1).mean().item()
            stats[f"frac_zero_dists_{tau_key}"] = (A_row == 0).float().mean().item()
            stats[f"attn_pos_mass_frac_{tau_key}"] = (
                (pos_mass / (pos_mass + neg_mass).clamp_min(1e-8)).mean().item()
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


def compute_inverse_attn_drift_loss(
    phi_gen: torch.Tensor,
    phi_real: torch.Tensor,
    temperatures: list[float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Normalized drifting field loss (inverse attention weighting, multi-tau).

    Returns (loss, stats). Raises TrainingDivergedException on non-finite loss.
    """
    phi_gen = torch.nan_to_num(phi_gen.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    phi_real = torch.nan_to_num(phi_real.float(), nan=0.0, posinf=1e4, neginf=-1e4)

    N_gen, D = phi_gen.shape
    N_real = phi_real.shape[0]
    N_targets = N_gen + N_real

    dist_pos = torch.cdist(phi_gen, phi_real)
    dist_neg = torch.cdist(phi_gen, phi_gen)
    dist_neg.fill_diagonal_(1e6)

    all_dists = torch.cat([dist_pos.flatten(), dist_neg.flatten()])
    valid_mask = torch.isfinite(all_dists) & (all_dists < 1e5)
    valid_dists = all_dists[valid_mask]

    if valid_dists.numel() == 0:
        S = torch.tensor(1.0, device=phi_gen.device, dtype=phi_gen.dtype)
    else:
        S = (valid_dists.mean() / (D**0.5)).detach()
    S = S.clamp(min=1e-5, max=1e3)

    phi_gen_w = phi_gen / S
    phi_real_w = phi_real / S
    dist_pos_w = dist_pos / S
    dist_neg_w = dist_neg / S

    stats: dict[str, float] = {}
    with torch.no_grad():
        stats["scale_S"] = S.item()
        stats["invalid_dist_frac"] = (~valid_mask).float().mean().item()
        stats.update(_shared_embedding_stats(phi_gen_w, phi_real_w, dist_pos))

    aggregated_v = torch.zeros_like(phi_gen_w)

    for tau in temperatures:
        tau_key = str(tau).replace(".", "_")
        tau_eff = tau * (D**0.5)

        V_tau, A_row, A_pos, A_neg = _attention_weighted_field(
            phi_gen_w, phi_real_w, dist_pos_w, dist_neg_w, tau_eff, "inverse_attn"
        )

        lambda_tau = (
            torch.sqrt(((V_tau**2).sum(dim=-1).mean() / D).clamp(min=1e-10))
            .detach()
            .clamp(min=1e-5, max=1e3)
        )
        V_tau_norm = V_tau / lambda_tau
        aggregated_v += V_tau_norm

        with torch.no_grad():
            row_entropy = -(A_row * (A_row + 1e-30).log()).sum(dim=-1).mean()
            row_entropy_uniform = torch.log(
                torch.tensor(N_targets, device=A_row.device, dtype=torch.float)
            )
            pos_mass = A_pos.sum(dim=1)
            neg_mass = A_neg.sum(dim=1)
            stats[f"attn_entropy_{tau_key}"] = row_entropy.item()
            stats[f"attn_entropy_rel_{tau_key}"] = (
                row_entropy / row_entropy_uniform
            ).item()
            stats[f"lambda_{tau_key}"] = lambda_tau.item()
            stats[f"v_norm_{tau_key}"] = V_tau_norm.norm(dim=-1).mean().item()
            stats[f"frac_zero_dists_{tau_key}"] = (A_row == 0).float().mean().item()
            stats[f"attn_pos_mass_frac_{tau_key}"] = (
                (pos_mass / (pos_mass + neg_mass).clamp_min(1e-8)).mean().item()
            )

    target = (phi_gen_w + aggregated_v).detach()
    loss = F.mse_loss(phi_gen_w, target)

    if not torch.isfinite(loss):
        raise TrainingDivergedException(
            f"Non-finite loss ({loss.item()!r}). "
            f"phi_gen range: [{phi_gen.min().item():.3g}, {phi_gen.max().item():.3g}], "
            f"phi_real range: [{phi_real.min().item():.3g}, {phi_real.max().item():.3g}], "
            f"S={S.item():.3g}"
        )

    return loss, stats


def compute_norm_based_drift_loss(
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

    N_gen, D = phi_gen.shape

    old_gen = phi_gen.detach()
    targets = torch.cat([old_gen, phi_real], dim=0)  # [N_gen + N_real, D]

    dist = torch.cdist(old_gen, targets)  # [N_gen, N_gen + N_real]

    all_dists = dist.flatten()
    valid_mask = torch.isfinite(all_dists) & (all_dists < 1e5)
    valid = all_dists[valid_mask]

    if valid.numel() == 0:
        raise TrainingDivergedException("No valid distances found; loss is unstable.")

    scale = valid.mean().detach().clamp(min=1e-3)
    scale_inputs = (scale / (D**0.5)).clamp(min=1e-3)

    old_gen_scaled = old_gen / scale_inputs  # [N_gen, D]
    phi_real_scaled = phi_real / scale_inputs  # [N_real, D]

    gen_norms = (old_gen_scaled**2).sum(dim=1)  # [N_gen]
    real_norms = (phi_real_scaled**2).sum(dim=1)  # [N_real]

    diff_pos = gen_norms[:, None] - real_norms[None, :]  # [N_gen, N_real]
    diff_neg = gen_norms[:, None] - gen_norms[None, :]  # [N_gen, N_gen]
    diff_neg = diff_neg + torch.eye(N_gen, device=phi_gen.device) * 1e6

    stats = {}
    with torch.no_grad():
        stats["scale_S"] = scale_inputs.item()
        stats.update(
            _shared_embedding_stats(old_gen_scaled, phi_real_scaled, dist[:, N_gen:])
        )
        stats["invalid_dist_frac"] = (~valid_mask).float().mean().item()

        phi_gen_unit = F.normalize(old_gen, dim=-1)
        triu_idx = torch.triu_indices(N_gen, N_gen, offset=1, device=phi_gen.device)
        gen_cos_sim = (phi_gen_unit @ phi_gen_unit.T)[triu_idx[0], triu_idx[1]]
        gen_l2_dist = dist[:, :N_gen][triu_idx[0], triu_idx[1]]
        stats["gen_pairwise_cos_sim_hist"] = wandb.Histogram(
            gen_cos_sim.float().cpu().numpy()
        )
        stats["gen_pairwise_l2_dist_hist"] = wandb.Histogram(
            gen_l2_dist.float().cpu().numpy()
        )

    V_across_taus = torch.zeros_like(old_gen_scaled)

    for tau in temperatures:
        tau_key = str(tau).replace(".", "_")

        kernel_pos = torch.exp(-(diff_pos**2) / tau)  # [N_gen, N_real]
        kernel_neg = torch.exp(-(diff_neg**2) / tau)  # [N_gen, N_gen]

        Z_p = kernel_pos.sum(dim=1).clamp(min=1e-8)  # [N_gen]
        Z_q = kernel_neg.sum(dim=1).clamp(min=1e-8)  # [N_gen]

        grad_pos = (kernel_pos * diff_pos).sum(dim=1)  # [N_gen]
        grad_neg = (kernel_neg * diff_neg).sum(dim=1)  # [N_gen]

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


def compute_position_drift_loss(
    pos_gen: torch.Tensor,
    pos_real: torch.Tensor,
    gen_batch_vec: torch.Tensor,
    real_batch_vec: torch.Tensor,
    temperatures: list[float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Drifting field loss directly in 3D position space, treating each molecule as
    one sample from the distribution.

    Molecules are padded into flat position vectors. Every generated molecule is
    attracted to all real molecules and repelled from all other generated molecules.
    """
    pos_gen = pos_gen.float()
    pos_real = pos_real.float()
    max_nodes = max(
        _max_nodes_per_graph(gen_batch_vec),
        _max_nodes_per_graph(real_batch_vec),
    )
    gen_mol, gen_mask = _positions_to_padded_molecules(
        pos_gen, gen_batch_vec, max_nodes=max_nodes
    )
    real_mol, _ = _positions_to_padded_molecules(
        pos_real, real_batch_vec, max_nodes=max_nodes
    )

    N_gen, D = gen_mol.shape
    N_real = real_mol.shape[0]
    N_targets = N_gen + N_real

    old_gen = gen_mol.detach()
    dist_pos = torch.cdist(old_gen, real_mol)
    dist_neg = torch.cdist(old_gen, old_gen)
    dist_neg.fill_diagonal_(1e6)

    all_dists = torch.cat([dist_pos.flatten(), dist_neg.flatten()])
    valid_mask = torch.isfinite(all_dists) & (all_dists < 1e5)
    valid_dists = all_dists[valid_mask]

    if valid_dists.numel() == 0:
        raise TrainingDivergedException(
            "No valid position distances found; loss is unstable."
        )

    scale = (valid_dists.mean() / (D**0.5)).detach().clamp(min=1e-5, max=1e3)
    old_gen_scaled = old_gen / scale
    real_mol_scaled = real_mol / scale
    dist_pos_scaled = dist_pos / scale
    dist_neg_scaled = dist_neg / scale

    stats: dict[str, float] = {}
    with torch.no_grad():
        pos_gen_norms = pos_gen.norm(dim=-1)
        pos_real_norms = pos_real.norm(dim=-1)
        stats["pos_scale_S"] = scale.item()
        stats["pos_num_gen_molecules"] = float(N_gen)
        stats["pos_num_real_molecules"] = float(N_real)
        stats["pos_gen_norm_mean"] = pos_gen_norms.mean().item()
        stats["pos_gen_norm_std"] = pos_gen_norms.std().item()
        stats["pos_real_norm_mean"] = pos_real_norms.mean().item()
        stats["pos_mol_nn_l2_distance"] = dist_pos.min(dim=1).values.mean().item()
        stats["pos_invalid_dist_frac"] = (~valid_mask).float().mean().item()

    aggregated_v = torch.zeros_like(old_gen_scaled)

    for tau in temperatures:
        tau_key = str(tau).replace(".", "_")
        tau_eff = tau * (D**0.5)

        V_tau, A_row, A_pos, A_neg = _attention_weighted_field(
            old_gen_scaled,
            real_mol_scaled,
            dist_pos_scaled,
            dist_neg_scaled,
            tau_eff,
            "inverse_attn",
        )

        force_scale = (
            torch.sqrt(((V_tau**2).sum(dim=-1).mean() / D).clamp(min=1e-10))
            .detach()
            .clamp(min=1e-5, max=1e3)
        )
        V_tau_norm = V_tau / force_scale
        aggregated_v += V_tau_norm

        with torch.no_grad():
            row_entropy = -(A_row * (A_row + 1e-30).log()).sum(dim=-1).mean()
            row_entropy_uniform = torch.log(
                torch.tensor(N_targets, device=A_row.device, dtype=torch.float)
            )
            pos_mass = A_pos.sum(dim=1)
            neg_mass = A_neg.sum(dim=1)
            stats[f"pos_attn_entropy_{tau_key}"] = row_entropy.item()
            stats[f"pos_attn_entropy_rel_{tau_key}"] = (
                row_entropy / row_entropy_uniform
            ).item()
            stats[f"pos_force_scale_{tau_key}"] = force_scale.item()
            stats[f"pos_v_norm_{tau_key}"] = V_tau_norm.norm(dim=-1).mean().item()
            stats[f"pos_attn_real_mass_frac_{tau_key}"] = (
                (pos_mass / (pos_mass + neg_mass).clamp_min(1e-8)).mean().item()
            )

    target = (old_gen_scaled + aggregated_v).detach()
    gen_scaled = gen_mol / scale
    coord_mask = gen_mask.unsqueeze(-1).expand(-1, -1, pos_gen.shape[-1])
    coord_mask = coord_mask.flatten(start_dim=1)
    loss = F.mse_loss(gen_scaled[coord_mask], target[coord_mask])

    if not torch.isfinite(loss):
        raise TrainingDivergedException(
            f"Non-finite position loss ({loss.item()!r}). "
            f"pos_gen range: [{pos_gen.min().item():.3g}, {pos_gen.max().item():.3g}], "
            f"pos_real range: [{pos_real.min().item():.3g}, {pos_real.max().item():.3g}], "
            f"scale={scale.item():.3g}"
        )

    return loss, stats


def _positions_to_padded_molecules(
    pos: torch.Tensor,
    batch_vec: torch.Tensor,
    max_nodes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    graph_ids = torch.unique(batch_vec, sorted=True)
    counts = torch.stack([(batch_vec == graph_id).sum() for graph_id in graph_ids])
    if max_nodes is None:
        max_nodes = int(counts.max().item())

    padded = pos.new_zeros((graph_ids.numel(), max_nodes, pos.shape[-1]))
    mask = torch.zeros(
        (graph_ids.numel(), max_nodes), dtype=torch.bool, device=pos.device
    )

    for i, graph_id in enumerate(graph_ids):
        graph_pos = pos[batch_vec == graph_id]
        n = min(graph_pos.shape[0], max_nodes)
        padded[i, :n] = graph_pos[:n]
        mask[i, :n] = True

    return padded.flatten(start_dim=1), mask


def _max_nodes_per_graph(batch_vec: torch.Tensor) -> int:
    graph_ids = torch.unique(batch_vec, sorted=True)
    counts = torch.stack([(batch_vec == graph_id).sum() for graph_id in graph_ids])
    return int(counts.max().item())


def positions_to_flat_molecules(pos, batch_vec, max_nodes=19):
    graph_ids = torch.unique(batch_vec, sorted=True)
    out = pos.new_zeros((graph_ids.numel(), max_nodes, 3))

    for i, graph_id in enumerate(graph_ids):
        p = pos[batch_vec == graph_id]
        n = min(p.shape[0], max_nodes)
        out[i, :n] = p[:n]

    return out.flatten(start_dim=1)
