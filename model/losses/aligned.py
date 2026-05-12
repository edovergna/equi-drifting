from typing import Tuple

import torch
import torch.nn.functional as F
import wandb

from . import TrainingDivergedException

def compute_aligning_drift_loss(
    pos_gen: torch.Tensor,
    pos_real: torch.Tensor,
    x_gen_sphere: torch.Tensor,
    x_real: torch.Tensor,
    gen_batch_vec: torch.Tensor,
    real_batch_vec: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:

    return None

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
    real_mol, real_mask = _positions_to_padded_molecules(
        pos_real, real_batch_vec, max_nodes=max_nodes
    )

    N_gen = gen_mol.shape[0]
    N_real = real_mol.shape[0]
    N_targets = N_gen + N_real

    old_gen = gen_mol.detach()
    dist_pos = torch.cdist(old_gen.flatten(start_dim=1), real_mol.flatten(start_dim=1))
    dist_neg = torch.cdist(old_gen.flatten(start_dim=1), old_gen.flatten(start_dim=1))
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

    return padded, mask


def _max_nodes_per_graph(batch_vec: torch.Tensor) -> int:
    graph_ids = torch.unique(batch_vec, sorted=True)
    counts = torch.stack([(batch_vec == graph_id).sum() for graph_id in graph_ids])
    return int(counts.max().item())
