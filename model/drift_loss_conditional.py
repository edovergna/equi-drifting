"""Position-only drift loss for conditional conformer generation.

In the conditional setting, atom types are fixed inputs (not generated),
so only position drift is computed. The existing Kabsch + Hungarian
alignment is reused unchanged: types are projected to the sphere and
used as matching cues in the Hungarian cost matrix, but no spherical
drift loss is computed for them.
"""

import torch

from .align import (
    apply_pairwise_rotation,
    find_rotation_and_permutation,
    permute_generated_to_real_order,
)
from .chem_loss import compute_chem_loss
from .drift_loss import (
    TrainingDivergedException,
    _calc_drift_direction,
    _pairwise_geodesic_distance_and_log,
)
from .spherical_utils import probs_to_sphere


def compute_conditional_drift_loss(
    gen_pos: torch.Tensor,
    real_pos: torch.Tensor,
    atom_types: torch.Tensor,
    num_atoms: int,
    chem_refinement: bool,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute position-only drift loss for conditional conformer generation.

    Atom types are provided as fixed conditioning — they are used for
    alignment (Hungarian cost) but not as drift targets. Only atomic
    positions are optimized.

    The positive drift attracts generated positions towards the real
    QM9 conformers; the negative drift repels generated molecules from
    each other to prevent mode collapse. Both use Kabsch + Hungarian
    alignment to handle rotational and permutation symmetries.

    Args:
        gen_pos: Generated positions [N_total, 3] or [B, N, 3].
        real_pos: Target QM9 positions, same shape as gen_pos.
        atom_types: Fixed one-hot atom types [N_total, num_types] or [B, N, num_types].
            These are the SAME for generated and real molecules (conditional setting).
        num_atoms: Number of atoms per molecule (all molecules in a batch share this).
        chem_refinement: Whether to add a chemical validity penalty term.
        cfg: Hyperparameter dict. Required keys:
            p_sigma, p_eta, scale_eucl, eps, max_iter, p_tol, p_weight, t_weight.
            Optional: lambda_clash, lambda_valence_excess, lambda_hydrogen_valence,
            clash_threshold, bond_temperature (used if chem_refinement=True).

    Returns:
        Tuple of (scalar loss tensor, diagnostics dict with float values).

    Raises:
        TrainingDivergedException: If the loss becomes non-finite.
    """
    eps = cfg["eps"]
    p_eta = cfg["p_eta"]
    scale_eucl = cfg["scale_eucl"]

    # Reshape to [B, N, D] for pairwise drift computation.
    gen_pos = gen_pos.reshape(-1, num_atoms, 3)
    real_pos = real_pos.reshape(-1, num_atoms, 3)
    atom_types = atom_types.reshape(-1, num_atoms, atom_types.shape[-1]).float()

    N_gen = gen_pos.shape[0]
    N_atoms = gen_pos.shape[1]
    sqrt_N_a = N_atoms ** 0.5

    # Project one-hot types to sphere for use as matching cues in alignment.
    # One-hot vertex i maps to sphere basis vector e_i (already unit length),
    # so same-type atoms have zero angular cost and different types have π/2.
    types_sphere = probs_to_sphere(atom_types, eps)

    with torch.no_grad():
        # --- Positive alignment: each gen molecule → all real molecules ---
        permutation_pos, R_pos, _, _ = find_rotation_and_permutation(
            gen_pos, real_pos, types_sphere, types_sphere, cfg
        )
        # --- Negative alignment: each gen molecule → all other gen molecules ---
        permutation_neg, R_neg, _, _ = find_rotation_and_permutation(
            gen_pos, gen_pos, types_sphere, types_sphere, cfg
        )

        aligned_posit_pos = permute_generated_to_real_order(gen_pos, permutation_pos)
        aligned_posit_neg = permute_generated_to_real_order(gen_pos, permutation_neg)

        aligned_posit_pos = apply_pairwise_rotation(aligned_posit_pos, R_pos)
        aligned_posit_neg = apply_pairwise_rotation(aligned_posit_neg, R_neg)

        posit_dist_pos, posit_diff_pos = _pairwise_geodesic_distance_and_log(
            aligned_posit_pos, real_pos, "euclidean", eps
        )
        posit_dist_neg, posit_diff_neg = _pairwise_geodesic_distance_and_log(
            aligned_posit_neg, gen_pos, "euclidean", eps
        )

        # Normalise distances by sqrt(N_atoms), same as joint loss.
        posit_dist_pos = posit_dist_pos / sqrt_N_a
        posit_dist_neg = posit_dist_neg / sqrt_N_a

        # Mask diagonal of negative pairs (gen[i] vs gen[i] is trivially 0).
        eye = torch.eye(N_gen, device=gen_pos.device, dtype=torch.bool)
        posit_dist_neg = posit_dist_neg.masked_fill(eye, 1e6)

        V_posit_pos = _calc_drift_direction(
            posit_dist_pos,
            posit_diff_pos,
            permutation_pos,
            R_pos,
            sigma=cfg["p_sigma"],
            eps=eps,
        )
        V_posit_neg = _calc_drift_direction(
            posit_dist_neg,
            posit_diff_neg,
            permutation_neg,
            R_neg,
            sigma=cfg["p_sigma"],
            eps=eps,
        )

        V_posit = p_eta * (V_posit_pos - V_posit_neg)
        target_posit = gen_pos + V_posit

    # Loss: MSE between generated positions and drift targets.
    molecule_position_dist = (
        ((gen_pos - target_posit) ** 2).sum(dim=-1).sum(dim=-1)
    )  # [N_gen]

    loss = (scale_eucl * molecule_position_dist).mean()

    if chem_refinement:
        # atom_types is one-hot, which is already a valid probability simplex element.
        chem_loss, chem_stats = compute_chem_loss(gen_pos, atom_types, cfg)
        loss = loss + chem_loss

    if not torch.isfinite(loss):
        raise TrainingDivergedException()

    stats: dict[str, float] = {}
    with torch.no_grad():
        stats["mean_euclidean_distance"] = molecule_position_dist.mean().item()
        stats["std_euclidean_distance"] = molecule_position_dist.std().item()
        stats["norm_V_posit"] = V_posit.norm(dim=-1).mean().item()
        stats["mean_V_posit_pos"] = V_posit_pos.abs().mean().item()
        stats["mean_V_posit_neg"] = V_posit_neg.abs().mean().item()

    if chem_refinement:
        stats.update(chem_stats)

    return loss, stats
