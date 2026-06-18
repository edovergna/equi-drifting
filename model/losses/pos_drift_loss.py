"""Drift-based loss for generative molecular modeling.

Implements the core training objective based on computing drift vectors that
align generated molecules to real molecules on both Euclidean and spherical manifolds.
"""

import torch
import torch.nn.functional as F

from ..aligning_utils.align_pos import (
    apply_pairwise_rotation,
    kabsch_rotations,
)

class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""

    pass

def compute_pos_drift_loss(
    gen_pos: torch.Tensor,
    real_pos: torch.Tensor,
    num_atoms: int,
    cfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    # TO BE CHANGED
    """Compute drift loss aligning generated to real molecules on both manifolds.

    Finds optimal permutation and rotation, computes drift vectors on position and
    type manifolds, then measures distance between generated and target positions/types.

    Args:
        gen_pos: Generated positions [batch, num_atoms, 3].
        real_pos: Real positions [batch, num_atoms, 3].
        gen_types_sphere: Generated types on sphere [batch, num_atoms, D].
        real_types: Real types one-hot [batch, num_atoms, num_types].
        num_atoms: Number of atoms (for reshaping).
        chem_refinement: Whether to include chemical loss.
        cfg: Config dict with all hyperparameters.

    Returns:
        Tuple of (loss, dict of diagnostic statistics), where stats is a flat dict of float diagnostics safe to pass directly to self.log().

    Raises:
        TrainingDivergedException: If loss becomes non-finite.
    """

    eps = cfg["eps"]
    eta = cfg["eta"]

    # Reshape for loss
    gen_pos = gen_pos.reshape(-1, num_atoms, 3)
    real_pos = real_pos.reshape(-1, num_atoms, 3)

    N_gen = gen_pos.shape[0]
    N_real = real_pos.shape[0]

    N_atoms = gen_pos.shape[1]
    sqrt_N_a = N_atoms**0.5

    with torch.no_grad():
        R_pos = kabsch_rotations(
            gen_pos, real_pos
        )
        R_neg = kabsch_rotations(
            gen_pos, gen_pos
        )

        aligned_posit_pos = apply_pairwise_rotation(gen_pos, R_pos)
        aligned_posit_neg = apply_pairwise_rotation(gen_pos, R_neg)

        # Distances of shape [N_gen, N_real/N_gen] and Differences of shape [N_gen, N_real/N_gen, N_atoms, 3/5]
        posit_dist_pos, posit_diff_pos = _pairwise_geodesic_distance_and_log(
            aligned_posit_pos, real_pos
        )
        posit_dist_neg, posit_diff_neg = _pairwise_geodesic_distance_and_log(
            aligned_posit_neg, gen_pos
        )

        posit_dist_pos = posit_dist_pos / sqrt_N_a
        posit_dist_neg = posit_dist_neg / sqrt_N_a

        # Ignore self if y_neg is x
        eye = torch.eye(N_gen, device=gen_pos.device, dtype=torch.bool)
        posit_dist_neg = posit_dist_neg.masked_fill(eye, 1e6)

        V_posit_pos = _calc_drift_direction(
            posit_dist_pos,
            posit_diff_pos,
            R_pos,
            sigma=cfg["sigma"],
            eps=eps,
        )
        V_posit_neg = _calc_drift_direction(
            posit_dist_neg,
            posit_diff_neg,
            R_neg,
            sigma=cfg["sigma"],
            eps=eps,
        )

        V_posit = V_posit_pos - V_posit_neg

        V_posit = eta * V_posit

        # Calculate targets for each
        target_posit = gen_pos + V_posit

    # Calculate distances per molecule on each manifold
    molecule_position_dist = (
        ((gen_pos - target_posit) ** 2).sum(dim=-1).sum(dim=-1)
    )  # shape: [N_gen]

    loss = (
        molecule_position_dist
    ).mean()

    if not torch.isfinite(loss):
        raise TrainingDivergedException()

    stats: dict[str, float] = {}
    with torch.no_grad():
        stats["mean_euclidean_distance"] = (molecule_position_dist).mean().item()
        stats["std_euclidean_distance"] = molecule_position_dist.std().item()
        stats["norm_V_posit"] = V_posit.norm(dim=-1).mean().item()
        stats["mean_V_posit_pos"] = V_posit_pos.abs().mean().item()
        stats["mean_V_posit_neg"] = V_posit_neg.abs().mean().item()

    return loss, stats


def _pairwise_geodesic_distance_and_log(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise geodesic distances and log-space differences between molecules.       Assumed that molecules are aligned.

    For Euclidean: Uses standard L2 distance.
    For Spherical: Uses arccos to compute angles on sphere.

    Args:
        x: [N_x, N_y, N_atoms, D] pairwise tensor or [N_x, N_atoms, D] batch tensor.
        y: [N_y, N_atoms, D] reference tensor.
        manifold: "euclidean" or "spherical".
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (distances, tangent_differences).
    """

    diff = x - y[None, :, :, :]  # shape: (N_x, N_y, N_atoms, z)
    sq_dist_per_atom = (diff**2).sum(dim=-1)  # shape: (N_x, N_y, N_atoms)

    sq_distances = sq_dist_per_atom.sum(dim=-1)  # shape: (N_x, N_y)
    return sq_distances, diff


def _calc_drift_direction(dist, diff, R, sigma, eps=1e-8):
    """Calculate drift vector direction from distances and differences.

    Computes kernel-weighted drift vectors and un-applies the alignment rotation
    so the result is in the original (unaligned) coordinate frame.

    Args:
        dist: Distance matrix [N_gen, N_real].
        diff: Difference vectors [N_gen, N_real, N_atoms, D].
        permutation: Atom assignment [N_gen, N_real, N_atoms].
        R: Rotation matrix [N_gen, N_real, 3, 3].
        sigma: Kernel bandwidth.
        eps: Numerical stability constant.
        euclidean: Whether to apply rotation correction (True for Euclidean, False for spherical).

    Returns:
        Drift vectors [N_gen, N_atoms, D].
    """
    kernel = torch.exp(-dist / (2 * sigma**2))  # shape [N_gen, N_real]
    grad_kernel = - (diff * kernel.unsqueeze(-1).unsqueeze(-1)) / (
        sigma**2
    )  # shape [N_gen, N_real, N_atoms, D]

    inv_R = R.transpose(-2, -1)
    grad_kernel = grad_kernel @ inv_R

    V_dir = grad_kernel.sum(dim=1) / (
        kernel.sum(dim=1).clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    )  # shape [N_gen, N_atoms, D]

    return V_dir
