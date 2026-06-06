"""Drift-based loss for generative molecular modeling of atomic types.

Implements the core training objective based on computing drift vectors that
align generated molecules to real molecules on spherical manifolds.
"""

import torch
import torch.nn.functional as F

from ..spherical_utils import (
    product_tangent_norm,
    sphere_exp,
    geodesic_distance,
    sphere_normalize,
    sphere_project_tangent,
)

from ..aligning_utils.align_types import (
    hungarian_method_batched,
    permute_generated_to_real_order,
    unpermute_real_order_to_gen_order,
)


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""
    pass


def compute_types_drift_loss(
    gen_types_sphere: torch.Tensor,
    real_types: torch.Tensor,
    num_atoms: int,
    cfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute drift loss aligning generated atom types to real atom types.

    Finds optimal permutation based on atomic types distances, computes drift vectors on
    type manifold, then measures distance between generated and target types.

    Args:
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
    real_types = real_types.float()

    eps = cfg["eps"]
    eta = cfg["eta"]

    # Reshape for loss
    gen_types_sphere = gen_types_sphere.reshape(-1, num_atoms, cfg["num_atom_types"])
    real_types = real_types.reshape(-1, num_atoms, cfg["num_atom_types"])

    N_gen = gen_types_sphere.shape[0]
    N_real = real_types.shape[0]

    N_atoms = gen_types_sphere.shape[1]
    sqrt_N_a = N_atoms**0.5

    with torch.no_grad():
        permutation_pos = hungarian_method_batched(
            gen_types_sphere, real_types, cfg
        )
        permutation_neg = hungarian_method_batched(
            gen_types_sphere, gen_types_sphere, cfg
        )

        aligned_types_pos = permute_generated_to_real_order(
            gen_types_sphere, permutation_pos
        )
        aligned_types_neg = permute_generated_to_real_order(
            gen_types_sphere, permutation_neg
        )

        # Distances of shape [N_gen, N_real/N_gen] and Differences of shape [N_gen, N_real/N_gen, N_atoms, 3/5]
        types_dist_pos, types_diff_pos = _pairwise_geodesic_distance_and_log(
            aligned_types_pos, real_types, eps
        )
        types_dist_neg, types_diff_neg = _pairwise_geodesic_distance_and_log(
            aligned_types_neg, gen_types_sphere, eps
        )

        types_dist_pos = types_dist_pos / sqrt_N_a
        types_dist_neg = types_dist_neg / sqrt_N_a

        # Ignore self if y_neg is x
        eye = torch.eye(N_gen, device=gen_types_sphere.device, dtype=torch.bool)
        types_dist_neg = types_dist_neg.masked_fill(eye, 1e6)

        V_types_pos = _calc_drift_direction(
            types_dist_pos,
            types_diff_pos,
            permutation_pos,
            sigma=cfg["sigma"],
            eps=eps,
        )
        V_types_neg = _calc_drift_direction(
            types_dist_neg,
            types_diff_neg,
            permutation_neg,
            sigma=cfg["sigma"],
            eps=eps,
        )

        V_types = V_types_pos - V_types_neg
        V_types = sphere_project_tangent(gen_types_sphere, V_types)
        V_types = eta * V_types

        # Calculate targets for each
        target_types = sphere_exp(gen_types_sphere, V_types, eps)

    # Calculate distances per molecule on each manifold
    molecule_types_dist = (
        geodesic_distance(gen_types_sphere, target_types, eps) ** 2
    ).sum(
        dim=-1
    )  # shape: [N_gen]

    loss = (
        molecule_types_dist
    ).mean()

    if not torch.isfinite(loss):
        raise TrainingDivergedException()

    stats: dict[str, float] = {}
    with torch.no_grad():
        stats["mean_spherical_distance"] = molecule_types_dist.mean().item()
        stats["std_spherical_distance"] = molecule_types_dist.std().item()
        stats["norm_V_types"] = product_tangent_norm(V_types, eps).mean().item()
        stats["mean_V_types_pos"] = product_tangent_norm(V_types_pos, eps).mean().item()
        stats["mean_V_types_neg"] = product_tangent_norm(V_types_neg, eps).mean().item()

    return loss, stats


def _pairwise_geodesic_distance_and_log(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise geodesic distances and log-space differences between molecules.      
      Assumed that molecules are aligned.

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

    x = sphere_normalize(x, eps)
    y = sphere_normalize(y.unsqueeze(0), eps)

    dot = (x * y).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)  # [N_x, N_y, N_atoms]

    sq_distances = theta.pow(2).sum(dim=-1).clamp_min(eps)  # [N_x, N_y]

    u = y - dot.unsqueeze(-1) * x  # [N_x, N_y, N_atoms, z]
    u_norm = u.norm(dim=-1, keepdim=True)

    scale = theta.unsqueeze(-1) / u_norm.clamp_min(eps)
    out = scale * u

    small = theta.unsqueeze(-1) < 1e-5
    first_order = sphere_project_tangent(x, y - x)

    out = torch.where(small, first_order, out)
    diff = sphere_project_tangent(x, out)  # shape: (N_x, N_y, N_atoms, z)
    
    return sq_distances, diff


def _calc_drift_direction(dist, diff, permutation, sigma, eps=1e-8):
    """Calculate drift vector direction from distances and differences.

    Computes kernel-weighted drift vectors and un-applies the alignment
    so the result is in the original (unaligned) coordinate frame.

    Args:
        dist: Distance matrix [N_gen, N_real].
        diff: Difference vectors [N_gen, N_real, N_atoms, D].
        permutation: Atom assignment [N_gen, N_real, N_atoms].
        sigma: Kernel bandwidth.
        eps: Numerical stability constant.

    Returns:
        Drift vectors [N_gen, N_atoms, D].
    """
    kernel = torch.exp(-dist / (2 * sigma**2))  # shape [N_gen, N_real]
    grad_kernel = (diff * kernel.unsqueeze(-1).unsqueeze(-1)) / (
        sigma**2
    )  # shape [N_gen, N_real, N_atoms, D]

    grad_kernel = unpermute_real_order_to_gen_order(grad_kernel, permutation)

    V_dir = grad_kernel.sum(dim=1) / (
        kernel.sum(dim=1).clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    )  # shape [N_gen, N_atoms, D]

    return V_dir
