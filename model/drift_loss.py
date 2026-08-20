"""Drift-based loss for generative molecular modeling.

Implements the core training objective by comparing generated and real molecules
directly, without rotating or permuting either molecule.
"""

import torch

from .spherical_utils import (
    product_tangent_norm,
    sphere_exp,
    geodesic_distance,
    sphere_normalize,
    sphere_project_tangent,
    sphere_to_probs,
)
from .chem_loss import compute_chem_loss


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""

    pass


def compute_drift_loss(
    gen_pos: torch.Tensor,
    real_pos: torch.Tensor,
    gen_types_sphere: torch.Tensor,
    real_types: torch.Tensor,
    num_atoms: int,
    chem_refinement: bool,
    cfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute an unaligned drift loss on position and atom-type spaces.

    Generated and real molecules are compared in their existing atom order and
    coordinate frame. Attraction and repulsion are empirical means of Gaussian-
    kernel gradients rather than kernel-mass-normalized averages. Atom types use
    spherical geometry by default and Euclidean probability-space geometry when
    ``cfg["spherical_space"]`` is false.

    Args:
        gen_pos: Generated positions [batch, num_atoms, 3].
        real_pos: Real positions [batch, num_atoms, 3].
        gen_types_sphere: Generated atom-type representation [batch, num_atoms,
            D]. This is a sphere embedding in spherical mode and probabilities
            in Euclidean mode. The name is retained for API compatibility.
        real_types: Real types one-hot [batch, num_atoms, num_types].
        num_atoms: Number of atoms (for reshaping).
        chem_refinement: Whether to include chemical loss.
        cfg: Config dict with all hyperparameters.

    Returns:
        Tuple of loss and a flat dictionary of float diagnostics that can be
        passed directly to ``self.log()``.

    Raises:
        TrainingDivergedException: If loss becomes non-finite.
    """
    real_types = real_types.float()

    eps = cfg["eps"]
    p_eta = cfg["p_eta"]
    t_eta = cfg["t_eta"]
    scale_eucl = cfg["scale_eucl"]
    scale_spher = cfg["scale_spher"]
    spherical_space = cfg.get("spherical_space", True)
    types_manifold = "spherical" if spherical_space else "euclidean"

    # Reshape for loss
    gen_pos = gen_pos.reshape(-1, num_atoms, 3)
    real_pos = real_pos.reshape(-1, num_atoms, 3)
    gen_types_sphere = gen_types_sphere.reshape(-1, num_atoms, cfg["num_atom_types"])
    real_types = real_types.reshape(-1, num_atoms, cfg["num_atom_types"])

    N_gen = gen_pos.shape[0]
    N_atoms = gen_pos.shape[1]
    sqrt_N_a = N_atoms**0.5

    with torch.no_grad():
        # Distances have shape [N_gen, N_real/N_gen], while differences have
        # shape [N_gen, N_real/N_gen, N_atoms, 3/num_atom_types]. Molecules
        # remain in their original atom order and coordinate frame.
        posit_dist_pos, posit_diff_pos = _pairwise_geodesic_distance_and_log(
            gen_pos, real_pos, "euclidean", eps
        )
        posit_dist_neg, posit_diff_neg = _pairwise_geodesic_distance_and_log(
            gen_pos, gen_pos, "euclidean", eps
        )

        types_dist_pos, types_diff_pos = _pairwise_geodesic_distance_and_log(
            gen_types_sphere, real_types, types_manifold, eps
        )
        types_dist_neg, types_diff_neg = _pairwise_geodesic_distance_and_log(
            gen_types_sphere, gen_types_sphere, types_manifold, eps
        )

        posit_dist_pos = posit_dist_pos / sqrt_N_a
        posit_dist_neg = posit_dist_neg / sqrt_N_a

        types_dist_pos = types_dist_pos / sqrt_N_a
        types_dist_neg = types_dist_neg / sqrt_N_a

        # Ignore self if y_neg is x
        eye = torch.eye(N_gen, device=gen_pos.device, dtype=torch.bool)
        posit_dist_neg = posit_dist_neg.masked_fill(eye, 1e6)
        types_dist_neg = types_dist_neg.masked_fill(eye, 1e6)

        V_posit_pos = _calc_drift_direction(
            posit_dist_pos,
            posit_diff_pos,
            sigma=cfg["p_sigma"],
        )
        V_posit_neg = _calc_drift_direction(
            posit_dist_neg,
            posit_diff_neg,
            sigma=cfg["p_sigma"],
        )

        V_types_pos = _calc_drift_direction(
            types_dist_pos,
            types_diff_pos,
            sigma=cfg["t_sigma"],
            euclidean=not spherical_space,
        )
        V_types_neg = _calc_drift_direction(
            types_dist_neg,
            types_diff_neg,
            sigma=cfg["t_sigma"],
            euclidean=not spherical_space,
        )

        V_posit = V_posit_pos - V_posit_neg
        V_types = V_types_pos - V_types_neg
        if spherical_space:
            V_types = sphere_project_tangent(gen_types_sphere, V_types)

        V_posit = p_eta * V_posit
        V_types = t_eta * V_types

        # Calculate targets for each
        target_posit = gen_pos + V_posit
        if spherical_space:
            target_types = sphere_exp(gen_types_sphere, V_types, eps)
        else:
            target_types = gen_types_sphere + V_types

    # Calculate distances per molecule on each manifold
    molecule_position_dist = (
        ((gen_pos - target_posit) ** 2).sum(dim=-1).sum(dim=-1)
    )  # shape: [N_gen]
    if spherical_space:
        molecule_types_dist = (
            geodesic_distance(gen_types_sphere, target_types, eps) ** 2
        ).sum(dim=-1)
    else:
        molecule_types_dist = (
            (gen_types_sphere - target_types).pow(2).sum(dim=-1).sum(dim=-1)
        )

    loss = (
        scale_eucl * molecule_position_dist + scale_spher * molecule_types_dist
    ).mean()

    if chem_refinement:
        gen_type_probs = (
            sphere_to_probs(gen_types_sphere, eps)
            if spherical_space
            else gen_types_sphere
        )
        chem_loss, chem_stats = compute_chem_loss(
            gen_pos, gen_type_probs, cfg
        )
        loss = loss + chem_loss

    if not torch.isfinite(loss):
        raise TrainingDivergedException()

    stats: dict[str, float] = {}
    with torch.no_grad():
        types_metric = "spherical" if spherical_space else "euclidean_types"
        type_norm = (
            product_tangent_norm(V_types, eps)
            if spherical_space
            else V_types.norm(dim=-1, keepdim=True)
        )
        type_pos_norm = (
            product_tangent_norm(V_types_pos, eps)
            if spherical_space
            else V_types_pos.norm(dim=-1, keepdim=True)
        )
        type_neg_norm = (
            product_tangent_norm(V_types_neg, eps)
            if spherical_space
            else V_types_neg.norm(dim=-1, keepdim=True)
        )

        stats["mean_euclidean_distance"] = (molecule_position_dist).mean().item()
        stats["std_euclidean_distance"] = molecule_position_dist.std().item()
        stats[f"mean_{types_metric}_distance"] = molecule_types_dist.mean().item()
        stats[f"std_{types_metric}_distance"] = molecule_types_dist.std().item()
        stats["norm_V_posit"] = V_posit.norm(dim=-1).mean().item()
        stats["norm_V_types"] = type_norm.mean().item()
        stats["mean_V_posit_pos"] = V_posit_pos.abs().mean().item()
        stats["mean_V_posit_neg"] = V_posit_neg.abs().mean().item()
        stats["mean_V_types_pos"] = type_pos_norm.mean().item()
        stats["mean_V_types_neg"] = type_neg_norm.mean().item()

    if chem_refinement:
        stats.update(chem_stats)

    return loss, stats


def _pairwise_geodesic_distance_and_log(
    x: torch.Tensor,
    y: torch.Tensor,
    manifold: str = "euclidean",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise geodesic distances and log-space differences.

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

    if manifold == "euclidean":
        diff = x - y[None, :, :, :]  # shape: (N_x, N_y, N_atoms, z)
        sq_dist_per_atom = (diff**2).sum(dim=-1)  # shape: (N_x, N_y, N_atoms)

        sq_distances = sq_dist_per_atom.sum(dim=-1)  # shape: (N_x, N_y)
    elif manifold == "spherical":
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
    else:
        raise ValueError("Undefined manifold.")
    return sq_distances, diff


def _calc_drift_direction(dist, diff, sigma, euclidean=True):
    """Calculate an unnormalized kernel-gradient drift term.

    Each reference molecule contributes its Gaussian-kernel gradient. Their
    empirical mean estimates the distributional expectation; unlike normalized
    drifting, the result is not divided by the total kernel mass.

    Args:
        dist: Distance matrix [N_gen, N_real].
        diff: Difference vectors [N_gen, N_real, N_atoms, D].
        sigma: Kernel bandwidth.
        euclidean: Whether ``diff`` is Euclidean ``x - y`` and therefore needs
            its sign reversed. Spherical differences already point from x to y.

    Returns:
        Drift vectors [N_gen, N_atoms, D].
    """
    kernel = torch.exp(-dist / (2 * sigma**2))  # shape [N_gen, N_real]
    grad_kernel = (diff * kernel.unsqueeze(-1).unsqueeze(-1)) / (
        sigma**2
    )  # shape [N_gen, N_real, N_atoms, D]

    if euclidean:
        grad_kernel = -grad_kernel

    return grad_kernel.mean(dim=1)  # shape [N_gen, N_atoms, D]
