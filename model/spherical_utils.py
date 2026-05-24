"""Spherical geometry utilities for manifold operations.

Implements operations on the positive orthant of the sphere S^3, including
normalization, exponential/logarithmic maps, and geodesic distances for
atom type embeddings.
"""

import torch


def sphere_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors onto the unit sphere S^D.

    Args:
        x: Input tensor with any shape ending in dimension D.
        eps: Small constant for numerical stability.

    Returns:
        Normalized vectors with unit norm along last dimension.
    """
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def probs_to_sphere(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map probability simplex to positive orthant of sphere via square-root map.
        phi(p) = sqrt(p)

    Args:
        p: Probability vectors [N_atoms, D] summing to 1 along last dimension.
        eps: Numerical stability constant.

    Returns:
        Sphere vectors [N_atoms, D] on S^D_+.
    """
    p = p.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    x = torch.sqrt(p)
    return sphere_normalize(x, eps)


def sphere_to_probs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map sphere vectors to probability simplex via squaring map.
        p_i = x_i^2

    Args:
        x: Sphere vectors [N_atoms, D] on S^D_+.
        eps: Numerical stability constant.

    Returns:
        Probability vectors [N_atoms, D] summing to 1 along last dimension.
    """
    x = sphere_normalize(x, eps).clamp_min(0.0)
    p = x.pow(2)
    return p / p.sum(dim=-1, keepdim=True).clamp_min(eps)


def sphere_project_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project vector v onto the tangent space of the sphere at point x positionwise.

    Args:
        x: Point on the sphere, shape [N_atoms, D].
        v: Vector to project, shape [N_atoms, D].

    Returns:
        Tangent vector at x, shape [N_atoms, D].
    """
    return v - (x * v).sum(dim=-1, keepdim=True) * x


def product_tangent_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute product metric norm of tangent vectors.
        ||v||_g^2 = sum_l ||v_l||_2^2

    Args:
        v: Tangent vectors [N_atoms, D].
        eps: Numerical stability constant.

    Returns:
        Norms with shape [N_atoms, 1].
    """
    return torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)


def sphere_log(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Logarithmic map on the sphere S^D (inverse of exponential map).

    Computes the tangent vector at x that points toward y.

    Args:
        x: Base point on sphere [N_atoms, D].
        y: Target point on sphere [N_atoms, D].
        eps: Numerical stability constant.

    Returns:
        Tangent vector at x [N_atoms, D].
    """
    x = sphere_normalize(x, eps)
    y = sphere_normalize(y, eps)

    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)  # [..., L, 1]

    u = y - dot * x
    u_norm = u.norm(dim=-1, keepdim=True)

    scale = theta / u_norm.clamp_min(eps)
    out = scale * u

    small = theta < 1e-5
    first_order = sphere_project_tangent(x, y - x)
    out = torch.where(small, first_order, out)
    return sphere_project_tangent(x, out)


def sphere_exp(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map on the sphere S^D.

    Moves from point x in direction v (tangent vector) a distance ||v||.

    Args:
        x: Base point on sphere [N_atoms, D], normalized.
        v: Tangent vector at x [N_atoms, D].
        eps: Numerical stability constant.

    Returns:
        Point on sphere [N_atoms, D].
    """
    x = sphere_normalize(x, eps)
    v = sphere_project_tangent(x, v)

    v_norm = v.norm(dim=-1, keepdim=True)
    direction = v / v_norm.clamp_min(eps)

    out = torch.cos(v_norm) * x + torch.sin(v_norm) * direction
    approx = sphere_normalize(x + v, eps)

    out = torch.where(v_norm < 1e-5, approx, out)
    return sphere_normalize(out, eps)


def geodesic_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute geodesic distance between points on the sphere.

    Args:
        x: Points on sphere [*, D].
        y: Points on sphere [*, D].
        eps: Numerical stability constant.

    Returns:
        Distances along great circles [N_atoms].
    """
    x = sphere_normalize(x, eps)
    y = sphere_normalize(y, eps)

    dot = (x * y).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(dot)  # [N_mol, max_atoms]
