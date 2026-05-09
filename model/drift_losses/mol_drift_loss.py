import torch
from ..mol_kernel import molecule_kernel
from ..spherical_utils import sphere_exp, geodesic_distance


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""


def compute_molecule_based_drift_loss(
    pos_gen: torch.Tensor,
    x_gen: torch.Tensor,
    pos_real: torch.Tensor,
    x_real: torch.Tensor,
    gen_index: torch.Tensor,
    real_index: torch.Tensor,
    eps: float = 1e-8,
    sigma_r: float = 1.0,
    sigma_a: float = 0.5,
    eta_pos: float = 1.0,
    eta_type: float = 1.0,
    weight_pos: float = 1.0,
    weight_type: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Drifting field loss directly on 3D molecules.
    Assumes that the x_gen and x_real are already mapped to the spherical space.

    Returns (loss, stats) where stats is a flat dict of float diagnostics safe to
    pass directly to self.log(). Raises TrainingDivergedException on non-finite loss.
    """
    stats: dict[str, float] = {}

    # Detach to create clean leaf variables for drift field computation.
    # torch.enable_grad() is required because validation_step runs under torch.no_grad().
    # Two separate autograd.grad calls on the same graph would fail after the first
    # frees intermediate buffers, so both inputs are requested in one call each.
    with torch.enable_grad():
        pos_leaf = pos_gen.detach().requires_grad_(True)
        x_leaf = x_gen.detach().requires_grad_(True)

        kernel_pos = molecule_kernel(
            pos_leaf,
            x_leaf,
            pos_real,
            x_real,
            gen_index,
            real_index,
            sigma_r=sigma_r,
            sigma_a=sigma_a,
        )
        kernel_neg = molecule_kernel(
            pos_leaf,
            x_leaf,
            pos_real=pos_leaf,
            x_real=x_leaf,
            gen_index=gen_index,
            real_index=gen_index,
            sigma_r=sigma_r,
            sigma_a=sigma_a,
            same_samples=True,
        )

        N_pos, N_neg = kernel_pos.shape[1], kernel_neg.shape[1]
        raw_exp_pos = kernel_pos.sum(dim=1) / N_pos
        raw_exp_neg = kernel_neg.sum(dim=1) / N_neg
        exp_pos = torch.clamp(raw_exp_pos, min=1e-8)
        exp_neg = torch.clamp(raw_exp_neg, min=1e-8)

        # Score-style drift: use gradients of log kernel expectations rather
        # than raw expectations, with diagnostics tracking clamp saturation.
        log_exp_pos = torch.log(exp_pos)
        log_exp_neg = torch.log(exp_neg)

        # Both inputs requested in one call to avoid retain_graph issues
        grad_pos_pos, grad_types_pos = torch.autograd.grad(
            outputs=log_exp_pos.sum(), inputs=[pos_leaf, x_leaf]
        )
        grad_pos_neg, grad_types_neg = torch.autograd.grad(
            outputs=log_exp_neg.sum(), inputs=[pos_leaf, x_leaf]
        )
    
    v_positions = grad_pos_pos - grad_pos_neg
    v_types = grad_types_pos - grad_types_neg

    # Targets are derived from the leaf copies (detached) so the final loss gradient
    # flows only through pos_gen / x_gen back to the model parameters.
    target_positions = (pos_leaf + eta_pos * v_positions).detach()

    # Target for atom types is through the exponential mapping of the spherical space:
    target_types = sphere_exp(x_leaf, eta_type * v_types, eps).detach()

    # Next, calculate loss per Riemannian manifold, then combine weighted squared
    # distances.
    # Distance metric for the euclidean space
    euclidean_distances = ((pos_gen - target_positions) ** 2).sum(-1)

    # Calculating distances for atom types on the sphere
    spherical_distances = geodesic_distance(x_gen, target_types, eps) ** 2

    weighted_euclidean_distances = weight_pos * euclidean_distances
    weighted_spherical_distances = weight_type * spherical_distances
    combined_distances = weighted_euclidean_distances + weighted_spherical_distances

    # Calculate final loss as the mean atom distance per molecule, then average
    # molecules so each molecule contributes equally regardless of atom count.
    num_molecules = int(gen_index.max().item()) + 1
    sum_per_molecule = torch.zeros(
        num_molecules,
        device=combined_distances.device,
        dtype=combined_distances.dtype,
    )
    sum_per_molecule.scatter_add_(0, gen_index, combined_distances)
    counts_per_molecule = torch.zeros(
        num_molecules,
        device=combined_distances.device,
        dtype=combined_distances.dtype,
    )
    counts_per_molecule.scatter_add_(0, gen_index, torch.ones_like(combined_distances))
    mean_per_molecule = sum_per_molecule / counts_per_molecule.clamp_min(1.0)
    loss = mean_per_molecule.mean()

    if not torch.isfinite(loss):
        raise TrainingDivergedException(f"Non-finite loss ({loss.item()!r}). ")

    with torch.no_grad():
        # Size of euclidean and spherical distances
        stats["average_euclidean_distance"] = euclidean_distances.mean().item()
        stats["average_spherical_distance"] = spherical_distances.mean().item()
        stats["average_weighted_euclidean_distance"] = (
            weighted_euclidean_distances.mean().item()
        )
        stats["average_weighted_spherical_distance"] = (
            weighted_spherical_distances.mean().item()
        )
        stats["std_euclidean_distance"] = euclidean_distances.std().item()
        stats["std_spherical_distance"] = spherical_distances.std().item()
        stats.update(_kernel_stats("kernel/pos", kernel_pos))
        stats.update(_kernel_stats("kernel/neg", kernel_neg))
        stats.update(_expectation_stats("kernel/exp_pos", raw_exp_pos))
        stats.update(_expectation_stats("kernel/exp_neg", raw_exp_neg))
        stats["kernel/exp_pos_clamped_frac"] = (
            raw_exp_pos < 1e-8
        ).float().mean().item()
        stats["kernel/exp_neg_clamped_frac"] = (
            raw_exp_neg < 1e-8
        ).float().mean().item()
        stats.update(_norm_stats("drift/pos_norm", v_positions.norm(dim=-1)))
        stats.update(_norm_stats("drift/type_norm", v_types.norm(dim=-1)))
        stats["kernel/sigma_r"] = float(sigma_r)
        stats["kernel/sigma_a"] = float(sigma_a)
        stats["drift/eta_pos"] = float(eta_pos)
        stats["drift/eta_type"] = float(eta_type)
        stats["loss/weight_pos"] = float(weight_pos)
        stats["loss/weight_type"] = float(weight_type)
        stats.update(_norm_stats("loss/atoms_per_molecule", counts_per_molecule))

    return loss, stats


def _kernel_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().flatten()
    if flat.numel() == 0:
        return {}

    return {
        f"{prefix}_mean": flat.mean().item(),
        f"{prefix}_max": flat.max().item(),
        f"{prefix}_std": flat.std(unbiased=False).item(),
    }


def _expectation_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().flatten()
    if flat.numel() == 0:
        return {}

    return {
        f"{prefix}_mean": flat.mean().item(),
        f"{prefix}_min": flat.min().item(),
        f"{prefix}_median": flat.quantile(0.50).item(),
        f"{prefix}_p05": flat.quantile(0.05).item(),
    }


def _norm_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().flatten()
    if flat.numel() == 0:
        return {}

    return {
        f"{prefix}_mean": flat.mean().item(),
        f"{prefix}_p95": flat.quantile(0.95).item(),
        f"{prefix}_max": flat.max().item(),
    }
