import torch
import torch.nn.functional as F
from ..mol_kernel import molecule_kernel
from ..spherical_utils import (product_tangent_norm, sphere_exp, geodesic_distance)


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""


def compute_molecule_based_drift_loss(
    pos_gen: torch.Tensor,
    x_gen: torch.Tensor,
    pos_real: torch.Tensor,
    x_real: torch.Tensor,
    gen_index: torch.Tensor,
    real_index: torch.Tensor,
    eps: float = 1e-8
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Drifting field loss directly on 3D molecules.
    Assumes that the x_gen and x_real are already mapped to the spherical space.

    Returns (loss, stats) where stats is a flat dict of float diagnostics safe to
    pass directly to self.log(). Raises TrainingDivergedException on non-finite loss.
    """

    # First we calculate the drift field. For this we calculate the kernel with positive and negative samples.
    # We take the gradient of the log of the expectation of the kernel and then subtract to obtain the drifting field.

    with torch.no_grad():
        kernel_pos = molecule_kernel(pos_gen, x_gen, pos_real, x_real, gen_index, real_index)  # shape: [Num_Gen_Mol, Num_Real_Mol]
        kernel_neg = molecule_kernel(pos_gen, x_gen, pos_real=pos_gen, x_real=x_gen, gen_index=gen_index, real_index=real_index)  # shape: [Num_Gen_Mol, Num_Gen_Mol]

        N_pos, N_neg = kernel_pos.shape[1], kernel_neg.shape[1]
        exp_pos = torch.clamp(kernel_pos.sum(dim=1) / N_pos, min=1e-8)
        exp_neg = torch.clamp(kernel_neg.sum(dim=1) / N_neg, min=1e-8)

        # TODO: check up whether it is allowed to use the log 
        log_exp_pos = torch.log(exp_pos)
        log_exp_neg = torch.log(exp_neg)

        # Automatically obtain relevant gradients with regards to the inputs separately
        grad_pos_pos = torch.autograd.grad(outputs=log_exp_pos.sum(), inputs=pos_gen)[0]
        grad_pos_neg = torch.autograd.grad(outputs=log_exp_neg.sum(), inputs=pos_gen)[0]

        grad_types_pos = torch.autograd.grad(outputs=log_exp_pos.sum(), inputs=x_gen)[0]
        grad_types_neg = torch.autograd.grad(outputs=log_exp_neg.sum(), inputs=x_gen)[0]

        # Obtain drift field by subtracting
        v_positions = grad_pos_pos - grad_pos_neg
        v_types = grad_types_pos - grad_types_neg

        # TODO: add possibility of multiplying drifting field with some eta learning rate

        # Can now obtain the targets
        target_positions = (pos_gen + v_positions)

        # Target for atom types is through the exponential mapping of the spherical space:
        # TODO: add step scaling of drifting field for atom types
        # v_types_norm = product_tangent_norm(v_types, eps)
        target_types = sphere_exp(x_gen, v_types, eps)

    # Next, calculate loss per riemannian manifold, then combine by the summing the squared distances:
    # Distance metric for the euclidean space
    euclidean_distances = ((pos_gen - target_positions) ** 2).sum(-1)

    # Calculating distances for atom types on the sphere
    spherical_distances = geodesic_distance(x_gen, target_types, eps) ** 2

    # Combine the distances
    # TODO: add weighting depending on stats
    combined_distances = euclidean_distances + spherical_distances

    # Calculate final loss as the expectation over the distances per molecule
    num_molecules = int(gen_index.max().item()) + 1
    sum_per_molecule = torch.zeros(num_molecules, device=combined_distances.device, dtype=combined_distances.dtype)
    sum_per_molecule.scatter_add_(0, gen_index, combined_distances)
    loss = sum_per_molecule.mean()

    if not torch.isfinite(loss):
        raise TrainingDivergedException(f"Non-finite loss ({loss.item()!r}). ")

    stats: dict[str, float] = {}

    with torch.no_grad():
        # Size of euclidean and spherical distances
        stats["average_euclidean_distance"] = euclidean_distances.mean().item()
        stats["average_spherical_distance"] = spherical_distances.mean().item()
        stats["std_euclidean_distance"] = euclidean_distances.std().item()
        stats["std_spherical_distance"] = spherical_distances.std().item()

    return loss, stats

