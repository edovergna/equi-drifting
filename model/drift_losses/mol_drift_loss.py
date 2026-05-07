import torch
import torch.nn.functional as F
from ..mol_kernel import molecule_kernel


class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""


def compute_molecule_based_drift_loss(
    pos_gen: torch.Tensor,
    x_gen: torch.Tensor,
    pos_real: torch.Tensor,
    x_real: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Drifting field loss directly on 3D molecules.
    Assumes that the x_gen and x_real are already mapped to the spherical space.

    Returns (loss, stats) where stats is a flat dict of float diagnostics safe to
    pass directly to self.log(). Raises TrainingDivergedException on non-finite loss.
    """

    # TODO: refactor to only calling molecule kernel
    gen_distances, gen_angles = ...# prep_batch_for_kernel(pos_gen, index)
    real_distances, real_angles = ...# prep_batch_for_kernel(pos_real, index)

    kernel_pos = molecule_kernel(
        gen_distances, gen_angles, x_gen, real_distances, real_angles, x_real, index
    )  # shape: [Num_Gen_Mol, Num_Real_Mol]
    kernel_neg = molecule_kernel(
        gen_distances,
        gen_angles,
        x_gen,
        real_distances=gen_distances,
        real_angles=gen_angles,
        x_real=x_gen,
        index=index,
    )  # shape: [Num_Gen_Mol, Num_Gen_Mol]

    N_pos, N_neg = kernel_pos.shape[1], kernel_neg.shape[1]
    exp_pos = torch.clamp(kernel_pos.sum(dim=1) / N_pos, min=1e-8)
    exp_neg = torch.clamp(kernel_neg.sum(dim=1) / N_neg, min=1e-8)

    # TODO: check up tomorrow whether it is allowed to use the log 
    log_exp_pos = torch.log(exp_pos)
    log_exp_neg = torch.log(exp_neg)

    # Automatically obtain relevant gradients with regards to the inputs separately
    # Can check whether to obtain an analytical gradient instead
    grad_pos_pos = torch.autograd.grad(outputs=log_exp_pos.sum(), inputs=pos_gen)[0]
    grad_pos_neg = torch.autograd.grad(outputs=log_exp_neg.sum(), inputs=pos_gen)[0]

    grad_types_pos = torch.autograd.grad(outputs=log_exp_pos.sum(), inputs=x_gen)[0]
    grad_types_neg = torch.autograd.grad(outputs=log_exp_neg.sum(), inputs=x_gen)[0]

    # Obtain drift field by subtracting
    v_positions = grad_pos_pos - grad_pos_neg
    v_types = grad_types_pos - grad_types_neg

    # Can now obtain the targets
    target_positions = (pos_gen + v_positions).detach()

    # Target for atom types is through the exponential mapping of the spherical space:
    # TODO: Check whether tangent projection is necessary
    # v_types = (
    #     v_types - ((x_gen * v_types).sum(dim=-1, keepdim=True)) * x_gen
    # )  # Tangent projection of drifting field
    v_types_norm = torch.clamp(torch.norm(v_types, dim=-1, keepdim=True), min=1e-8)
    target_types = (
        torch.cos(v_types_norm) * x_gen
        + torch.sin(v_types_norm) * (v_types / v_types_norm)
    ).detach()

    # Distance metric for the euclidean space
    euclidean_distances = ((pos_gen - target_positions) ** 2).sum(-1)

    # Calculating distances for atom types on the sphere
    cos_sim = (x_gen * target_types).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    spherical_distances = torch.acos(cos_sim) ** 2

    # Combine the distances
    # TODO: check whether it would be better to use weights by looking at the distances from the stats
    combined_distances = euclidean_distances + spherical_distances

    # Calculate final loss as the expectation over the distances per molecule
    num_molecules = int(index.max().item()) + 1

    sum_per_molecule = torch.zeros(
        num_molecules,
        device=combined_distances.device,
        dtype=combined_distances.dtype,
    )

    sum_per_molecule.scatter_add_(0, index, combined_distances)
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

