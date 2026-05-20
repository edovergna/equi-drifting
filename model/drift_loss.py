import torch
import torch.nn.functional as F

from .spherical_utils import product_tangent_norm, sphere_exp, geodesic_distance, sphere_normalize, sphere_project_tangent
from .align import find_rotation_and_permutation, permute_generated_to_real_order, apply_pairwise_rotation, unpermute_real_order_to_gen_order

# TODO: add reshaping within loss
class TrainingDivergedException(Exception):
    """Raised when the drift loss becomes non-finite. Triggers a clean training stop."""

def compute_drift_loss(
    gen_pos: torch.Tensor,
    real_pos: torch.Tensor,
    gen_types_sphere: torch.Tensor,
    real_types: torch.Tensor,
    num_atoms: int,
    cfg
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Drifting field loss directly on 3D molecules by aligning the generated molecules with the real ones.
    Assumes that x_gen_sphere is already mapped to the spherical space

    Returns (loss, stats) where stats is a flat dict of float diagnostics safe to
    pass directly to self.log(). Raises TrainingDivergedException on non-finite loss.
    """
    real_types = real_types.float()

    eps = cfg["eps"]
    p_eta = cfg["p_eta"]
    t_eta = cfg["t_eta"]
    scale_eucl = cfg["scale_eucl"]
    scale_spher = cfg["scale_spher"]

    # Reshape for loss
    gen_pos = gen_pos.reshape(-1, num_atoms, 3)
    real_pos = real_pos.reshape(-1, num_atoms, 3)
    gen_types_sphere = gen_types_sphere.reshape(-1, num_atoms, cfg["num_atom_types"])
    real_types = real_types.reshape(-1, num_atoms, cfg["num_atom_types"])

    N_gen = gen_pos.shape[0]
    N_real = real_pos.shape[0]

    permutation_pos, R_pos, _, _ = find_rotation_and_permutation(gen_pos, real_pos, gen_types_sphere, real_types, cfg)
    permutation_neg, R_neg, _, _ = find_rotation_and_permutation(gen_pos, gen_pos, gen_types_sphere, gen_types_sphere, cfg)

    aligned_posit_pos = permute_generated_to_real_order(gen_pos, permutation_pos)
    aligned_types_pos = permute_generated_to_real_order(gen_types_sphere, permutation_pos)
    aligned_posit_neg = permute_generated_to_real_order(gen_pos, permutation_neg)
    aligned_types_neg = permute_generated_to_real_order(gen_types_sphere, permutation_neg)

    aligned_posit_pos = apply_pairwise_rotation(aligned_posit_pos, R_pos)
    aligned_posit_neg = apply_pairwise_rotation(aligned_posit_neg, R_neg)

    # Distances of shape [N_gen, N_real/N_gen] and Differences of shape [N_gen, N_real/N_gen, N_atoms, 3/5]
    posit_dist_pos, posit_diff_pos = _pairwise_geodesic_distance_and_log(aligned_posit_pos, real_pos, "euclidean", eps)
    posit_dist_neg, posit_diff_neg = _pairwise_geodesic_distance_and_log(aligned_posit_neg, gen_pos, "euclidean", eps)

    types_dist_pos, types_diff_pos = _pairwise_geodesic_distance_and_log(aligned_types_pos, real_types, "spherical", eps)
    types_dist_neg, types_diff_neg = _pairwise_geodesic_distance_and_log(aligned_types_neg, gen_types_sphere, "spherical", eps)

    # PERHAPS DIVIDE DIST BY SQRT N_ATOMS

    # Ignore self if y_neg is x
    eye = torch.eye(N_gen, device=gen_pos.device, dtype=torch.bool)
    posit_dist_neg = posit_dist_neg.masked_fill(eye, 1e6)
    types_dist_neg = types_dist_neg.masked_fill(eye, 1e6)
    
    V_posit_pos = _calc_drift_direction(posit_dist_pos, posit_diff_pos, permutation_pos, R_pos, sigma=cfg["p_sigma"], eps=eps)
    V_posit_neg = _calc_drift_direction(posit_dist_neg, posit_diff_neg, permutation_neg, R_neg, sigma=cfg["p_sigma"], eps=eps)

    V_types_pos = _calc_drift_direction(types_dist_pos, types_diff_pos, permutation_pos, R_pos, sigma=cfg["t_sigma"], eps=eps, euclidean=False)
    V_types_neg = _calc_drift_direction(types_dist_neg, types_diff_neg, permutation_neg, R_neg, sigma=cfg["t_sigma"], eps=eps, euclidean=False)

    V_posit = V_posit_pos - V_posit_neg
    V_types = V_types_pos - V_types_neg
    V_types = sphere_project_tangent(gen_types_sphere, V_types)

    V_posit = p_eta * V_posit
    V_types = t_eta * V_types

    # Calculate targets for each 
    target_posit = (gen_pos + V_posit).detach()
    target_types = sphere_exp(gen_types_sphere, V_types, eps).detach()

    # Calculate distances per molecule on each manifold
    molecule_position_dist = ((gen_pos - target_posit) ** 2).sum(dim=-1).sum(dim=-1)    # shape: [N_gen]
    molecule_types_dist = (geodesic_distance(gen_types_sphere, target_types, eps) ** 2).sum(dim=-1) # shape: [N_gen]

    loss = (scale_eucl * molecule_position_dist + scale_spher * molecule_types_dist).mean()

    stats: dict[str, float] = {}
    with torch.no_grad():
        stats["mean_euclidean_distance"] = (molecule_position_dist).mean().item()
        stats["std_euclidean_distance"] = molecule_position_dist.std().item()
        stats["mean_spherical_distance"] = molecule_types_dist.mean().item()
        stats["std_spherical_distance"] = molecule_types_dist.std().item()
        stats["norm_V_posit"] = V_posit.norm(dim=-1).mean().item()
        stats["norm_V_types"] = product_tangent_norm(V_types, eps).mean().item()
        stats["mean_V_posit_pos"] = V_posit_pos.abs().mean().item()
        stats["mean_V_posit_neg"] = V_posit_neg.abs().mean().item()
        stats["mean_V_types_pos"] = product_tangent_norm(V_types_pos, eps).mean().item()
        stats["mean_V_types_neg"] = product_tangent_norm(V_types_neg, eps).mean().item()
    
    return loss, stats


def _pairwise_geodesic_distance_and_log(
        x: torch.Tensor,
        y: torch.Tensor,
        manifold: str = "euclidean",
        eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the pairwise distance between atoms for the molecule attributes depending on 
    the specified geodesic distance. Assumed that molecules are aligned.
    Args:
        x: [N_x, N_y, N_atoms, z]
        y: [N_y, N_atoms, z]
    """

    if manifold == "euclidean":
        diff = x - y[None, :, :, :]                              # shape: (N_x, N_y, N_atoms, z)
        sq_dist_per_atom = (diff ** 2).sum(dim=-1)                              # shape: (N_x, N_y, N_atoms)

        sq_distances = sq_dist_per_atom.sum(dim=-1)   # shape: (N_x, N_y)
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


def _calc_drift_direction(dist, diff, permutation, R, sigma, eps=1e-8, euclidean=True):
    """"
    Calculates the kernel and gradient for given distances and differences. 
    Also unaligns the gradient already!
    """
    kernel = torch.exp(-dist / (2 * sigma**2))      # shape [N_gen, N_real]
    grad_kernel = (diff * kernel.unsqueeze(-1).unsqueeze(-1)) / (sigma ** 2)    # shape [N_gen, N_real, N_atoms, D]
   
    if euclidean:
        grad_kernel = - grad_kernel
        inv_R = R.transpose(-2, -1)
        grad_kernel = grad_kernel @ inv_R

    grad_kernel = unpermute_real_order_to_gen_order(grad_kernel, permutation)

    V_dir = grad_kernel.sum(dim=1) / (kernel.sum(dim=1).clamp_min(eps).unsqueeze(-1).unsqueeze(-1)) # shape [N_gen, N_atoms, D]

    return V_dir