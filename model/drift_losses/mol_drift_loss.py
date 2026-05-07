import torch
import torch.nn.functional as F


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

    gen_distances, gen_angles = prep_batch_for_kernel(pos_gen, index)
    real_distances, real_angles = prep_batch_for_kernel(pos_real, index)

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
    v_types = (
        v_types - ((x_gen * v_types).sum(dim=-1, keepdim=True)) * x_gen
    )  # Tangent projection of drifting field
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


# TODO
def prep_batch_for_kernel(positions: torch.Tensor, index: torch.Tensor):
    """
    Returns for a batch of molecules (per molecule); the pairwise distances between atoms,
    the angles between the connections of atoms.
    """
    # For a batch [Total_Atoms, 3], it should return a tensor that stores the distances between atoms
    # within molecules, and a tensor that stores the angles between connections of atoms within molcules.
    # Shape for distance tensor should be ig [Total_Atoms, Total_Atoms], where we skip the calculation based
    # on the indexing for which atoms belong to which molecule.
    # Shape for angle tensor should be ig . Again skip based on molecules,
    # and the angle stored is for the first index of the tensor as "the middle atom", so that the angle between the
    # connections of this atom with the other atoms are checked. It stores cosine angles.
    #
    # Args:
    #     positions: [Total_Atoms, 3] 3D coordinates for all atoms across the batch.
    #     index:     [Total_Atoms]    molecule index per atom (0-based, contiguous integers).
    #
    # Returns:
    #     distances: [Total_Atoms, Total_Atoms]
    #         distances[i, j] = ||pos_i - pos_j|| if i and j are in the same molecule, else 0.
    #     angles: [Total_Atoms, Total_Atoms, Total_Atoms]
    #         angles[i, j, k] = cosine of the angle j-i-k (atom i is the vertex) if i, j, k
    #         are all in the same molecule and j != i and k != i, else 0.
    N = positions.shape[0]
    device = positions.device

    # --- Sparse pairwise distances ---
    # Only store intra-molecule, non-self pairs: avoids building a dense [N, N, 3] diff tensor
    same_mol = index.unsqueeze(0) == index.unsqueeze(1)  # [N, N]
    not_self = ~torch.eye(N, dtype=torch.bool, device=device)  # [N, N]
    valid_pair = same_mol & not_self  # [N, N]

    i_idx, j_idx = valid_pair.nonzero(as_tuple=True)  # [E] each
    diff = positions[i_idx] - positions[j_idx]  # [E, 3]
    dist_values = diff.norm(dim=-1)  # [E]

    distances = torch.sparse_coo_tensor(
        torch.stack([i_idx, j_idx]), dist_values, size=(N, N)
    ).coalesce()

    # --- Sparse cosine angles ---
    # Enumerate valid triples per molecule; avoids the dense [N, N, N] einsum
    t_i, t_j, t_k = [], [], []
    for m in index.unique():
        atom_ids = (index == m).nonzero(as_tuple=True)[0]  # global indices for mol m
        M = atom_ids.shape[0]
        if M < 3:
            continue
        # all ordered triples (vertex i, arm j, arm k) with i≠j, i≠k, j≠k
        gi, gj, gk = torch.meshgrid(atom_ids, atom_ids, atom_ids, indexing="ij")
        valid = (gi != gj) & (gi != gk) & (gj != gk)
        t_i.append(gi[valid])
        t_j.append(gj[valid])
        t_k.append(gk[valid])

    if t_i:
        t_i = torch.cat(t_i)  # [T]
        t_j = torch.cat(t_j)  # [T]
        t_k = torch.cat(t_k)  # [T]

        vec_ij = positions[t_j] - positions[t_i]  # [T, 3]
        vec_ik = positions[t_k] - positions[t_i]  # [T, 3]
        unit_ij = vec_ij / vec_ij.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [T, 3]
        unit_ik = vec_ik / vec_ik.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [T, 3]
        angle_values = (unit_ij * unit_ik).sum(dim=-1)  # [T] cosines
    else:
        t_i = t_j = t_k = torch.zeros(0, dtype=torch.long, device=device)
        angle_values = torch.zeros(0, device=device)

    angles = torch.sparse_coo_tensor(
        torch.stack([t_i, t_j, t_k]), angle_values, size=(N, N, N)
    ).coalesce()

    # Both tensors are returned as sparse COO tensors to save memory.
    # If downstream code needs dense tensors (e.g. for matrix ops that don't
    # support sparse), convert with:
    #     distances_dense = distances.to_dense()   # [N, N]
    #     angles_dense    = angles.to_dense()      # [N, N, N]
    # To access only the stored values and their coordinates directly:
    #     idx = distances.coalesce().indices()     # [2, nnz]  — (i, j) pairs
    #     val = distances.coalesce().values()      # [nnz]     — distances
    #     idx = angles.coalesce().indices()        # [3, nnz]  — (i, j, k) triples
    #     val = angles.coalesce().values()         # [nnz]     — cosines
    return distances, angles


# TODO
def molecule_kernel(
    gen_distances: torch.Tensor,
    gen_angles: torch.Tensor,
    x_gen: torch.Tensor,
    real_distances: torch.Tensor,
    real_angles: torch.Tensor,
    x_real: torch.Tensor,
    index: torch.Tensor,
):
    """
    Calculates the molecule kernel between each combination of real versus generated molecules
    """
