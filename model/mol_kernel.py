import torch
import torch.nn.functional as F


# TO BE CHECKED
def prep_batch_for_kernel(positions: torch.Tensor, index: torch.Tensor):
    """
    Returns for a batch of molecules (per molecule); the pairwise distances between atoms,
    the angles between the connections of atoms.

    Args:
        positions: [Total_Atoms, 3] 3D coordinates for all atoms across the batch.
        index:     [Total_Atoms]    molecule index per atom (0-based, contiguous integers).

    Returns:
        distances: [Total_Atoms, Total_Atoms]
            distances[i, j] = ||pos_i - pos_j|| if i and j are in the same molecule, else 0.
        angles: [Total_Atoms, Total_Atoms, Total_Atoms]
            angles[i, j, k] = cosine of the angle j-i-k (atom i is the vertex) if i, j, k
            are all in the same molecule and j != i and k != i, else 0.
    """

    N = positions.shape[0]

    # --- Sparse pairwise distances ---
    # Only store intra-molecule, non-self pairs: avoids building a dense [N, N, 3] diff tensor
    same_mol = index.unsqueeze(0) == index.unsqueeze(1)  # [N, N]
    not_self = ~torch.eye(N, dtype=torch.bool, device=positions.device)  # [N, N]
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
        t_i = t_j = t_k = torch.zeros(0, dtype=torch.long, device=positions.device)
        angle_values = torch.zeros(0, device=positions.device)

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
    pos_gen: torch.Tensor,
    x_gen: torch.Tensor,
    pos_real: torch.Tensor,
    x_real: torch.Tensor,
    index: torch.Tensor,
):
    """
    Calculates the molecule kernel between each combination of real versus generated molecules
    """
    gen_distances, gen_angles = prep_batch_for_kernel(pos_gen, index)
    real_distances, real_angles = prep_batch_for_kernel(pos_real, index)

    