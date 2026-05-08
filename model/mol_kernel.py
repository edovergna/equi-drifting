# import torch
# import torch.nn.functional as F


# # TO BE CHECKED
# def prep_batch_for_kernel(positions: torch.Tensor, index: torch.Tensor):
#     """
#     Returns for a batch of molecules (per molecule); the pairwise distances between atoms,
#     the angles between the connections of atoms.

#     Args:
#         positions: [Total_Atoms, 3] 3D coordinates for all atoms across the batch.
#         index:     [Total_Atoms]    molecule index per atom (0-based, contiguous integers).

#     Returns:
#         distances: [Total_Atoms, Total_Atoms]
#             distances[i, j] = ||pos_i - pos_j|| if i and j are in the same molecule, else 0.
#         angles: [Total_Atoms, Total_Atoms, Total_Atoms]
#             angles[i, j, k] = cosine of the angle j-i-k (atom i is the vertex) if i, j, k
#             are all in the same molecule and j != i and k != i, else 0.
#     """

#     N = positions.shape[0]

#     # --- Sparse pairwise distances ---
#     # Only store intra-molecule, non-self pairs: avoids building a dense [N, N, 3] diff tensor
#     same_mol = index.unsqueeze(0) == index.unsqueeze(1)  # [N, N]
#     not_self = ~torch.eye(N, dtype=torch.bool, device=positions.device)  # [N, N]
#     valid_pair = same_mol & not_self  # [N, N]

#     i_idx, j_idx = valid_pair.nonzero(as_tuple=True)  # [E] each
#     diff = positions[i_idx] - positions[j_idx]  # [E, 3]
#     dist_values = diff.norm(dim=-1)  # [E]

#     distances = torch.sparse_coo_tensor(
#         torch.stack([i_idx, j_idx]), dist_values, size=(N, N)
#     ).coalesce()

#     # --- Sparse cosine angles ---
#     # Enumerate valid triples per molecule; avoids the dense [N, N, N] einsum
#     t_i, t_j, t_k = [], [], []
#     for m in index.unique():
#         atom_ids = (index == m).nonzero(as_tuple=True)[0]  # global indices for mol m
#         M = atom_ids.shape[0]
#         if M < 3:
#             continue
#         # all ordered triples (vertex i, arm j, arm k) with i≠j, i≠k, j≠k
#         gi, gj, gk = torch.meshgrid(atom_ids, atom_ids, atom_ids, indexing="ij")
#         valid = (gi != gj) & (gi != gk) & (gj != gk)
#         t_i.append(gi[valid])
#         t_j.append(gj[valid])
#         t_k.append(gk[valid])

#     if t_i:
#         t_i = torch.cat(t_i)  # [T]
#         t_j = torch.cat(t_j)  # [T]
#         t_k = torch.cat(t_k)  # [T]

#         vec_ij = positions[t_j] - positions[t_i]  # [T, 3]
#         vec_ik = positions[t_k] - positions[t_i]  # [T, 3]
#         unit_ij = vec_ij / vec_ij.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [T, 3]
#         unit_ik = vec_ik / vec_ik.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [T, 3]
#         angle_values = (unit_ij * unit_ik).sum(dim=-1)  # [T] cosines
#     else:
#         t_i = t_j = t_k = torch.zeros(0, dtype=torch.long, device=positions.device)
#         angle_values = torch.zeros(0, device=positions.device)

#     angles = torch.sparse_coo_tensor(
#         torch.stack([t_i, t_j, t_k]), angle_values, size=(N, N, N)
#     ).coalesce()

#     # Both tensors are returned as sparse COO tensors to save memory.
#     # If downstream code needs dense tensors (e.g. for matrix ops that don't
#     # support sparse), convert with:
#     #     distances_dense = distances.to_dense()   # [N, N]
#     #     angles_dense    = angles.to_dense()      # [N, N, N]
#     # To access only the stored values and their coordinates directly:
#     #     idx = distances.coalesce().indices()     # [2, nnz]  — (i, j) pairs
#     #     val = distances.coalesce().values()      # [nnz]     — distances
#     #     idx = angles.coalesce().indices()        # [3, nnz]  — (i, j, k) triples
#     #     val = angles.coalesce().values()         # [nnz]     — cosines
#     return distances, angles


# # TODO
# def molecule_kernel(
#     pos_gen: torch.Tensor,
#     x_gen: torch.Tensor,
#     pos_real: torch.Tensor,
#     x_real: torch.Tensor,
#     gen_index: torch.Tensor,
#     real_index: torch.Tensor,
# ):
#     """
#     Calculates the molecule kernel between each combination of real versus generated molecules
#     """
#     gen_distances, gen_angles = prep_batch_for_kernel(pos_gen, gen_index)
#     real_distances, real_angles = prep_batch_for_kernel(pos_real, real_index)

import torch

# ============================================================
# Fisher–Rao atom similarity
# ============================================================

def fisher_rao_distance(
    x1: torch.Tensor,
    x2: torch.Tensor,
    eps: float = 1e-8,
):
    """
    x1, x2: [..., D]
    assumes embeddings lie on positive orthant of unit sphere
    """

    x1 = x1 / (x1.norm(dim=-1, keepdim=True) + eps)
    x2 = x2 / (x2.norm(dim=-1, keepdim=True) + eps)

    inner = (x1 * x2).sum(dim=-1)
    inner = inner.clamp(-1.0 + eps, 1.0 - eps)

    return torch.arccos(inner)


def atom_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    sigma_a: float,
):
    """
    w(s_i, t_j)
    """

    d = fisher_rao_distance(x1, x2)

    return torch.exp(-(d ** 2) / (sigma_a ** 2))


# ============================================================
# Geometry helpers
# ============================================================

def pairwise_distances(pos: torch.Tensor):
    """
    pos: [N, 3]

    returns:
        dists: [N, N]
    """

    diff = pos[:, None, :] - pos[None, :, :]
    return torch.linalg.norm(diff, dim=-1)


def triplet_angle(
    pos: torch.Tensor,
    i: int,
    j: int,
    k: int,
    eps: float = 1e-8,
):
    """
    angle at anchor i between j and k
    """

    v1 = pos[j] - pos[i]
    v2 = pos[k] - pos[i]

    v1 = v1 / (torch.linalg.norm(v1) + eps)
    v2 = v2 / (torch.linalg.norm(v2) + eps)

    cos_theta = torch.dot(v1, v2)
    cos_theta = cos_theta.clamp(-1.0 + eps, 1.0 - eps)

    return torch.arccos(cos_theta)


# ============================================================
# Enumerate anchored triplets
# ============================================================

def enumerate_triplets(atom_indices):
    """
    Returns all anchored triplets (i,j,k)
    with j < k and j,k != i
    """

    triplets = []

    atom_indices = atom_indices.tolist()

    for i in atom_indices:

        neigh = [a for a in atom_indices if a != i]

        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):

                j = neigh[a]
                k = neigh[b]

                triplets.append((i, j, k))

    return triplets


# ============================================================
# Precompute triplet descriptors
# ============================================================

def build_triplet_descriptors(
    pos: torch.Tensor,
    x: torch.Tensor,
    mol_index: torch.Tensor,
):
    """
    Precompute descriptors for every molecule.

    returns:
        dict[mol_id] -> list of triplet dicts
    """

    dists = pairwise_distances(pos)

    mol_ids = torch.unique(mol_index)

    out = {}

    for mol_id in mol_ids:

        atom_ids = torch.where(mol_index == mol_id)[0]

        triplets = enumerate_triplets(atom_ids)

        mol_triplets = []

        for (i, j, k) in triplets:

            descriptor = {
                "anchor": i,
                "j": j,
                "k": k,

                # geometry
                "d_ij": dists[i, j],
                "d_ik": dists[i, k],
                "theta": triplet_angle(pos, i, j, k),

                # atom embeddings
                "s_j": x[j],
                "s_k": x[k],
            }

            mol_triplets.append(descriptor)

        out[int(mol_id)] = mol_triplets

    return out


# ============================================================
# Geometric kernel
# ============================================================

def geometric_kernel(
    trip_a,
    trip_b,
    sigma_r: float,
    sigma_theta: float,
):
    """
    K_geom
    """

    k1 = torch.exp(
        -((trip_a["d_ij"] - trip_b["d_ij"]) ** 2)
        / (sigma_r ** 2)
    )

    k2 = torch.exp(
        -((trip_a["d_ik"] - trip_b["d_ik"]) ** 2)
        / (sigma_r ** 2)
    )

    k_angle = torch.exp(
        -((trip_a["theta"] - trip_b["theta"]) ** 2)
        / (sigma_theta ** 2)
    )

    return k1 * k2 * k_angle


# ============================================================
# Symmetrized chemical kernel
# ============================================================

def chemical_kernel(
    trip_a,
    trip_b,
    sigma_a: float,
):
    """
    Symmetrized over neighbor order.
    """

    s_j = trip_a["s_j"]
    s_k = trip_a["s_k"]

    t_r = trip_b["s_j"]
    t_s = trip_b["s_k"]

    term1 = (
        atom_similarity(s_j, t_r, sigma_a)
        * atom_similarity(s_k, t_s, sigma_a)
    )

    term2 = (
        atom_similarity(s_j, t_s, sigma_a)
        * atom_similarity(s_k, t_r, sigma_a)
    )

    return term1 + term2


# ============================================================
# Full molecule kernel
# ============================================================

def molecule_kernel(
    pos_gen: torch.Tensor,
    x_gen: torch.Tensor,
    pos_real: torch.Tensor,
    x_real: torch.Tensor,
    gen_index: torch.Tensor,
    real_index: torch.Tensor,
    sigma_r: float = 1.0,
    sigma_theta: float = 1.0,
    sigma_a: float = 1.0,
):
    """
    Computes anchored triplet kernel between all
    generated and real molecules.

    Inputs
    ------
    pos_gen:  [Ng, 3]
    x_gen:    [Ng, 5]

    pos_real: [Nr, 3]
    x_real:   [Nr, 5]

    gen_index:  [Ng]
    real_index: [Nr]

    Returns
    -------
    K: [n_gen_mols, n_real_mols]
    """

    device = pos_gen.device

    # --------------------------------------------
    # precompute triplets
    # --------------------------------------------

    gen_triplets = build_triplet_descriptors(
        pos_gen,
        x_gen,
        gen_index,
    )

    real_triplets = build_triplet_descriptors(
        pos_real,
        x_real,
        real_index,
    )

    gen_mols = sorted(gen_triplets.keys())
    real_mols = sorted(real_triplets.keys())

    K = torch.zeros(
        len(gen_mols),
        len(real_mols),
        device=device,
    )

    # --------------------------------------------
    # molecule pair kernel
    # --------------------------------------------

    for gi, gmol in enumerate(gen_mols):

        g_triplets = gen_triplets[gmol]

        for ri, rmol in enumerate(real_mols):

            r_triplets = real_triplets[rmol]

            total = 0.0

            for tg in g_triplets:

                for tr in r_triplets:

                    k_geom = geometric_kernel(
                        tg,
                        tr,
                        sigma_r=sigma_r,
                        sigma_theta=sigma_theta,
                    )

                    k_chem = chemical_kernel(
                        tg,
                        tr,
                        sigma_a=sigma_a,
                    )

                    total = total + k_geom * k_chem

            K[gi, ri] = total

    return K