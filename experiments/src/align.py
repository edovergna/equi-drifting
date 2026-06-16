import torch
from torch_scatter import scatter_mean


def kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Find rotation R and translation t that minimize ||R @ P.T + t - Q.T||^2,
    then return P aligned to Q.

    Args:
        P: [n_nodes, 3] source points (prediction)
        Q: [n_nodes, 3] target points

    Returns:
        P_aligned: [n_nodes, 3] - P after optimal rigid transform to match Q
    """
    # 1. Center both point clouds
    P_centroid = P.mean(dim=0, keepdim=True)  # [1, 3]
    Q_centroid = Q.mean(dim=0, keepdim=True)  # [1, 3]
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # 2. Compute covariance matrix H = P_centered^T @ Q_centered
    H = P_centered.T @ Q_centered  # [3, 3]

    # 3. SVD: H = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(H)

    # 4. Correct for reflection (ensure proper rotation, det = +1)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))

    # 5. Optimal rotation
    R = Vt.T @ D @ U.T  # [3, 3]

    # 6. Apply rotation and translate to Q's centroid
    P_aligned = (R @ P_centered.T).T + Q_centroid

    return P_aligned

def kabsch_align_pyg(P: torch.Tensor, Q: torch.Tensor,
                    batch: torch.Tensor, n_mols: int) -> torch.Tensor:
    """
    Vectorized Kabsch alignment for PyG-style flat layout with uniform mol size.

    Args:
        P: [N_total, 3] predictions
        Q: [N_total, 3] targets
        batch: [N_total] long tensor, molecule index per node
        n_mols: number of molecules in the batch

    Returns:
        P_aligned: [N_total, 3], each molecule rigidly aligned to its target
    """
    # 1. Per-molecule centroids and centering
    P_centroid = scatter_mean(P, batch, dim=0)  # [n_mols, 3]
    Q_centroid = scatter_mean(Q, batch, dim=0)  # [n_mols, 3]
    P_c = P - P_centroid[batch]                 # [N_total, 3]
    Q_c = Q - Q_centroid[batch]                 # [N_total, 3]

    # 2. Per-molecule covariance H_m = P_c_m^T @ Q_c_m, computed via scatter.
    # Outer product per node: [N_total, 3, 3], then sum within each molecule.
    outer = P_c.unsqueeze(2) * Q_c.unsqueeze(1)  # [N_total, 3, 3]
    H = scatter_mean(outer, batch, dim=0) * (outer.shape[0] / n_mols)
    H = H + 1e-4 * torch.eye(3, device=H.device).expand_as(H)

    # Note: scatter_mean gives the mean; multiplying by nodes-per-mol recovers
    # the sum. Since SVD's rotation is invariant to positive scaling of H,
    # you can actually skip this rescale — keeping it for clarity.

    # 3. Batched SVD
    U, S, Vt = torch.linalg.svd(H)  # all [n_mols, 3, 3]

    # 4. Reflection correction per molecule
    det = torch.linalg.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2))  # [n_mols]
    D = torch.eye(3, device=P.device, dtype=P.dtype).expand(n_mols, -1, -1).clone()
    D[:, 2, 2] = torch.sign(det)

    R = Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)  # [n_mols, 3, 3]

    # 5. Apply rotation per node, then translate to target centroid
    R_per_node = R[batch]                                # [N_total, 3, 3]
    P_rot = torch.einsum('nij,nj->ni', R_per_node, P_c)  # [N_total, 3]
    P_aligned = P_rot + Q_centroid[batch]

    return P_aligned, R_per_node


def kabsch_mse_pyg(P, Q, batch, n_mols):
    P_aligned = kabsch_align_pyg(P, Q, batch, n_mols)
    return ((P_aligned - Q) ** 2).mean()


def kabsch_rotations_pairwise(
    gen_pos: torch.Tensor,
    target_pos: torch.Tensor,
) -> torch.Tensor:
    """Find the optimal Kabsch rotation for every (gen, target) pair.

    Both inputs must be zero-centered (CoM = 0); no translation is applied.

    Args:
        gen_pos: [N_gen, N_atoms, 3]
        target_pos: [N_target, N_atoms, 3]

    Returns:
        R: [N_gen, N_target, 3, 3] — rotation matrices such that gen @ R ≈ target
    """
    N_gen = gen_pos.shape[0]
    N_target = target_pos.shape[0]

    # Cross-covariance: H[g, t] = gen[g]^T @ target[t]
    # [N_gen, N_target, 3, 3]
    H = torch.einsum("gni,tnj->gtij", gen_pos, target_pos)
    # [N_gen * N_target, 3, 3]
    H_flat = H.reshape(-1, 3, 3)

    U, _, Vh = torch.linalg.svd(H_flat)

    # Reflection correction: ensure det(R) = +1
    # [N_gen * N_target]
    det = torch.linalg.det(Vh.transpose(-2, -1) @ U.transpose(-2, -1))
    # [N_gen * N_target, 3, 3]
    D = torch.eye(3, device=gen_pos.device, dtype=gen_pos.dtype).unsqueeze(0).expand(N_gen * N_target, -1, -1).clone()
    D[:, 2, 2] = torch.sign(det)

    # [N_gen * N_target, 3, 3]
    R_flat = Vh.transpose(-2, -1) @ D @ U.transpose(-2, -1)
    # [N_gen, N_target, 3, 3]
    return R_flat.reshape(N_gen, N_target, 3, 3)
