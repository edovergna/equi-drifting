import torch


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
    """w(s_i, t_j)"""
    d = fisher_rao_distance(x1, x2)
    return torch.exp(-(d ** 2) / (sigma_a ** 2))


def atom_similarity_cross(
    a: torch.Tensor,
    b: torch.Tensor,
    sigma_a: float,
    eps: float = 1e-8,
):
    """a: [A, D], b: [B, D] → [A, B] pairwise Fisher-Rao similarities via matmul"""
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    inner = (a @ b.T).clamp(-1.0 + eps, 1.0 - eps)
    d = torch.arccos(inner)
    return torch.exp(-(d ** 2) / (sigma_a ** 2))


def pairwise_distances(pos: torch.Tensor):
    """pos: [N, 3] → dists: [N, N]"""
    diff = pos[:, None, :] - pos[None, :, :]
    return torch.linalg.norm(diff, dim=-1)


def build_triplet_descriptors(
    pos: torch.Tensor,
    x: torch.Tensor,
    mol_index: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Enumerates all anchored triplets (anchor i, j, k) with j < k across all molecules,
    returning flat tensors of shape [T_total] / [T_total, D] ready for batched kernel ops.

    mol_idx maps each triplet to a 0-indexed molecule position in sorted(unique(mol_index)).
    Molecules with fewer than 3 atoms contribute no triplets; their K row/col stays 0.
    """
    mol_ids = torch.unique(mol_index)
    n_mols = mol_ids.shape[0]
    D = x.shape[1]

    all_d_ij, all_d_ik, all_theta = [], [], []
    all_s_j, all_s_k, all_mol_idx = [], [], []

    for local_idx, mol_id in enumerate(mol_ids):
        atom_ids = (mol_index == mol_id).nonzero(as_tuple=True)[0]
        M = atom_ids.shape[0]

        if M < 3:
            continue

        pos_m = pos[atom_ids]  # [M, 3]
        x_m = x[atom_ids]     # [M, D]

        diff = pos_m[:, None] - pos_m[None, :]
        dists_m = diff.norm(dim=-1)  # [M, M]

        local = torch.arange(M, device=pos.device)
        li, lj, lk = torch.meshgrid(local, local, local, indexing="ij")
        mask = (lj < lk) & (li != lj) & (li != lk)
        li, lj, lk = li[mask], lj[mask], lk[mask]  # each [T]

        v1 = pos_m[lj] - pos_m[li]  # [T, 3]
        v2 = pos_m[lk] - pos_m[li]  # [T, 3]
        v1 = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
        v2 = v2 / (v2.norm(dim=-1, keepdim=True) + eps)

        T = li.shape[0]
        all_d_ij.append(dists_m[li, lj])
        all_d_ik.append(dists_m[li, lk])
        all_theta.append(torch.arccos((v1 * v2).sum(dim=-1).clamp(-1 + eps, 1 - eps)))
        all_s_j.append(x_m[lj])
        all_s_k.append(x_m[lk])
        all_mol_idx.append(torch.full((T,), local_idx, device=pos.device, dtype=torch.long))

    if not all_d_ij:
        return {
            "d_ij": pos.new_zeros(0),
            "d_ik": pos.new_zeros(0),
            "theta": pos.new_zeros(0),
            "s_j": x.new_zeros(0, D),
            "s_k": x.new_zeros(0, D),
            "mol_idx": mol_index.new_zeros(0),
            "n_mols": n_mols,
        }

    return {
        "d_ij": torch.cat(all_d_ij),       # [T_total]
        "d_ik": torch.cat(all_d_ik),       # [T_total]
        "theta": torch.cat(all_theta),     # [T_total]
        "s_j": torch.cat(all_s_j),         # [T_total, D]
        "s_k": torch.cat(all_s_k),         # [T_total, D]
        "mol_idx": torch.cat(all_mol_idx), # [T_total] — 0-indexed into n_mols
        "n_mols": n_mols,
    }


def build_pair_descriptors(
    pos: torch.Tensor,
    x: torch.Tensor,
    mol_index: torch.Tensor,
):
    mol_ids = torch.unique(mol_index)

    all_d, all_s_i, all_s_j, all_mol_idx = [], [], [], []

    for local_idx, mol_id in enumerate(mol_ids):
        atom_ids = (mol_index == mol_id).nonzero(as_tuple=True)[0]
        M = atom_ids.shape[0]

        if M < 2:
            continue

        pos_m = pos[atom_ids]
        x_m = x[atom_ids]

        local = torch.arange(M, device=pos.device)
        li, lj = torch.meshgrid(local, local, indexing="ij")
        mask = li < lj
        li, lj = li[mask], lj[mask]

        diff = pos_m[li] - pos_m[lj]
        d = diff.norm(dim=-1)
        P = li.shape[0]

        all_d.append(d)
        all_s_i.append(x_m[li])
        all_s_j.append(x_m[lj])
        all_mol_idx.append(
            torch.full((P,), local_idx, device=pos.device, dtype=torch.long)
        )

    if not all_d:
        D = x.shape[1]
        return {
            "d": pos.new_zeros(0),
            "s_i": x.new_zeros(0, D),
            "s_j": x.new_zeros(0, D),
            "mol_idx": mol_index.new_zeros(0),
            "n_mols": mol_ids.shape[0],
        }

    return {
        "d": torch.cat(all_d),
        "s_i": torch.cat(all_s_i),
        "s_j": torch.cat(all_s_j),
        "mol_idx": torch.cat(all_mol_idx),
        "n_mols": mol_ids.shape[0],
    }


def molecule_kernel(
    pos_gen: torch.Tensor,
    x_gen: torch.Tensor,
    pos_real: torch.Tensor,
    x_real: torch.Tensor,
    gen_index: torch.Tensor,
    real_index: torch.Tensor,
    sigma_r: float = 0.1,
    sigma_a: float = 0.1,
    eps: float = 1e-8,
    same_samples: bool = False,
):
    """
    Pairwise molecule kernel between all generated and real molecules.

    Uses pairwise distances (geometric) and pairwise atom similarities (chemical).
    Combined kernel: geometric × chemical.
    """
    device = pos_gen.device

    g = build_pair_descriptors(pos_gen, x_gen, gen_index)
    r = build_pair_descriptors(pos_real, x_real, real_index)

    n_gen, n_real = g["n_mols"], r["n_mols"]
    Pg = g["d"].shape[0]
    Pr = r["d"].shape[0]

    if Pg == 0 or Pr == 0:
        return torch.zeros(n_gen, n_real, device=device)

    # Geometric kernel
    k_geom = torch.exp(
        -((g["d"][:, None] - r["d"][None, :]) ** 2) / (sigma_r ** 2)
    )  # [Pg, Pr]

    # Chemical kernel
    g_i = g["s_i"] / (g["s_i"].norm(dim=-1, keepdim=True) + eps)
    g_j = g["s_j"] / (g["s_j"].norm(dim=-1, keepdim=True) + eps)
    r_i = r["s_i"] / (r["s_i"].norm(dim=-1, keepdim=True) + eps)
    r_j = r["s_j"] / (r["s_j"].norm(dim=-1, keepdim=True) + eps)

    sim_ii = g_i @ r_i.T
    sim_jj = g_j @ r_j.T
    sim_ij = g_i @ r_j.T
    sim_ji = g_j @ r_i.T

    if same_samples:
        same_pair = g["mol_idx"][:, None] == r["mol_idx"][None, :]  # [Pg, Pr]
        sim_ii = sim_ii.masked_fill(same_pair, 0.0)
        sim_jj = sim_jj.masked_fill(same_pair, 0.0)
        sim_ij = sim_ij.masked_fill(same_pair, 0.0)
        sim_ji = sim_ji.masked_fill(same_pair, 0.0)

    k_chem = (
        torch.exp(-((1 - sim_ii) ** 2) / (sigma_a ** 2))
        * torch.exp(-((1 - sim_jj) ** 2) / (sigma_a ** 2))
        + torch.exp(-((1 - sim_ij) ** 2) / (sigma_a ** 2))
        * torch.exp(-((1 - sim_ji) ** 2) / (sigma_a ** 2))
    )  # [Pg, Pr]

    K_flat = k_geom * k_chem  # [Pg, Pr]

    # Aggregate pair contributions to molecule kernel
    gen_mol_mask = (
        g["mol_idx"][None, :] == torch.arange(n_gen, device=device)[:, None]
    ).to(K_flat.dtype)  # [n_gen, Pg]

    real_mol_mask = (
        r["mol_idx"][None, :] == torch.arange(n_real, device=device)[:, None]
    ).to(K_flat.dtype)  # [n_real, Pr]

    return gen_mol_mask @ K_flat @ real_mol_mask.T
