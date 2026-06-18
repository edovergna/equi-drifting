"""Molecular alignment of positions using Kabsch algorithm.

Aligns generated molecules to real molecules by finding optimal rotations 
through the Kabsch algorithm.
"""

# Adding Kabsch algorithm
import torch
from contextlib import nullcontext

@torch.no_grad()
def kabsch_rotations(gen_pos, real_pos):
    """Compute optimal rotation matrices aligning generated to real molecules.

    Uses Singular Value Decomposition (SVD) to find rotation matrices minimizing
    squared distance between centered point clouds.

    Args:
        gen_pos: Generated positions, either [N_gen, N_atoms, 3] or [N_gen, N_real, N_atoms, 3].
        real_pos: Real positions [N_real, N_atoms, 3].

    Returns:
        Rotation matrices [N_gen, N_real, 3, 3] (or [N_gen, N_gen, 3, 3] if pairwise).
    """
    out_dtype = gen_pos.dtype
    device_type = gen_pos.device.type
    autocast_ctx = (
        torch.autocast(device_type=device_type, enabled=False)
        if device_type in {"cuda", "cpu"}
        else nullcontext()
    )

    with autocast_ctx:
        gen_pos = gen_pos.float()
        real_pos = real_pos.float()

        if real_pos.ndim != 3:
            raise ValueError(
                f"real_pos must be [N_real, N_atoms, 3], got {real_pos.shape}"
            )

        if gen_pos.ndim == 3:
            if gen_pos.shape[1:] != real_pos.shape[1:]:
                raise ValueError(
                    f"Shape mismatch: gen_pos {gen_pos.shape}, real_pos {real_pos.shape}"
                )

            gen_c = gen_pos - gen_pos.mean(dim=1, keepdim=True)
            real_c = real_pos - real_pos.mean(dim=1, keepdim=True)
            H = torch.einsum("gni,rnj->grij", gen_c, real_c)

        elif gen_pos.ndim == 4:
            if gen_pos.shape[1] != real_pos.shape[0]:
                raise ValueError(
                    f"Pairwise gen_pos has N_real={gen_pos.shape[1]}, "
                    f"but real_pos has N_real={real_pos.shape[0]}"
                )
            if gen_pos.shape[2:] != real_pos.shape[1:]:
                raise ValueError(
                    f"Shape mismatch: gen_pos {gen_pos.shape}, real_pos {real_pos.shape}"
                )

            gen_c = gen_pos - gen_pos.mean(dim=2, keepdim=True)
            real_c = real_pos - real_pos.mean(dim=1, keepdim=True)
            H = torch.einsum("grni,rnj->grij", gen_c, real_c)

        else:
            raise ValueError(f"gen_pos must be 3D or 4D, got {gen_pos.shape}")

        H_flat = H.reshape(-1, 3, 3)

        U, S, Vh = torch.linalg.svd(H_flat)

        V = Vh.transpose(-2, -1)
        Ut = U.transpose(-2, -1)

        R = V @ Ut

        return R.reshape(H.shape[0], H.shape[1], 3, 3).to(out_dtype)


def _to_pairwise(gen, n_real):
    """Expand tensor to pairwise form for all real molecules.

    Args:
        gen: Tensor [N_gen, N_atoms, D] to expand.
        n_real: Number of real molecules.

    Returns:
        Pairwise tensor [N_gen, N_real, N_atoms, D].
    """
    if gen.ndim == 3:
        return gen[:, None, :, :].expand(-1, n_real, -1, -1)
    if gen.ndim == 4:
        return gen
    raise ValueError(f"Expected 3D or 4D tensor, got {gen.shape}")


def apply_pairwise_rotation(gen_pos, R):
    """Apply rotation matrices to generated positions.

    Args:
        gen_pos: Positions [N_gen, N_atoms, 3] or [N_gen, N_real, N_atoms, 3].
        R: Rotation matrices [N_gen, N_real, 3, 3].

    Returns:
        Rotated positions [N_gen, N_real, N_atoms, 3].
    """
    if gen_pos.ndim == 3:
        gen_pairwise = gen_pos[:, None, :, :].expand(-1, R.shape[1], -1, -1)
    elif gen_pos.ndim == 4:
        gen_pairwise = gen_pos
    else:
        raise ValueError(f"gen_pos must be 3D or 4D, got {gen_pos.shape}")

    return gen_pairwise @ R
