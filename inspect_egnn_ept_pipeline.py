"""
Test the SE(3)-invariance of the EPT encoder as implemented in ept/ept.py.

WHY INVARIANCE, NOT EQUIVARIANCE
─────────────────────────────────
Equivariance means the output *transforms with* the input: f(R·x) = R·f(x).
That applies to node-level geometric quantities (e.g. force vectors).
Our EPT encoder produces a single graph-level embedding vector phi ∈ R^D per
molecule.  For phi to be a useful, rotation-independent fingerprint it must
be *invariant*: phi(R·pos) = phi(pos).  That is what we test here.

WHY NO EGNN / GUMBEL SOFTMAX
──────────────────────────────
Gumbel softmax is only used to discretise the EGNN's atom-logit output during
generation.  Here we feed the EPT encoder real QM9 atom types (hard one-hot
vectors), so the test is deterministic and does not depend on a trained EGNN.

Tests (per molecule, on real QM9 positions and atom types)
───────────────────────────────────────────────────────────
  1. Translation      – shift all atoms by a random vector
  2. Rotation         – apply a random SO(3) matrix
  3. Roto-translation – both at once
  4. Reflection       – flip the x-axis (O(3) but not SO(3))

For each, we check:
  |phi(T·pos) - phi(pos)| / |phi(pos)|  ≈  0  (< ATOL)

Run with:
    python inspect_egnn_ept_pipeline.py
"""

import sys
import torch

# ── Molecule indices to inspect (change these to any valid QM9 dataset indices) ──
MOLECULE_INDICES = [0, 1, 5]

# ── QM9 data root ──
QM9_ROOT = "data/QM9"

# ── Tolerance for the invariance check ──
ATOL = 1e-4  # absolute tolerance on the relative error

# ─────────────────────────────────────────────────────────────────────────────


def load_qm9_molecules(indices, root):
    """Return a list of processed PyG Data objects for the requested indices."""
    from torch_geometric.transforms import Center, Compose
    from model.datamodule import EncodeAtomTypesTransform, FullyConnectedTransform

    # Mirror the rdkit workaround used in QM9DataModule.setup()
    _rdkit_saved = {
        k: v for k, v in sys.modules.items() if k == "rdkit" or k.startswith("rdkit.")
    }
    for k in list(_rdkit_saved):
        sys.modules[k] = None  # type: ignore[assignment]
    sys.modules.setdefault("rdkit", None)  # type: ignore[assignment]

    try:
        from torch_geometric.datasets import QM9

        dataset = QM9(
            root,
            pre_transform=Compose(
                [Center(), FullyConnectedTransform(), EncodeAtomTypesTransform()]
            ),
        )
    finally:
        for k in list(sys.modules):
            if sys.modules[k] is None and (k == "rdkit" or k.startswith("rdkit.")):
                del sys.modules[k]
        sys.modules.update(_rdkit_saved)

    return [dataset[i] for i in indices]


def build_batch(molecules, device):
    """
    Batch a list of PyG Data objects into the flat tensors our EPT wrapper expects.
    Returns x, pos, batch_vec, dense_edge_index.
    """
    from model.sample_prior import get_dense_edge_index

    xs, poss, batches, edges = [], [], [], []
    offset = 0
    for graph_idx, mol in enumerate(molecules):
        n = mol.num_nodes
        xs.append(mol.real_atom_types)  # [n, 5]
        poss.append(mol.pos)  # [n, 3]
        batches.append(torch.full((n,), graph_idx, dtype=torch.long))
        local_edge = get_dense_edge_index(n, device)  # [2, n*(n-1)]
        edges.append(local_edge + offset)
        offset += n

    x = torch.cat(xs, dim=0).to(device)
    pos = torch.cat(poss, dim=0).to(device)
    batch_vec = torch.cat(batches, dim=0).to(device)
    dense_edge_index = torch.cat(edges, dim=1).to(device)
    return x, pos, batch_vec, dense_edge_index


def random_rotation(device, dtype=torch.float32):
    """Return a uniformly random SO(3) rotation matrix [3, 3]."""
    # QR decomposition of a random Gaussian matrix gives a Haar-distributed
    # orthogonal matrix; flip sign if det = -1 to stay in SO(3).
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # Make det(Q) = +1
    Q = Q * torch.sign(torch.det(Q))
    return Q


def run_ept(feature_extractor, pos, x, batch_vec, dense_edge_index, device):
    """Single EPT forward pass; returns graph embeddings [G, D]."""
    with torch.no_grad():
        phi = feature_extractor(
            pos=pos,
            atom_types=x,
            block_id=torch.arange(pos.shape[0], device=device),
            batch_id=batch_vec,
            dense_edge_index=dense_edge_index,
        )
    return phi


def check_invariance(label, phi_orig, phi_transformed, atol):
    """
    Print per-molecule relative error and whether it is below `atol`.
    Returns True if all molecules pass.
    """
    passed = True
    for g_idx in range(phi_orig.shape[0]):
        diff = (phi_orig[g_idx] - phi_transformed[g_idx]).norm().item()
        denom = phi_orig[g_idx].norm().item()
        rel = diff / (denom + 1e-12)
        ok = rel < atol
        status = "PASS" if ok else "FAIL"
        print(
            f"    [{status}] mol[{MOLECULE_INDICES[g_idx]}]  "
            f"|Δphi|={diff:.2e}  |phi|={denom:.4f}  rel={rel:.2e}"
        )
        if not ok:
            passed = False
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── 1. Load QM9 molecules ────────────────────────────────────────────────
    print(f"Loading QM9 molecules at indices {MOLECULE_INDICES} …")
    molecules = load_qm9_molecules(MOLECULE_INDICES, QM9_ROOT)
    atom_symbols = {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"}
    for idx, mol in zip(MOLECULE_INDICES, molecules):
        names = [atom_symbols[int(r.argmax())] for r in mol.real_atom_types]
        print(f"  mol[{idx}]: {mol.num_nodes} atoms  → {names}")

    # ── 2. Build batch and load EPT ──────────────────────────────────────────
    x, pos, batch_vec, dense_edge_index = build_batch(molecules, device)
    print(
        f"\nBatch shapes: x={tuple(x.shape)}  pos={tuple(pos.shape)}  "
        f"batch_vec={tuple(batch_vec.shape)}  edges={tuple(dense_edge_index.shape)}"
    )

    print("\nLoading EPT encoder …")
    from ept.ept_loader import load_ept_feature_extractor

    ept = load_ept_feature_extractor().to(device)
    ept.eval()

    # ── 3. Baseline embedding ────────────────────────────────────────────────
    print("\n── Baseline EPT embeddings ──")
    phi_orig = run_ept(ept, pos, x, batch_vec, dense_edge_index, device)
    print(f"  phi shape: {tuple(phi_orig.shape)}  (one row per molecule)")
    for g_idx, mol_idx in enumerate(MOLECULE_INDICES):
        e = phi_orig[g_idx]
        print(
            f"  mol[{mol_idx}]  mean={e.mean():.4f}  std={e.std():.4f}  norm={e.norm():.4f}"
        )

    all_passed = True

    # ── 4. Translation invariance ────────────────────────────────────────────
    print("\n── Test 1: Translation invariance ──")
    t = torch.randn(3, device=device, dtype=pos.dtype) * 10.0  # large shift
    print(f"  shift vector: {t.tolist()}")
    pos_trans = pos + t.unsqueeze(0)  # broadcast over atoms
    phi_trans = run_ept(ept, pos_trans, x, batch_vec, dense_edge_index, device)
    all_passed &= check_invariance("translation", phi_orig, phi_trans, ATOL)

    # ── 5. Rotation invariance ───────────────────────────────────────────────
    print("\n── Test 2: Rotation invariance ──")
    R = random_rotation(device, pos.dtype)
    print(f"  rotation matrix:\n{R.cpu().numpy()}")
    # Apply the *same* rotation matrix to every atom (pos is already center-zeroed
    # by the Center() transform, but it doesn't matter for invariance).
    pos_rot = pos @ R.T  # [N, 3] @ [3, 3]
    phi_rot = run_ept(ept, pos_rot, x, batch_vec, dense_edge_index, device)
    all_passed &= check_invariance("rotation", phi_orig, phi_rot, ATOL)

    # ── 6. Roto-translation invariance ──────────────────────────────────────
    print("\n── Test 3: Roto-translation invariance ──")
    pos_rt = pos_rot + t.unsqueeze(0)
    phi_rt = run_ept(ept, pos_rt, x, batch_vec, dense_edge_index, device)
    all_passed &= check_invariance("roto-translation", phi_orig, phi_rt, ATOL)

    # ── 7. Reflection (improper rotation, det=-1) ────────────────────────────
    print("\n── Test 4: Reflection invariance (x → -x flip) ──")
    pos_refl = pos.clone()
    pos_refl[:, 0] = -pos_refl[:, 0]  # flip x-axis
    phi_refl = run_ept(ept, pos_refl, x, batch_vec, dense_edge_index, device)
    all_passed &= check_invariance("reflection", phi_orig, phi_refl, ATOL)

    # ── 8. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL INVARIANCE TESTS PASSED")
    else:
        print("ONE OR MORE INVARIANCE TESTS FAILED  ← encoder is NOT invariant")
    print("=" * 60)


if __name__ == "__main__":
    main()
