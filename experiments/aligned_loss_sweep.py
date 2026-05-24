#!/usr/bin/env python3
"""Standalone aligned-loss gamma sweep.

This is the notebook sweep from `notebooks/aligned_loss copy.ipynb` reduced to
the pieces needed for a cluster run: dataset loading, the notebook EGNN,
alignment loss, training/evaluation, and CSV checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import Center, Compose

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.mol_utils.stability import batch_to_stability, heavy_atom_counts
from model.mol_utils.validity import batch_to_validity
from model.sample_prior import get_dense_edge_index
from model.spherical_utils import (
    geodesic_distance,
    probs_to_sphere,
    product_tangent_norm,
    sphere_exp,
    sphere_normalize,
    sphere_project_tangent,
)

try:
    from torch_linear_assignment import batch_linear_assignment as _batch_assignment
except ImportError:
    _batch_assignment = None


NUM_ATOM_TYPES = 5


@dataclass(frozen=True)
class SweepConfig:
    """Frozen configuration for one sweep plan (architecture, lr, eta, and gamma grids)."""
    num_blocks: int
    hidden_nf: int
    lr: float
    p_eta: float
    t_eta: float
    train_gen: int
    pos_gammas: tuple[float, ...]
    types_gammas: tuple[float, ...]


DEFAULT_PLANS: dict[int, SweepConfig] = {
    # 4: SweepConfig(
    #     num_blocks=4,
    #     hidden_nf=192,
    #     lr=7e-5,
    #     p_eta=0.45,
    #     t_eta=0.50,
    #     train_gen=48,
    #     pos_gammas=(1.5, 2.0, 3.0, 4.0, 5.0),
    #     types_gammas=(0.5, 1.0, 1.5),
    # ),
    5: SweepConfig(
        num_blocks=5,
        hidden_nf=256,
        lr=6e-5,
        p_eta=0.45,
        t_eta=0.50,
        train_gen=48,
        pos_gammas=(2.0, 3.0, 4.0, 5.0, 6.0),
        types_gammas=(0.5, 1.0, 1.5),
    ),
    6: SweepConfig(
        num_blocks=5,
        hidden_nf=256,
        lr=5e-5,
        p_eta=0.40,
        t_eta=0.50,
        train_gen=32,
        pos_gammas=(2.0, 3.0, 4.0, 5.0, 6.0),
        types_gammas=(0.5, 1.0, 1.5),
    ),
    7: SweepConfig(
        num_blocks=6,
        hidden_nf=256,
        lr=4e-5,
        p_eta=0.40,
        t_eta=0.45,
        train_gen=32,
        pos_gammas=(3.0, 4.0, 5.0, 6.0, 7.0),
        types_gammas=(0.5, 1.0, 1.5, 2.0),
    ),
}


class EncodeAtomTypesTransform:
    """Add one-hot QM9 atom types in notebook order: H, C, N, O, F."""

    def __call__(self, data: Data) -> Data:
        """Encode QM9 atom types as one-hot vectors and attach them as real_atom_types.

        Args:
            data: PyG Data object with a z attribute containing atomic numbers.

        Returns:
            The same Data object with data.real_atom_types set to a [N, 5] float tensor.
        """
        z_to_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        real_indices = torch.tensor(
            [z_to_index[int(v.item())] for v in data.z], device=data.z.device
        )
        data.real_atom_types = F.one_hot(real_indices, num_classes=NUM_ATOM_TYPES).float()
        return data


class FullyConnectedTransform:
    """Add dense fully-connected no-self-loop edge index."""

    def __call__(self, data: Data) -> Data:
        """Add a dense fully-connected (no self-loops) edge index to the graph.

        Args:
            data: PyG Data object whose num_nodes is used to build the edge index.

        Returns:
            The same Data object with data.dense_edge_index set.
        """
        device = data.edge_index.device if data.edge_index is not None else torch.device("cpu")
        data.dense_edge_index = get_dense_edge_index(data.num_nodes, device)
        return data


class TypeGCN(MessagePassing):
    """Graph convolutional layer that updates node type features via message passing."""

    propagate_type = {"type_feat": torch.Tensor, "edge_attr": torch.Tensor}

    def __init__(self, hidden_nf: int, attention: bool = True, aggr_type: str = "sum"):
        """Initialise TypeGCN with MLPs for message computation, node update, and optional attention.

        Args:
            hidden_nf: Hidden feature dimensionality used throughout all MLPs.
            attention: If True, applies a learned sigmoid attention gate to each message.
            aggr_type: Aggregation scheme passed to MessagePassing (e.g. "sum").
        """
        super().__init__(aggr=aggr_type)
        in_message_dim = hidden_nf * 2 + 2
        self.message_mlp = nn.Sequential(
            nn.Linear(in_message_dim, in_message_dim),
            nn.SiLU(),
            nn.Linear(in_message_dim, hidden_nf),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.attention = attention
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def message(
        self,
        type_feat_i: torch.Tensor,
        type_feat_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute an attention-gated message from neighbour j to node i.

        Args:
            type_feat_i: Type features of destination nodes, shape [E, hidden_nf].
            type_feat_j: Type features of source nodes, shape [E, hidden_nf].
            edge_attr: Edge attributes (squared distances), shape [E, 2].

        Returns:
            Message tensor of shape [E, hidden_nf].
        """
        out = self.message_mlp(torch.cat([type_feat_i, type_feat_j, edge_attr], dim=-1))
        return out * self.att_mlp(out) if self.attention else out

    def update(self, aggr_out: torch.Tensor, type_feat: torch.Tensor) -> torch.Tensor:
        """Update each node's type feature with a residual MLP applied to the aggregated messages.

        Args:
            aggr_out: Aggregated messages for each node, shape [N, hidden_nf].
            type_feat: Current node type features, shape [N, hidden_nf].

        Returns:
            Updated node type features of shape [N, hidden_nf].
        """
        return type_feat + self.update_mlp(torch.cat([aggr_out, type_feat], dim=-1))

    def forward(
        self, type_feat: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Run one TypeGCN message-passing step and return updated node features.

        Args:
            type_feat: Node type features, shape [N, hidden_nf].
            edge_index: Graph connectivity, shape [2, E].
            edge_attr: Edge attributes (squared distances), shape [E, 2].

        Returns:
            Updated node type features of shape [N, hidden_nf].
        """
        return self.propagate(edge_index, type_feat=type_feat, edge_attr=edge_attr)


class PosGCN(MessagePassing):
    """Graph convolutional layer that updates atom positions via equivariant message passing."""

    propagate_type = {
        "type_feat": torch.Tensor,
        "scaled_dir_vector": torch.Tensor,
        "edge_attr": torch.Tensor,
    }

    def __init__(
        self,
        hidden_nf: int,
        tanh_coord_updates: bool = True,
        coords_range: float = 15.0,
        aggr_type: str = "sum",
    ):
        """Initialise PosGCN with a coordinate MLP and optional tanh clamping.

        Args:
            hidden_nf: Hidden feature dimensionality used in the coordinate MLP.
            tanh_coord_updates: If True, clamps raw MLP output with tanh scaled by coords_range.
            coords_range: Scale factor applied after tanh to bound positional updates.
            aggr_type: Aggregation scheme passed to MessagePassing (e.g. "sum").
        """
        super().__init__(aggr=aggr_type)
        in_message_dim = hidden_nf * 2 + 2
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(in_message_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            layer,
        )
        self.tanh_coord_updates = tanh_coord_updates
        self.coords_range = coords_range

    def message(
        self,
        type_feat_i: torch.Tensor,
        type_feat_j: torch.Tensor,
        edge_attr: torch.Tensor,
        scaled_dir_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a weighted directional message for updating atom positions.

        Args:
            type_feat_i: Type features of destination nodes, shape [E, hidden_nf].
            type_feat_j: Type features of source nodes, shape [E, hidden_nf].
            edge_attr: Edge attributes (squared distances), shape [E, 2].
            scaled_dir_vector: Unit-scaled displacement vectors from j to i, shape [E, 3].

        Returns:
            Weighted displacement message of shape [E, 3].
        """
        weight = self.coord_mlp(torch.cat([type_feat_i, type_feat_j, edge_attr], dim=-1))
        if self.tanh_coord_updates:
            weight = torch.tanh(weight) * self.coords_range
        return scaled_dir_vector * weight

    def forward(
        self,
        pos: torch.Tensor,
        type_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        scaled_dir_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Run one PosGCN step and return updated atom positions.

        Args:
            pos: Current atom positions, shape [N, 3].
            type_feat: Node type features, shape [N, hidden_nf].
            edge_index: Graph connectivity, shape [2, E].
            edge_attr: Edge attributes (squared distances), shape [E, 2].
            scaled_dir_vector: Unit-scaled displacement vectors, shape [E, 3].

        Returns:
            Updated atom positions of shape [N, 3].
        """
        delta = self.propagate(
            edge_index,
            type_feat=type_feat,
            edge_attr=edge_attr,
            scaled_dir_vector=scaled_dir_vector,
        )
        return pos + delta


class EquivariantBlock(nn.Module):
    """One EGNN block: a stack of TypeGCN layers followed by a single PosGCN update."""

    def __init__(
        self,
        hidden_nf: int,
        n_layers: int = 1,
        attention: bool = True,
        tanh_coord_updates: bool = True,
        coords_range: float = 15.0,
        aggr_type: str = "sum",
    ):
        """Initialise the equivariant block with n_layers TypeGCN modules and one PosGCN.

        Args:
            hidden_nf: Hidden feature dimensionality shared across all sub-layers.
            n_layers: Number of sequential TypeGCN layers applied before the PosGCN.
            attention: If True, enables attention gating in every TypeGCN.
            tanh_coord_updates: If True, enables tanh clamping in PosGCN.
            coords_range: Tanh scale factor for PosGCN coordinate updates.
            aggr_type: Aggregation scheme for all MessagePassing layers (e.g. "sum").
        """
        super().__init__()
        self.type_update = nn.ModuleList(
            [
                TypeGCN(hidden_nf, attention=attention, aggr_type=aggr_type)
                for _ in range(n_layers)
            ]
        )
        self.coord_update = PosGCN(
            hidden_nf,
            tanh_coord_updates=tanh_coord_updates,
            coords_range=coords_range,
            aggr_type=aggr_type,
        )

    def forward(
        self,
        type_feat: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        scaled_dir_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the equivariant block, returning updated type features and positions.

        Args:
            type_feat: Node type features, shape [N, hidden_nf].
            pos: Atom positions, shape [N, 3].
            edge_index: Graph connectivity, shape [2, E].
            edge_attr: Edge attributes (squared distances), shape [E, 2].
            scaled_dir_vector: Unit-scaled displacement vectors, shape [E, 3].

        Returns:
            Tuple of (updated type features [N, hidden_nf], updated positions [N, 3]).
        """
        for type_gcn in self.type_update:
            type_feat = type_gcn(type_feat=type_feat, edge_index=edge_index, edge_attr=edge_attr)
        pos = self.coord_update(
            pos=pos,
            type_feat=type_feat,
            edge_index=edge_index,
            scaled_dir_vector=scaled_dir_vector,
            edge_attr=edge_attr,
        )
        return type_feat, pos


def compute_edge_properties(
    pos: torch.Tensor, edge_index: torch.Tensor, norm_constant: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute squared inter-atom distances and normalised direction vectors for all edges.

    Args:
        pos: Atom positions, shape [N, 3].
        edge_index: Graph connectivity, shape [2, E].
        norm_constant: Small constant added to the norm before dividing to avoid division by zero.

    Returns:
        Tuple of (squared_norm [E], scaled_dir_vector [E, 3]).
    """
    src, dst = edge_index
    dist = pos[src] - pos[dst]
    squared_norm = dist.pow(2).sum(dim=-1)
    norm = squared_norm.sqrt()
    scaled_dir_vector = dist / (norm.unsqueeze(-1) + norm_constant)
    return squared_norm, scaled_dir_vector


class EGNN(nn.Module):
    """E(3)-equivariant graph neural network that jointly refines atom positions and type logits."""

    def __init__(
        self,
        num_atom_types: int,
        num_blocks: int = 5,
        hidden_nf: int = 256,
        num_layers_per_block: int = 1,
        attention: bool = True,
        tanh_coord_updates: bool = True,
        coords_range: float = 15.0,
        aggr_type: str = "sum",
    ):
        """Initialise EGNN with an input embedding, a stack of equivariant blocks, and an output head.

        Args:
            num_atom_types: Number of atom type classes (input/output dimensionality).
            num_blocks: Number of EquivariantBlock layers to stack.
            hidden_nf: Hidden feature dimensionality used throughout the network.
            num_layers_per_block: Number of TypeGCN layers inside each EquivariantBlock.
            attention: If True, enables attention gating in TypeGCN layers.
            tanh_coord_updates: If True, enables tanh clamping in PosGCN layers.
            coords_range: Tanh scale factor for positional updates.
            aggr_type: Message aggregation scheme (e.g. "sum").
        """
        super().__init__()
        self.type_embedding = nn.Linear(num_atom_types, hidden_nf)
        self.type_embedding_out = nn.Linear(hidden_nf, num_atom_types)
        self.blocks = nn.ModuleList(
            [
                EquivariantBlock(
                    hidden_nf,
                    n_layers=num_layers_per_block,
                    attention=attention,
                    tanh_coord_updates=tanh_coord_updates,
                    coords_range=coords_range,
                    aggr_type=aggr_type,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        pos_noise: torch.Tensor,
        type_noise: torch.Tensor,
        edge_index: torch.Tensor,
        return_change_in_pos: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Run the EGNN forward pass and return refined positions and type logits.

        Args:
            pos_noise: Noisy input positions, shape [N, 3].
            type_noise: Noisy input type features, shape [N, num_atom_types].
            edge_index: Graph connectivity, shape [2, E].
            return_change_in_pos: If True, also return the list of intermediate positions.

        Returns:
            (gen_pos, gen_types) or (gen_pos, gen_types, pos_list) when return_change_in_pos=True.
            gen_pos has shape [N, 3], gen_types has shape [N, num_atom_types], and pos_list
            contains one [N, 3] tensor per block (plus the initial positions).
        """
        gen_feats = self.type_embedding(type_noise)
        gen_pos = pos_noise
        initial_squared_norm, _ = compute_edge_properties(gen_pos, edge_index)
        pos_list: list[torch.Tensor] = []
        if return_change_in_pos:
            pos_list.append(gen_pos.clone())

        for block in self.blocks:
            squared_norm, scaled_dir_vector = compute_edge_properties(gen_pos, edge_index)
            edge_attr = torch.cat(
                [initial_squared_norm.unsqueeze(-1), squared_norm.unsqueeze(-1)], dim=-1
            )
            gen_feats, gen_pos = block(
                type_feat=gen_feats,
                pos=gen_pos,
                edge_index=edge_index,
                scaled_dir_vector=scaled_dir_vector,
                edge_attr=edge_attr,
            )
            if return_change_in_pos:
                pos_list.append(gen_pos.clone())

        gen_types = self.type_embedding_out(gen_feats)
        if return_change_in_pos:
            return gen_pos, gen_types, pos_list
        return gen_pos, gen_types


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (including all CUDA devices).

    Args:
        seed: Integer seed value to apply to all RNG sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    """Resolve a device name string to a torch.device, with auto-detection when name is "auto".

    Args:
        name: Device string such as "auto", "cpu", "cuda", "cuda:0", or "mps".

    Returns:
        The resolved torch.device.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_disable_rdkit_for_qm9() -> dict[str, Any]:
    """Temporarily shadow rdkit in sys.modules so QM9 dataset loading skips RDKit processing.

    Returns:
        A dict of previously loaded rdkit modules that must be restored afterwards.
    """
    saved = {
        k: v for k, v in sys.modules.items() if k == "rdkit" or k.startswith("rdkit.")
    }
    for key in list(saved):
        sys.modules[key] = None  # type: ignore[assignment]
    sys.modules.setdefault("rdkit", None)  # type: ignore[assignment]
    return saved


def restore_rdkit_modules(saved: dict[str, Any]) -> None:
    """Re-insert rdkit modules into sys.modules after QM9 dataset loading is complete.

    Args:
        saved: Dict returned by maybe_disable_rdkit_for_qm9 containing the original modules.
    """
    for key in list(sys.modules):
        if sys.modules[key] is None and (key == "rdkit" or key.startswith("rdkit.")):
            del sys.modules[key]
    sys.modules.update(saved)


def load_qm9(root: Path, force_reload: bool) -> QM9:
    """Load the QM9 dataset with centring, fully-connected edges, and atom-type encoding.

    RDKit is temporarily hidden from sys.modules during loading to prevent conflicts.

    Args:
        root: Path to the directory where the QM9 dataset is stored or will be downloaded.
        force_reload: If True, forces re-processing of the raw dataset files.

    Returns:
        The processed QM9 dataset object.
    """
    saved = maybe_disable_rdkit_for_qm9()
    try:
        return QM9(
            str(root),
            pre_transform=Compose([Center(), FullyConnectedTransform(), EncodeAtomTypesTransform()]),
            force_reload=force_reload,
        )
    finally:
        restore_rdkit_modules(saved)


def dense_edge_index(base_edge_index: torch.Tensor, num_gen: int, num_atoms: int) -> torch.Tensor:
    """Build a batched fully-connected edge index by tiling and offsetting a single-molecule index.

    Args:
        base_edge_index: Dense edge index for one molecule, shape [2, E_single].
        num_gen: Number of molecules in the batch.
        num_atoms: Number of atoms per molecule.

    Returns:
        Batched edge index of shape [2, E_single * num_gen].
    """
    return torch.cat([base_edge_index + offset * num_atoms for offset in range(num_gen)], dim=1)


def linear_assignment_batched(cost: torch.Tensor) -> torch.Tensor:
    """Solve a batch of linear assignment problems, using torch-linear-assignment when available.

    Falls back to SciPy's linear_sum_assignment when the fast library is not installed.

    Args:
        cost: Cost matrices, shape [B, N, N].

    Returns:
        Assignment tensors of shape [B, N] where entry [b, i] is the column assigned to row i
        in problem b.
    """
    cost_cpu = cost.detach().cpu().contiguous()
    if _batch_assignment is not None:
        return _batch_assignment(cost_cpu)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "Install torch-linear-assignment or scipy for Hungarian matching."
        ) from exc

    assignments = []
    for i in range(cost_cpu.shape[0]):
        _, tasks = linear_sum_assignment(cost_cpu[i].numpy(), maximize=False)
        assignments.append(torch.from_numpy(tasks).long())
    return torch.stack(assignments, dim=0)


@torch.no_grad()
def kabsch_rotations(gen_pos: torch.Tensor, real_pos: torch.Tensor) -> torch.Tensor:
    """Compute optimal Kabsch rotation matrices aligning generated molecules to real ones.

    Supports a 3-D generated tensor (one set of generated molecules) or a 4-D tensor
    (already expanded to pairwise form).  SVD is run on CPU when the input is on MPS.

    Args:
        gen_pos: Generated positions, shape [G, N, 3] or [G, R, N, 3].
        real_pos: Real positions, shape [R, N, 3].

    Returns:
        Rotation matrices of shape [G, R, 3, 3].
    """
    if real_pos.ndim != 3:
        raise ValueError(f"real_pos must be [N_real, N_atoms, 3], got {real_pos.shape}")

    if gen_pos.ndim == 3:
        if gen_pos.shape[1:] != real_pos.shape[1:]:
            raise ValueError(f"Shape mismatch: gen_pos {gen_pos.shape}, real_pos {real_pos.shape}")
        gen_c = gen_pos - gen_pos.mean(dim=1, keepdim=True)
        real_c = real_pos - real_pos.mean(dim=1, keepdim=True)
        h = torch.einsum("gni,rnj->grij", gen_c, real_c)
    elif gen_pos.ndim == 4:
        if gen_pos.shape[1] != real_pos.shape[0] or gen_pos.shape[2:] != real_pos.shape[1:]:
            raise ValueError(f"Shape mismatch: gen_pos {gen_pos.shape}, real_pos {real_pos.shape}")
        gen_c = gen_pos - gen_pos.mean(dim=2, keepdim=True)
        real_c = real_pos - real_pos.mean(dim=1, keepdim=True)
        h = torch.einsum("grni,rnj->grij", gen_c, real_c)
    else:
        raise ValueError(f"gen_pos must be 3D or 4D, got {gen_pos.shape}")

    h_flat = h.reshape(-1, 3, 3)
    svd_device = h_flat.device
    h_for_svd = h_flat.detach().cpu() if svd_device.type == "mps" else h_flat
    u, _, vh = torch.linalg.svd(h_for_svd)
    v = vh.transpose(-2, -1)
    ut = u.transpose(-2, -1)
    r = v @ ut
    det = torch.det(r)
    mask = det < 0
    if mask.any():
        v = v.clone()
        v[mask, :, -1] *= -1
        r = v @ ut
    return r.reshape(h.shape[0], h.shape[1], 3, 3).to(device=svd_device, dtype=h.dtype)


def build_cost_matrix(
    gen_types: torch.Tensor,
    real_types: torch.Tensor,
    gen_pos: torch.Tensor,
    real_pos: torch.Tensor,
    eps: float,
    type_weight: float = 1.0,
    pos_weight: float = 0.0,
) -> torch.Tensor:
    """Build a pairwise atom-assignment cost matrix combining type and position costs.

    Args:
        gen_types: Generated type features on the sphere, shape [G, N, D] or [G, R, N, D].
        real_types: Real type features, shape [R, N, D].
        gen_pos: Generated positions, shape [G, N, 3] or [G, R, N, 3].
        real_pos: Real positions, shape [R, N, 3].
        eps: Small epsilon for sphere normalisation.
        type_weight: Weight applied to the geodesic type cost.
        pos_weight: Weight applied to the squared positional cost.

    Returns:
        Cost matrix of shape [G, R, N, N].
    """
    gen_types = sphere_normalize(gen_types, eps)
    real_types = sphere_normalize(real_types, eps)

    if gen_types.ndim == 3:
        gen_types_exp = gen_types[:, None, None, :, :]
        gen_pos_exp = gen_pos[:, None, None, :, :]
    elif gen_types.ndim == 4:
        gen_types_exp = gen_types[:, :, None, :, :]
        gen_pos_exp = gen_pos[:, :, None, :, :]
    else:
        raise ValueError(f"Expected gen_types 3D or 4D, got {gen_types.shape}")

    real_types_exp = real_types[None, :, :, None, :]
    real_pos_exp = real_pos[None, :, :, None, :]
    dot = (gen_types_exp * real_types_exp).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    type_cost = torch.acos(dot).pow(2)
    pos_cost = (gen_pos_exp - real_pos_exp).pow(2).sum(dim=-1)
    return type_weight * type_cost + pos_weight * pos_cost


def hungarian_method_batched(
    gen_types: torch.Tensor,
    real_types: torch.Tensor,
    gen_pos: torch.Tensor,
    real_pos: torch.Tensor,
    eps: float,
    type_weight: float = 1.0,
    pos_weight: float = 0.0,
) -> torch.Tensor:
    """Solve the batched Hungarian assignment for generated-to-real atom matching.

    Args:
        gen_types: Generated type features, shape [G, N, D] or [G, R, N, D].
        real_types: Real type features, shape [R, N, D].
        gen_pos: Generated positions, shape [G, N, 3] or [G, R, N, 3].
        real_pos: Real positions, shape [R, N, 3].
        eps: Small epsilon for sphere normalisation.
        type_weight: Weight applied to the geodesic type cost.
        pos_weight: Weight applied to the squared positional cost.

    Returns:
        Assignment indices of shape [G, R, N].
    """
    cost_matrix = build_cost_matrix(
        gen_types=gen_types,
        real_types=real_types,
        gen_pos=gen_pos,
        real_pos=real_pos,
        eps=eps,
        type_weight=type_weight,
        pos_weight=pos_weight,
    )
    n_gen, n_real, n_atoms = cost_matrix.shape[:3]
    cost_flat = cost_matrix.reshape(n_gen * n_real, n_atoms, n_atoms).contiguous()
    assignment_flat = linear_assignment_batched(cost_flat).to(cost_matrix.device)
    return assignment_flat.reshape(n_gen, n_real, n_atoms)


def to_pairwise(gen: torch.Tensor, n_real: int) -> torch.Tensor:
    """Expand a generated tensor to pairwise form by repeating along the real-molecule dimension.

    Args:
        gen: Generated tensor, shape [G, N, D] or already [G, R, N, D].
        n_real: Number of real molecules R to expand to.

    Returns:
        Tensor of shape [G, R, N, D].
    """
    if gen.ndim == 3:
        return gen[:, None, :, :].expand(-1, n_real, -1, -1)
    if gen.ndim == 4:
        return gen
    raise ValueError(f"Expected 3D or 4D tensor, got {gen.shape}")


def pairwise_position_rmse(gen_pos_pairwise: torch.Tensor, real_pos: torch.Tensor) -> torch.Tensor:
    """Compute per-pair RMSE between generated and real atom positions.

    Args:
        gen_pos_pairwise: Generated positions in pairwise form, shape [G, R, N, 3].
        real_pos: Real positions, shape [R, N, 3].

    Returns:
        RMSE values of shape [G, R].
    """
    diff = gen_pos_pairwise - real_pos[None, :, :, :]
    return diff.pow(2).sum(dim=-1).mean(dim=-1).sqrt()


def permute_generated_to_real_order(gen: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """Reorder generated atoms according to an assignment that maps them to real-molecule order.

    Args:
        gen: Generated features, shape [G, N, D] or [G, R, N, D].
        assignment: Assignment indices from hungarian_method_batched, shape [G, R, N].

    Returns:
        Permuted tensor of shape [G, R, N, D].
    """
    d = gen.shape[-1]
    if gen.ndim == 3:
        gen_pairwise = gen[:, None, :, :].expand(-1, assignment.shape[1], -1, -1)
    elif gen.ndim == 4:
        gen_pairwise = gen
    else:
        raise ValueError(f"Expected gen to be 3D or 4D, got shape {gen.shape}")
    idx = assignment[..., None].expand(-1, -1, -1, d)
    return gen_pairwise.gather(dim=2, index=idx)


def apply_pairwise_rotation(gen_pos: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Apply Kabsch rotation matrices to generated positions in pairwise form.

    Args:
        gen_pos: Generated positions, shape [G, N, 3] or [G, R, N, 3].
        r: Rotation matrices, shape [G, R, 3, 3].

    Returns:
        Rotated positions of shape [G, R, N, 3].
    """
    if gen_pos.ndim == 3:
        gen_pairwise = gen_pos[:, None, :, :].expand(-1, r.shape[1], -1, -1)
    elif gen_pos.ndim == 4:
        gen_pairwise = gen_pos
    else:
        raise ValueError(f"gen_pos must be 3D or 4D, got {gen_pos.shape}")
    return gen_pairwise @ r


def unpermute_real_order_to_gen_order(
    x_perm: torch.Tensor, assignment: torch.Tensor
) -> torch.Tensor:
    """Invert a permutation applied by permute_generated_to_real_order, restoring generated order.

    Args:
        x_perm: Permuted tensor in real-molecule order, shape [G, R, N, D].
        assignment: Assignment indices used for the original permutation, shape [G, R, N].

    Returns:
        Tensor restored to generated-molecule atom order, shape [G, R, N, D].
    """
    x = torch.empty_like(x_perm)
    idx = assignment[..., None].expand_as(x_perm)
    x.scatter_(dim=2, index=idx, src=x_perm)
    return x


def find_rotation_and_permutation(
    gen_pos: torch.Tensor,
    real_pos: torch.Tensor,
    gen_types: torch.Tensor,
    real_types: torch.Tensor,
    sigma: float,
    eps: float,
    max_iter: int,
    pos_tol: float = 1e-4,
    min_iter: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iteratively find optimal atom permutations and Kabsch rotations aligning generated to real.

    Alternates between Hungarian assignment and Kabsch rotation until convergence or max_iter.
    MPS tensors are moved to CPU for SVD and returned on the original device.

    Args:
        gen_pos: Generated positions, shape [G, N, 3].
        real_pos: Real positions, shape [R, N, 3].
        gen_types: Generated type features, shape [G, N, D].
        real_types: Real type features, shape [R, N, D].
        sigma: Bandwidth parameter (unused directly here but passed for consistency).
        eps: Small epsilon for sphere normalisation.
        max_iter: Maximum number of alternating optimisation iterations.
        pos_tol: RMSE threshold below which a pair is considered converged.
        min_iter: Minimum number of iterations before early stopping is allowed.

    Returns:
        Tuple of (assignment [G, R, N], rotation [G, R, 3, 3],
        aligned_gen_pos [G, R, N, 3], aligned_gen_types [G, R, N, D]).
    """
    out_device = gen_pos.device
    out_pos_dtype = gen_pos.dtype
    out_type_dtype = gen_types.dtype

    if out_device.type == "mps":
        gen_pos = gen_pos.detach().cpu()
        real_pos = real_pos.detach().cpu()
        gen_types = gen_types.detach().cpu()
        real_types = real_types.detach().cpu()

    g_types = gen_types.clone().detach()
    g_pos = gen_pos.clone().detach()
    n_gen, n_real, n_atoms = gen_pos.shape[0], real_pos.shape[0], gen_pos.shape[1]

    active = torch.ones(n_gen, n_real, device=gen_pos.device, dtype=torch.bool)
    total_assignment = (
        torch.arange(n_atoms, device=gen_pos.device)
        .view(1, 1, n_atoms)
        .expand(n_gen, n_real, n_atoms)
        .clone()
    )
    total_r = (
        torch.eye(3, device=gen_pos.device, dtype=gen_pos.dtype)
        .view(1, 1, 3, 3)
        .expand(n_gen, n_real, 3, 3)
        .clone()
    )

    for step in range(max_iter):
        pos_weight = 0.0 if step == 0 else 0.1
        old_g_pos = to_pairwise(g_pos, n_real)
        old_g_types = to_pairwise(g_types, n_real)
        step_assignment = hungarian_method_batched(
            g_types,
            real_types,
            g_pos,
            real_pos,
            eps=eps,
            type_weight=1.0,
            pos_weight=pos_weight,
        )
        cand_g_pos = permute_generated_to_real_order(g_pos, step_assignment)
        cand_g_types = permute_generated_to_real_order(g_types, step_assignment)
        step_r = kabsch_rotations(cand_g_pos, real_pos)
        cand_g_pos = apply_pairwise_rotation(cand_g_pos, step_r)
        cand_total_assignment = total_assignment.gather(dim=2, index=step_assignment)
        cand_total_r = total_r @ step_r
        pair_mask = active[..., None, None]
        assign_mask = active[..., None]
        g_pos = torch.where(pair_mask, cand_g_pos, old_g_pos)
        g_types = torch.where(pair_mask, cand_g_types, old_g_types)
        total_assignment = torch.where(assign_mask, cand_total_assignment, total_assignment)
        total_r = torch.where(pair_mask, cand_total_r, total_r)
        done = pairwise_position_rmse(g_pos, real_pos) <= pos_tol
        active = torch.ones_like(done) if step + 1 < min_iter else ~done
        if not active.any():
            break

    if out_device.type == "mps":
        total_assignment = total_assignment.to(out_device)
        total_r = total_r.to(device=out_device, dtype=out_pos_dtype)
        g_pos = g_pos.to(device=out_device, dtype=out_pos_dtype)
        g_types = g_types.to(device=out_device, dtype=out_type_dtype)
    return total_assignment, total_r, g_pos, g_types


def pairwise_geodesic_distance_and_log(
    x: torch.Tensor,
    y: torch.Tensor,
    manifold: str = "euclidean",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise geodesic squared distances and logarithmic map vectors from x towards y.

    Supports Euclidean space (plain squared distances) and the product sphere (geodesic arcs).

    Args:
        x: Source tensor in pairwise form, shape [G, R, N, D].
        y: Target tensor, shape [R, N, D].
        manifold: "euclidean" or "spherical".
        eps: Small epsilon for numerical stability on the sphere.

    Returns:
        Tuple of (sq_distances [G, R], diff [G, R, N, D]) where diff is the log-map direction.
    """
    if manifold == "euclidean":
        diff = x - y[None, :, :, :]
        sq_distances = diff.pow(2).sum(dim=-1).sum(dim=-1)
    elif manifold == "spherical":
        x = sphere_normalize(x, eps)
        y = sphere_normalize(y.unsqueeze(0), eps)
        dot = (x * y).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(dot)
        sq_distances = theta.pow(2).sum(dim=-1).clamp_min(eps)
        u = y - dot.unsqueeze(-1) * x
        u_norm = u.norm(dim=-1, keepdim=True)
        out = (theta.unsqueeze(-1) / u_norm.clamp_min(eps)) * u
        small = theta.unsqueeze(-1) < 1e-5
        first_order = sphere_project_tangent(x, y - x)
        out = torch.where(small, first_order, out)
        diff = sphere_project_tangent(x, out)
    else:
        raise ValueError(f"Undefined manifold: {manifold}")
    return sq_distances, diff


def calc_drift_direction(
    dist: torch.Tensor,
    diff: torch.Tensor,
    permutation: torch.Tensor,
    r: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
    euclidean: bool = True,
) -> torch.Tensor:
    """Compute a kernel-weighted drift direction from aligned pairwise distances and log-maps.

    Weights log-map directions by a Gaussian kernel over pairwise distances, un-rotates and
    un-permutes back to the original generated-molecule atom ordering, then averages over real
    molecules.

    Args:
        dist: Pairwise squared distances, shape [G, R].
        diff: Log-map vectors from generated to target, shape [G, R, N, D].
        permutation: Assignment indices mapping generated to real order, shape [G, R, N].
        r: Kabsch rotation matrices, shape [G, R, 3, 3].
        sigma: Gaussian kernel bandwidth.
        eps: Small epsilon to avoid division by zero when normalising.
        euclidean: If True, negates the gradient and un-rotates with the transpose rotation.

    Returns:
        Drift direction tensor of shape [G, N, D].
    """
    kernel = torch.exp(-dist / (2 * sigma**2))
    grad_kernel = (diff * kernel.unsqueeze(-1).unsqueeze(-1)) / (sigma**2)
    if euclidean:
        grad_kernel = -grad_kernel
        grad_kernel = grad_kernel @ r.transpose(-2, -1)
    grad_kernel = unpermute_real_order_to_gen_order(grad_kernel, permutation)
    return grad_kernel.sum(dim=1) / kernel.sum(dim=1).clamp_min(eps).unsqueeze(-1).unsqueeze(-1)


def compute_aligning_drift_loss(
    gen_pos: torch.Tensor,
    real_pos: torch.Tensor,
    gen_types_sphere: torch.Tensor,
    real_types: torch.Tensor,
    posit_sigma: float = 2.0,
    types_sigma: float = 1.0,
    eps: float = 1e-8,
    scale_loss: float = 1.0,
    t_eta: float = 1.0,
    p_eta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the contrastive aligning drift loss for positions and atom types.

    Aligns generated molecules to real molecules (positive pairs) and to other generated
    molecules (negative pairs), then computes a loss that pushes generated samples towards
    real molecules on both the Euclidean position manifold and the product sphere for types.

    Args:
        gen_pos: Generated positions, shape [G, N, 3].
        real_pos: Real positions, shape [R, N, 3].
        gen_types_sphere: Generated type features on the sphere, shape [G, N, D].
        real_types: Real type features, shape [R, N, D].
        posit_sigma: Gaussian kernel bandwidth for position alignment.
        types_sigma: Gaussian kernel bandwidth for type alignment.
        eps: Small epsilon for numerical stability.
        scale_loss: Scalar multiplier applied to the type distance term in the loss.
        t_eta: Step-size multiplier for the type drift direction.
        p_eta: Step-size multiplier for the position drift direction.

    Returns:
        Tuple of (scalar loss tensor, stats dict with distance and drift-norm diagnostics).
    """
    real_types = real_types.float()
    n_gen = gen_pos.shape[0]

    permutation_pos, r_pos, _, _ = find_rotation_and_permutation(
        gen_pos, real_pos, gen_types_sphere, real_types, posit_sigma, eps, max_iter=10
    )
    permutation_neg, r_neg, _, _ = find_rotation_and_permutation(
        gen_pos, gen_pos, gen_types_sphere, gen_types_sphere, posit_sigma, eps, max_iter=10
    )

    aligned_posit_pos = apply_pairwise_rotation(
        permute_generated_to_real_order(gen_pos, permutation_pos), r_pos
    )
    aligned_types_pos = permute_generated_to_real_order(gen_types_sphere, permutation_pos)
    aligned_posit_neg = apply_pairwise_rotation(
        permute_generated_to_real_order(gen_pos, permutation_neg), r_neg
    )
    aligned_types_neg = permute_generated_to_real_order(gen_types_sphere, permutation_neg)

    posit_dist_pos, posit_diff_pos = pairwise_geodesic_distance_and_log(
        aligned_posit_pos, real_pos, "euclidean", eps
    )
    posit_dist_neg, posit_diff_neg = pairwise_geodesic_distance_and_log(
        aligned_posit_neg, gen_pos, "euclidean", eps
    )
    types_dist_pos, types_diff_pos = pairwise_geodesic_distance_and_log(
        aligned_types_pos, real_types, "spherical", eps
    )
    types_dist_neg, types_diff_neg = pairwise_geodesic_distance_and_log(
        aligned_types_neg, gen_types_sphere, "spherical", eps
    )

    eye = torch.eye(n_gen, device=gen_pos.device, dtype=torch.bool)
    posit_dist_neg = posit_dist_neg.masked_fill(eye, 1e6)
    types_dist_neg = types_dist_neg.masked_fill(eye, 1e6)

    v_posit_pos = calc_drift_direction(
        posit_dist_pos, posit_diff_pos, permutation_pos, r_pos, sigma=posit_sigma, eps=eps
    )
    v_posit_neg = calc_drift_direction(
        posit_dist_neg, posit_diff_neg, permutation_neg, r_neg, sigma=posit_sigma, eps=eps
    )
    v_types_pos = calc_drift_direction(
        types_dist_pos,
        types_diff_pos,
        permutation_pos,
        r_pos,
        sigma=types_sigma,
        eps=eps,
        euclidean=False,
    )
    v_types_neg = calc_drift_direction(
        types_dist_neg,
        types_diff_neg,
        permutation_neg,
        r_neg,
        sigma=types_sigma,
        eps=eps,
        euclidean=False,
    )

    v_posit = p_eta * (v_posit_pos - v_posit_neg)
    v_types = sphere_project_tangent(gen_types_sphere, t_eta * (v_types_pos - v_types_neg))
    target_posit = (gen_pos + v_posit).detach()
    target_types = sphere_exp(gen_types_sphere, v_types, eps).detach()
    molecule_position_dist = (gen_pos - target_posit).pow(2).sum(dim=-1).sum(dim=-1)
    molecule_types_dist = geodesic_distance(gen_types_sphere, target_types, eps).pow(2).sum(dim=-1)
    loss = (molecule_position_dist + scale_loss * molecule_types_dist).mean()

    with torch.no_grad():
        stats = {
            "mean_euclidean_distance": float(molecule_position_dist.mean().item()),
            "std_euclidean_distance": float(molecule_position_dist.std().item()),
            "mean_spherical_distance": float(molecule_types_dist.mean().item()),
            "std_spherical_distance": float(molecule_types_dist.std().item()),
            "norm_V_posit": float(v_posit.norm(dim=-1).mean().item()),
            "norm_V_types": float(product_tangent_norm(v_types, eps).mean().item()),
            "mean_V_posit_pos": float(v_posit_pos.abs().mean().item()),
            "mean_V_posit_neg": float(v_posit_neg.abs().mean().item()),
            "mean_V_types_pos": float(product_tangent_norm(v_types_pos, eps).mean().item()),
            "mean_V_types_neg": float(product_tangent_norm(v_types_neg, eps).mean().item()),
        }
    return loss, stats


def select_real_molecules(
    dataset: QM9, atom_count: int, real_limit: int | None
) -> tuple[list[Any], torch.Tensor, torch.Tensor]:
    """Filter QM9 for molecules with a specific atom count and return their positions and types.

    Args:
        dataset: The loaded QM9 dataset.
        atom_count: Number of atoms each selected molecule must have.
        real_limit: Maximum number of molecules to return; None means no limit.

    Returns:
        Tuple of (list of Data objects, real_positions [R, N, 3], real_types [R, N, D]).
    """
    indices = [i for i, mol in enumerate(dataset) if int(mol.num_nodes) == atom_count]
    if real_limit is not None:
        indices = indices[:real_limit]
    if not indices:
        raise ValueError(f"No QM9 molecules found with atom_count={atom_count}")
    real_mols = [dataset[i] for i in indices]
    real_positions = torch.stack([mol.pos.float() for mol in real_mols], dim=0)
    real_positions = real_positions - real_positions.mean(dim=1, keepdim=True)
    real_types = torch.stack([mol.real_atom_types.float() for mol in real_mols], dim=0)
    return real_mols, real_positions, real_types


def noise_banks(
    bank_size: int,
    num_atoms: int,
    num_atom_types: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-generate fixed Gaussian noise banks for positions and atom types.

    Args:
        bank_size: Number of noise samples to pre-generate.
        num_atoms: Number of atoms per sample.
        num_atom_types: Number of atom type dimensions.
        seed: Random seed for reproducible noise generation.
        device: Device on which the noise tensors are placed.

    Returns:
        Tuple of (pos_bank [bank_size, num_atoms, 3], type_bank [bank_size, num_atoms, num_atom_types]).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pos_bank = torch.randn((bank_size, num_atoms, 3), generator=generator)
    pos_bank = pos_bank - pos_bank.mean(dim=1, keepdim=True)
    type_bank = torch.randn((bank_size, num_atoms, num_atom_types), generator=generator)
    return pos_bank.to(device), type_bank.to(device)


def build_model(
    cfg: SweepConfig,
    num_atom_types: int,
    seed: int,
    device: torch.device,
    attention: bool,
    tanh_coord_updates: bool,
    aggr_type: str,
) -> nn.Module:
    """Construct and return an EGNN model initialised with a fixed seed on the given device.

    Args:
        cfg: Sweep configuration specifying num_blocks and hidden_nf.
        num_atom_types: Number of atom type classes.
        seed: Random seed for weight initialisation reproducibility.
        device: Device on which the model is placed.
        attention: If True, enables attention gating in TypeGCN layers.
        tanh_coord_updates: If True, enables tanh clamping in PosGCN layers.
        aggr_type: Message aggregation scheme (e.g. "sum").

    Returns:
        Initialised EGNN model on the specified device.
    """
    set_seed(seed)
    return EGNN(
        num_atom_types=num_atom_types,
        num_blocks=cfg.num_blocks,
        hidden_nf=cfg.hidden_nf,
        attention=attention,
        tanh_coord_updates=tanh_coord_updates,
        aggr_type=aggr_type,
    ).to(device)


def summarize_generated_molecules(
    gen_pos: torch.Tensor, gen_type_probs: torch.Tensor
) -> dict[str, float]:
    """Compute validity, uniqueness, heavy-atom count, and stability metrics for a batch of molecules.

    Args:
        gen_pos: Generated atom positions, shape [M, N, 3].
        gen_type_probs: Softmax type probabilities, shape [M, N, num_atom_types].

    Returns:
        Dict with keys "validity", "uniqueness", "heavy_atom_mean", "atom_stability",
        and "mol_stability", each holding a float in [0, 1] or a mean count.
    """
    num_mol, num_atoms, _ = gen_pos.shape
    batch_vec = torch.arange(num_mol, device=gen_pos.device).repeat_interleave(num_atoms)
    hard_types = gen_type_probs.argmax(dim=-1).float()
    pos_flat = gen_pos.reshape(num_mol * num_atoms, 3).detach().cpu()
    types_flat = hard_types.reshape(num_mol * num_atoms).detach().cpu()
    batch_cpu = batch_vec.detach().cpu()

    validity_results = batch_to_validity(pos_flat, types_flat, batch_cpu)
    heavy = heavy_atom_counts(types_flat, batch_cpu)
    atom_stability, mol_stability = batch_to_stability(pos_flat, types_flat, batch_cpu)
    n_total = len(validity_results)
    n_valid = sum(1 for ok, _ in validity_results if ok)
    valid_ids = [ident for ok, ident in validity_results if ok and ident is not None]
    return {
        "validity": n_valid / n_total if n_total else 0.0,
        "uniqueness": len(set(valid_ids)) / len(valid_ids) if valid_ids else 0.0,
        "heavy_atom_mean": float(np.mean(heavy)) if heavy else 0.0,
        "atom_stability": float(atom_stability),
        "mol_stability": float(mol_stability),
    }


@torch.no_grad()
def generate_eval_samples(
    model: nn.Module,
    num_gen: int,
    num_atoms: int,
    num_atom_types: int,
    seed: int,
    base_edge_index: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a fixed set of molecules from Gaussian noise for evaluation.

    Args:
        model: Trained EGNN model in eval mode.
        num_gen: Number of molecules to generate.
        num_atoms: Number of atoms per molecule.
        num_atom_types: Number of atom type classes.
        seed: Random seed for reproducible noise sampling.
        base_edge_index: Dense edge index for one molecule, shape [2, E_single].
        device: Device on which inference is run.

    Returns:
        Tuple of (gen_pos [num_gen, num_atoms, 3], gen_type_probs [num_gen, num_atoms, num_atom_types]).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pos_noise = torch.randn((num_gen, num_atoms, 3), generator=generator)
    pos_noise = pos_noise - pos_noise.mean(dim=1, keepdim=True)
    type_noise = torch.randn((num_gen, num_atoms, num_atom_types), generator=generator)
    eval_edge_index = dense_edge_index(base_edge_index, num_gen, num_atoms)
    gen_pos, gen_types = model(
        pos_noise.reshape(num_gen * num_atoms, 3).to(device),
        type_noise.reshape(num_gen * num_atoms, num_atom_types).to(device),
        eval_edge_index,
    )
    gen_types = gen_types.reshape(num_gen, num_atoms, num_atom_types)
    gen_pos = gen_pos.reshape(num_gen, num_atoms, 3)
    gen_pos = gen_pos - gen_pos.mean(dim=1, keepdim=True)
    return gen_pos, F.softmax(gen_types, dim=-1)


def history_columns() -> list[str]:
    """Return the ordered list of column names for the per-iteration history CSV.

    Returns:
        List of column name strings.
    """
    return [
        "atom_count",
        "pos_gamma",
        "types_gamma",
        "seed",
        "iter",
        "loss",
        "position_dist",
        "type_dist",
        "grad_norm",
        "monitor_n",
        "validity",
        "valid_mols",
        "uniqueness",
        "heavy_atom_mean",
        "atom_stability",
        "stable_atoms",
        "mol_stability",
        "stable_mols",
        "monitor_error",
    ]


def result_columns() -> list[str]:
    """Return the ordered list of column names for the per-gamma-pair results CSV.

    Returns:
        List of column name strings.
    """
    return [
        "atom_count",
        "n_real_molecules",
        "pos_gamma",
        "types_gamma",
        "seed",
        "iters",
        "final_loss",
        "final_position_dist",
        "final_type_dist",
        "validity",
        "uniqueness",
        "heavy_atom_mean",
        "atom_stability",
        "mol_stability",
        "runtime_sec",
        "diverged",
        "diverged_iter",
        "diverged_reason",
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a list of row dicts to a CSV file, creating parent directories as needed.

    Args:
        path: Destination file path.
        rows: List of dicts mapping column names to values.
        fieldnames: Ordered list of column names that determines column order in the output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort result rows by descending validity, mol_stability, uniqueness, atom_stability, and loss.

    Args:
        rows: List of result dicts as produced by run_single_gamma_pair.

    Returns:
        New sorted list with the best result first.
    """
    return sorted(
        rows,
        key=lambda r: (
            -float(r["validity"]),
            -float(r["mol_stability"]),
            -float(r["uniqueness"]),
            -float(r["atom_stability"]),
            float(r["final_loss"]) if np.isfinite(float(r["final_loss"])) else float("inf"),
        ),
    )


def run_single_gamma_pair(
    *,
    atom_count: int,
    n_real_molecules: int,
    pos_gamma: float,
    types_gamma: float,
    cfg: SweepConfig,
    seed: int,
    num_iters: int,
    num_eval_gen: int,
    monitor_gen: int,
    log_every: int,
    pos_bank: torch.Tensor,
    type_bank: torch.Tensor,
    base_edge_index: torch.Tensor,
    real_positions: torch.Tensor,
    real_types: torch.Tensor,
    device: torch.device,
    attention: bool,
    tanh_coord_updates: bool,
    aggr_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train one EGNN model for a given (pos_gamma, types_gamma) pair and return results.

    Builds a fresh model, runs the training loop with periodic monitoring, detects divergence,
    and performs a final evaluation pass when training completes normally.

    Args:
        atom_count: Number of atoms in the target molecules.
        n_real_molecules: Number of real molecules used as the target distribution.
        pos_gamma: Gaussian bandwidth for the position alignment kernel.
        types_gamma: Gaussian bandwidth for the type alignment kernel.
        cfg: Sweep configuration (architecture, learning rate, etc.).
        seed: Random seed for model initialisation and noise sampling.
        num_iters: Total number of training iterations.
        num_eval_gen: Number of molecules generated for the final evaluation.
        monitor_gen: Number of molecules generated for periodic monitoring.
        log_every: Log and monitor every this many iterations.
        pos_bank: Pre-generated position noise bank, shape [bank_size, num_atoms, 3].
        type_bank: Pre-generated type noise bank, shape [bank_size, num_atoms, num_atom_types].
        base_edge_index: Dense edge index for one molecule, shape [2, E_single].
        real_positions: Real molecule positions, shape [R, num_atoms, 3].
        real_types: Real molecule type features, shape [R, num_atoms, D].
        device: Device on which training is run.
        attention: If True, enables attention gating in TypeGCN layers.
        tanh_coord_updates: If True, enables tanh clamping in PosGCN layers.
        aggr_type: Message aggregation scheme (e.g. "sum").

    Returns:
        Tuple of (result_row dict, history list of per-log-step dicts).
    """
    num_atoms = real_positions.shape[1]
    num_atom_types = real_types.shape[-1]
    model = build_model(
        cfg,
        num_atom_types,
        seed,
        device,
        attention=attention,
        tanh_coord_updates=tanh_coord_updates,
        aggr_type=aggr_type,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_edge_index = dense_edge_index(base_edge_index, cfg.train_gen, num_atoms)
    real_pos_run = real_positions.to(device)
    real_types_run = real_types.to(device)
    history: list[dict[str, Any]] = []
    final_stats: dict[str, float] = {}
    start = time.time()
    diverged = False
    diverged_reason = ""
    diverged_iter = None

    set_seed(seed)
    for step in range(num_iters):
        model.train()
        optimizer.zero_grad()
        idx = torch.randint(0, pos_bank.shape[0], (cfg.train_gen,), device=device)
        pos_noise = pos_bank[idx].reshape(cfg.train_gen * num_atoms, 3)
        type_noise = type_bank[idx].reshape(cfg.train_gen * num_atoms, num_atom_types)
        gen_pos, gen_types = model(pos_noise, type_noise, train_edge_index)
        gen_types = gen_types.reshape(cfg.train_gen, num_atoms, num_atom_types)
        gen_pos = gen_pos.reshape(cfg.train_gen, num_atoms, 3)
        gen_pos = gen_pos - gen_pos.mean(dim=1, keepdim=True)

        if not torch.isfinite(gen_pos).all() or not torch.isfinite(gen_types).all():
            diverged = True
            diverged_iter = step + 1
            diverged_reason = "non-finite model output"
            break

        gen_type_probs = F.softmax(gen_types, dim=-1)
        gen_types_sphere = probs_to_sphere(gen_type_probs, eps=1e-8)
        try:
            loss, stats = compute_aligning_drift_loss(
                gen_pos,
                real_pos_run,
                gen_types_sphere,
                real_types_run,
                p_eta=cfg.p_eta,
                t_eta=cfg.t_eta,
                scale_loss=1.0,
                posit_sigma=pos_gamma,
                types_sigma=types_gamma,
            )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            diverged = True
            diverged_iter = step + 1
            diverged_reason = f"{type(exc).__name__}: {exc}"
            break

        if not torch.isfinite(loss):
            diverged = True
            diverged_iter = step + 1
            diverged_reason = "non-finite loss"
            break

        loss.backward()
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(total_grad_norm):
            diverged = True
            diverged_iter = step + 1
            diverged_reason = "non-finite gradient norm"
            break
        optimizer.step()
        final_stats = stats

        should_log = step == 0 or (step + 1) % log_every == 0 or step + 1 == num_iters
        if should_log:
            monitor_metrics = {
                "validity": float("nan"),
                "uniqueness": float("nan"),
                "heavy_atom_mean": float("nan"),
                "atom_stability": float("nan"),
                "mol_stability": float("nan"),
            }
            monitor_error = ""
            model.eval()
            try:
                monitor_pos, monitor_type_probs = generate_eval_samples(
                    model,
                    num_gen=monitor_gen,
                    num_atoms=num_atoms,
                    num_atom_types=num_atom_types,
                    seed=seed + 10_000,
                    base_edge_index=base_edge_index,
                    device=device,
                )
                monitor_metrics = summarize_generated_molecules(monitor_pos, monitor_type_probs)
            except Exception as exc:
                monitor_error = f"{type(exc).__name__}: {exc}"

            valid_count = (
                int(round(monitor_metrics["validity"] * monitor_gen))
                if np.isfinite(monitor_metrics["validity"])
                else None
            )
            stable_mol_count = (
                int(round(monitor_metrics["mol_stability"] * monitor_gen))
                if np.isfinite(monitor_metrics["mol_stability"])
                else None
            )
            stable_atom_count = (
                int(round(monitor_metrics["atom_stability"] * monitor_gen * num_atoms))
                if np.isfinite(monitor_metrics["atom_stability"])
                else None
            )
            history.append(
                {
                    "atom_count": atom_count,
                    "pos_gamma": float(pos_gamma),
                    "types_gamma": float(types_gamma),
                    "seed": int(seed),
                    "iter": step + 1,
                    "loss": float(loss.item()),
                    "position_dist": float(stats["mean_euclidean_distance"]),
                    "type_dist": float(stats["mean_spherical_distance"]),
                    "grad_norm": float(total_grad_norm.item()),
                    "monitor_n": int(monitor_gen),
                    "validity": monitor_metrics["validity"],
                    "valid_mols": valid_count,
                    "uniqueness": monitor_metrics["uniqueness"],
                    "heavy_atom_mean": monitor_metrics["heavy_atom_mean"],
                    "atom_stability": monitor_metrics["atom_stability"],
                    "stable_atoms": stable_atom_count,
                    "mol_stability": monitor_metrics["mol_stability"],
                    "stable_mols": stable_mol_count,
                    "monitor_error": monitor_error,
                }
            )
            print(
                f"atom={atom_count} pos={pos_gamma:g} type={types_gamma:g} "
                f"iter={step + 1}/{num_iters} loss={loss.item():.5g} "
                f"pos_dist={stats['mean_euclidean_distance']:.5g} "
                f"type_dist={stats['mean_spherical_distance']:.5g} "
                f"valid={monitor_metrics['validity']:.3f} ({valid_count}/{monitor_gen}) "
                f"mol_stab={monitor_metrics['mol_stability']:.3f} ({stable_mol_count}/{monitor_gen}) "
                f"atom_stab={monitor_metrics['atom_stability']:.3f}",
                flush=True,
            )

    if diverged:
        print(
            f"atom={atom_count} pos={pos_gamma:g} type={types_gamma:g} "
            f"diverged at iter={diverged_iter}: {diverged_reason}",
            flush=True,
        )
        metrics = {
            "validity": 0.0,
            "uniqueness": 0.0,
            "heavy_atom_mean": 0.0,
            "atom_stability": 0.0,
            "mol_stability": 0.0,
        }
        final_loss = history[-1]["loss"] if history else float("nan")
    else:
        model.eval()
        eval_pos, eval_type_probs = generate_eval_samples(
            model,
            num_gen=num_eval_gen,
            num_atoms=num_atoms,
            num_atom_types=num_atom_types,
            seed=seed + 10_000,
            base_edge_index=base_edge_index,
            device=device,
        )
        metrics = summarize_generated_molecules(eval_pos, eval_type_probs)
        final_loss = history[-1]["loss"] if history else float("nan")

    return (
        {
            "atom_count": atom_count,
            "n_real_molecules": n_real_molecules,
            "pos_gamma": float(pos_gamma),
            "types_gamma": float(types_gamma),
            "seed": int(seed),
            "iters": int(num_iters),
            "final_loss": final_loss,
            "final_position_dist": float(final_stats.get("mean_euclidean_distance", float("nan"))),
            "final_type_dist": float(final_stats.get("mean_spherical_distance", float("nan"))),
            "validity": metrics["validity"],
            "uniqueness": metrics["uniqueness"],
            "heavy_atom_mean": metrics["heavy_atom_mean"],
            "atom_stability": metrics["atom_stability"],
            "mol_stability": metrics["mol_stability"],
            "runtime_sec": time.time() - start,
            "diverged": bool(diverged),
            "diverged_iter": diverged_iter,
            "diverged_reason": diverged_reason,
        },
        history,
    )


def run_atom_count_sweep(
    *,
    atom_count: int,
    cfg: SweepConfig,
    dataset: QM9,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Run the full gamma grid sweep for one atom count and save incremental CSV results.

    Iterates over all (pos_gamma, types_gamma) combinations in cfg, calls
    run_single_gamma_pair for each, checkpoints results after every pair, and returns
    the best result row.

    Args:
        atom_count: Number of atoms in the target molecules.
        cfg: Sweep configuration including the gamma grids.
        dataset: The loaded QM9 dataset.
        args: Parsed command-line arguments.
        device: Device on which training is run.

    Returns:
        The result dict corresponding to the best-performing gamma pair.
    """
    real_mols, real_positions, real_types = select_real_molecules(
        dataset, atom_count, args.real_limit
    )
    print(f"\nStarting atom_count={atom_count}", flush=True)
    print(f"Found {len(real_mols)} real QM9 molecules with {atom_count} atoms.", flush=True)
    print(f"Config: {cfg}", flush=True)

    base_edge_index = real_mols[0].dense_edge_index.to(device)
    pos_bank, type_bank = noise_banks(
        bank_size=args.noise_bank_size,
        num_atoms=atom_count,
        num_atom_types=NUM_ATOM_TYPES,
        seed=args.seed,
        device=device,
    )
    results_csv = args.out_dir / f"aligned_loss_atoms{atom_count}_gamma_grid_results.csv"
    history_csv = args.out_dir / f"aligned_loss_atoms{atom_count}_gamma_grid_history.csv"

    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for pos_gamma, types_gamma in product(cfg.pos_gammas, cfg.types_gammas):
        print(f"\n=== atom_count={atom_count} pos_gamma={pos_gamma:g}, types_gamma={types_gamma:g} ===")
        row, history = run_single_gamma_pair(
            atom_count=atom_count,
            n_real_molecules=len(real_mols),
            pos_gamma=pos_gamma,
            types_gamma=types_gamma,
            cfg=cfg,
            seed=args.seed,
            num_iters=args.num_iters,
            num_eval_gen=args.eval_gen,
            monitor_gen=args.monitor_gen,
            log_every=args.log_every,
            pos_bank=pos_bank,
            type_bank=type_bank,
            base_edge_index=base_edge_index,
            real_positions=real_positions,
            real_types=real_types,
            device=device,
            attention=not args.no_attention,
            tanh_coord_updates=not args.no_tanh_coord_updates,
            aggr_type=args.aggr_type,
        )
        rows.append(row)
        history_rows.extend(history)
        sorted_rows = sort_results(rows)
        write_csv(results_csv, sorted_rows, result_columns())
        write_csv(history_csv, history_rows, history_columns())
        best = sorted_rows[0]
        print(
            "best so far: "
            f"pos={best['pos_gamma']:g} type={best['types_gamma']:g} "
            f"valid={best['validity']:.3f} mol_stab={best['mol_stability']:.3f} "
            f"atom_stab={best['atom_stability']:.3f} loss={best['final_loss']}",
            flush=True,
        )

    sorted_rows = sort_results(rows)
    write_csv(results_csv, sorted_rows, result_columns())
    write_csv(history_csv, history_rows, history_columns())
    print(f"Saved results to {results_csv}", flush=True)
    print(f"Saved history to {history_csv}", flush=True)
    return sorted_rows[0]


def parse_float_list(value: str | None) -> tuple[float, ...] | None:
    """Parse a comma-separated string of floats into a tuple, or return None if input is None.

    Args:
        value: Comma-separated float string (e.g. "1.0,2.0,3.0") or None.

    Returns:
        Tuple of parsed floats, or None when value is None.
    """
    if value is None:
        return None
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def apply_overrides(cfg: SweepConfig, args: argparse.Namespace) -> SweepConfig:
    """Apply command-line overrides to a SweepConfig, returning a new frozen config.

    Args:
        cfg: Base SweepConfig taken from DEFAULT_PLANS.
        args: Parsed command-line arguments; only non-None override fields are applied.

    Returns:
        A new SweepConfig with the specified fields replaced.
    """
    updates: dict[str, Any] = {}
    for arg_name, field_name in [
        ("num_blocks", "num_blocks"),
        ("hidden_nf", "hidden_nf"),
        ("lr", "lr"),
        ("p_eta", "p_eta"),
        ("t_eta", "t_eta"),
        ("train_gen", "train_gen"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.pos_gammas is not None:
        updates["pos_gammas"] = parse_float_list(args.pos_gammas)
    if args.types_gammas is not None:
        updates["types_gammas"] = parse_float_list(args.types_gammas)
    return replace(cfg, **updates)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the aligned-loss gamma sweep.

    Returns:
        Namespace object containing all parsed argument values.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-counts", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "QM9")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "notebooks" / "aligned_temperature_results"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=5000)
    parser.add_argument("--eval-gen", type=int, default=1000)
    parser.add_argument("--monitor-gen", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--noise-bank-size", type=int, default=10_000)
    parser.add_argument("--real-limit", type=int, default=None)
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--aggr-type", default="sum")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-tanh-coord-updates", action="store_true")
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--hidden-nf", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--p-eta", type=float, default=None)
    parser.add_argument("--t-eta", type=float, default=None)
    parser.add_argument("--train-gen", type=int, default=None)
    parser.add_argument("--pos-gammas", default=None, help="Comma-separated override, e.g. 2,3,4")
    parser.add_argument("--types-gammas", default=None, help="Comma-separated override, e.g. 0.5,1,1.5")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny 2-iteration sweep on atom_count=4 to verify the script.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse arguments, load QM9, run the gamma sweep, and save summary results."""
    args = parse_args()
    if args.smoke:
        args.atom_counts = [4]
        args.num_iters = 2
        args.eval_gen = 8
        args.monitor_gen = 4
        args.log_every = 1
        args.noise_bank_size = min(args.noise_bank_size, 128)
        args.real_limit = 4 if args.real_limit is None else args.real_limit
        args.pos_gammas = "2"
        args.types_gammas = "1"
        args.num_blocks = 2 if args.num_blocks is None else args.num_blocks
        args.hidden_nf = 64 if args.hidden_nf is None else args.hidden_nf
        args.train_gen = 2 if args.train_gen is None else args.train_gen

    device = choose_device(args.device)
    print(f"repo_root={REPO_ROOT}", flush=True)
    print(f"data_root={args.data_root}", flush=True)
    print(f"out_dir={args.out_dir}", flush=True)
    print(f"device={device}", flush=True)
    dataset = load_qm9(args.data_root, args.force_reload)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_rows = []
    for atom_count in args.atom_counts:
        if atom_count not in DEFAULT_PLANS:
            raise ValueError(f"No default plan for atom_count={atom_count}; use 4, 5, 6, or 7.")
        cfg = apply_overrides(DEFAULT_PLANS[atom_count], args)
        best_rows.append(
            run_atom_count_sweep(
                atom_count=atom_count,
                cfg=cfg,
                dataset=dataset,
                args=args,
                device=device,
            )
        )

    summary_csv = args.out_dir / "aligned_loss_sweep_best_by_atom_count.csv"
    write_csv(summary_csv, best_rows, result_columns())
    print(f"\nSaved best-by-atom-count summary to {summary_csv}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
