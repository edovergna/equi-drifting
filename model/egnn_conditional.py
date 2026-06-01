"""Conditional EGNN for position generation given fixed atom types.

Atom types are embedded once into invariant node features and remain
fixed throughout all equivariant position update layers. Unlike the
joint EGNN, no TypeGCN layers are used: types are treated as static
conditioning, not as generated outputs.
"""

import torch
from torch import nn

from .egnn import PosGCN, compute_edge_properties


class ConditionalEGNN(nn.Module):
    """Position generator conditioned on static atom types.

    Given fixed one-hot atom type vectors, embeds them into invariant
    node features once, then iteratively refines atomic positions via
    equivariant message passing. Only positions are output; types are
    never updated or predicted.

    Edge attributes follow the same format as the joint EGNN: a 2-vector
    of [initial_sq_norm, current_sq_norm] per edge, so PosGCN layers can
    be reused directly.
    """

    def __init__(
        self,
        num_atom_types: int = 5,
        num_blocks: int = 9,
        hidden_nf: int = 256,
        aggr_type: str = "sum",
        tanh_coord_updates: bool = True,
        coords_range: float = 15.0,
    ):
        """Initialize the conditional position generator.

        Args:
            num_atom_types: Number of distinct atom type classes (e.g. 5 for QM9).
            num_blocks: Number of PosGCN layers to stack.
            hidden_nf: Hidden feature dimension for type embeddings and messages.
            aggr_type: Aggregation for PosGCN message passing ("sum" or "mean").
            tanh_coord_updates: Whether PosGCN bounds coordinate updates with tanh.
            coords_range: Maximum coordinate displacement magnitude per layer.
        """
        super().__init__()

        # Static atom type embedding: one-hot → invariant dense features.
        # Two-layer MLP gives the network capacity to learn rich type representations
        # without the layers becoming a bottleneck.
        self.type_embedding = nn.Sequential(
            nn.Linear(num_atom_types, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
        )

        # Position-only update blocks. Each block uses the SAME fixed type_feat
        # (computed once from atom types) without any TypeGCN update in between.
        self.blocks = nn.ModuleList([
            PosGCN(
                hidden_nf,
                tanh_coord_updates=tanh_coord_updates,
                coords_range=coords_range,
                aggr_type=aggr_type,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        pos_noisy: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Generate positions conditioned on fixed atom types.

        Args:
            pos_noisy: Noisy initial positions [N, 3].
            atom_types: Fixed one-hot atom type vectors [N, num_atom_types].
            edge_index: Fully-connected graph connectivity [2, E].

        Returns:
            Predicted positions [N, 3].
        """
        # Embed atom types once — these stay constant for the entire forward pass.
        type_feat = self.type_embedding(atom_types.float())  # [N, hidden_nf]

        pos = pos_noisy
        # Store initial pairwise distances as one component of the edge attribute,
        # mirroring the joint EGNN which uses [initial_sq_norm, current_sq_norm].
        initial_sq_norm, _ = compute_edge_properties(pos, edge_index)

        for pos_gcn in self.blocks:
            sq_norm, dir_vec = compute_edge_properties(pos, edge_index)
            # Edge attr shape [E, 2]: same format expected by PosGCN's coord_mlp.
            edge_attr = torch.cat(
                [initial_sq_norm.unsqueeze(-1), sq_norm.unsqueeze(-1)],
                dim=-1,
            )
            pos = pos_gcn(pos, type_feat, edge_index, edge_attr, dir_vec)

        return pos
