"""Equivariant graph neural network layers and generator model.

This module defines message-passing networks for atom type and coordinate
updates, along with the EGNN model used for molecular generation.
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class TypeGCN(MessagePassing):
    """Graph convolution network for updating discrete atom type features.

    Passes messages between neighbouring nodes using their type embeddings and
    edge attributes, then applies an MLP update with optional attention gating.
    """

    propagate_type = {"type_feat": torch.Tensor, "edge_attr": torch.Tensor}

    def __init__(
        self, hidden_nf: int, attention: bool = True, aggr_type: str = "sum"
    ):
        """Initialize a graph convolution network for discrete type updates.

        Args:
            hidden_nf: Dimension of hidden node features.
            attention: Whether to apply attention weighting to messages.
            aggr_type: Aggregation type for message passing.
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
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )

    def message(
        self,
        type_feat_i: torch.Tensor,
        type_feat_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Compute one-step messages between connected node pairs.

        Args:
            type_feat_i: Feature vector for the destination node.
            type_feat_j: Feature vector for the source node.
            edge_attr: Edge features describing the connection.

        Returns:
            Updated message tensor used for node aggregation.
        """
        input_tensor = torch.cat(
            [type_feat_i, type_feat_j, edge_attr],
            dim=-1,
        )
        out = self.message_mlp(input_tensor)

        if self.attention:
            att_val = self.att_mlp(out)
            return out * att_val

        return out

    def update(self, aggr_out: torch.Tensor, type_feat: torch.Tensor) -> torch.Tensor:
        """Update node type features after message aggregation.

        Args:
            aggr_out: Aggregated messages for each node.
            type_feat: Previous type features for each node.

        Returns:
            Updated node type features.
        """
        out = torch.cat([aggr_out, type_feat], dim=-1)
        return type_feat + self.update_mlp(out)

    def forward(
        self, type_feat: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Run one message-passing step for discrete type prediction.

        Args:
            type_feat: Node type features.
            edge_index: Graph connectivity in COO format.
            edge_attr: Edge features for each edge.

        Returns:
            Updated node type features after propagation.
        """
        return self.propagate(
            edge_index, type_feat=type_feat, edge_attr=edge_attr
        )


class PosGCN(MessagePassing):
    """Equivariant coordinate update network for computing position shifts.

    Aggregates direction-weighted messages from neighbours to produce a
    translation-equivariant position delta for each node.
    """

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
        """Initialize a coordinate update block for equivariant position shifts.

        Args:
            hidden_nf: Hidden feature dimension for coordinate messages.
            tanh_coord_updates: Whether to bound coordinate updates with tanh.
            coords_range: Maximum coordinate displacement magnitude.
            aggr_type: Aggregation type for message passing.
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
        """Compute coordinate update messages for each edge.

        Args:
            type_feat_i: Feature vector for the destination node.
            type_feat_j: Feature vector for the source node.
            edge_attr: Edge features describing the edge.
            scaled_dir_vector: Scaled direction vectors between nodes.

        Returns:
            Position update contributions per edge.
        """
        input_tensor = torch.cat([type_feat_i, type_feat_j, edge_attr], dim=-1)
        weight = self.coord_mlp(input_tensor)
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
        """Compute updated positions using equivariant coordinate message passing.

        Args:
            pos: Current node positions.
            type_feat: Node feature embeddings.
            edge_index: Graph connectivity.
            edge_attr: Edge features.
            scaled_dir_vector: Scaled direction vectors between nodes.

        Returns:
            Updated node positions.
        """
        delta = self.propagate(
            edge_index,
            type_feat=type_feat,
            edge_attr=edge_attr,
            scaled_dir_vector=scaled_dir_vector,
        )
        return pos + delta


class EquivariantBlock(nn.Module):
    """A block that alternates type and position updates in the EGNN.

    Each block applies several TypeGCN layers followed by a positional update
    through PosGCN.
    """

    def __init__(
        self,
        hidden_nf: int,
        n_layers: int = 1,
        attention: bool = True,
        tanh_coord_updates: bool = True,
        coords_range: float = 15.0,
        aggr_type: str = "sum",
    ):
        """Initialize the equivariant block with type and coordinate update layers.

        Args:
            hidden_nf: Hidden feature dimension.
            n_layers: Number of TypeGCN layers per block.
            attention: Whether TypeGCN layers use attention gating.
            tanh_coord_updates: Whether PosGCN bounds updates with tanh.
            coords_range: Maximum coordinate update magnitude.
            aggr_type: Aggregation type for message passing.
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
        """Apply the equivariant block to update features and positions.

        Args:
            type_feat: Current node type features.
            pos: Current node positions.
            edge_index: Graph connectivity.
            edge_attr: Edge features.
            scaled_dir_vector: Scaled direction vectors used for position updates.

        Returns:
            A tuple of updated node type features and updated positions.
        """
        for type_gcn in self.type_update:
            type_feat = type_gcn(
                type_feat=type_feat,
                edge_index=edge_index,
                edge_attr=edge_attr
            )

        pos = self.coord_update(
            pos=pos,
            type_feat=type_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            scaled_dir_vector=scaled_dir_vector,
        )

        return type_feat, pos

def compute_edge_properties(
    pos: torch.Tensor, edge_index: torch.Tensor, norm_constant: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute squared distances and normalized direction vectors for edges.

    Args:
        pos: Node positions tensor of shape [N, 3].
        edge_index: Graph edge index in COO format.
        norm_constant: Small constant to stabilize division during normalization.

    Returns:
        squared_norm: Squared distances for each edge.
        scaled_dir_vector: Normalized direction vectors for each edge.
    """
    src, dst = edge_index
    dist = pos[src] - pos[dst]
    squared_norm = dist.pow(2).sum(dim=-1)
    norm = squared_norm.sqrt()
    scaled_dir_vector = dist / (norm.unsqueeze(-1) + norm_constant)
    return squared_norm, scaled_dir_vector

class EGNN(nn.Module):
    """Equivariant GNN generator model for molecular geometry and type prediction.

    The model alternates type and coordinate update blocks to produce new atomic
    positions and type logits from noisy inputs.
    """

    def __init__(
        self,
        num_atom_types: int,
        num_blocks: int = 9,
        hidden_nf: int = 256,
        num_layers_per_block: int = 1,
        attention: bool = True,
        tanh_coord_updates: bool = True,
        coords_range: float = 15.0,
        aggr_type: str = "sum",
    ):
        """Initialize the EGNN generator.

        Args:
            num_atom_types: Number of distinct atom type classes.
            num_blocks: Number of EquivariantBlock layers to stack.
            hidden_nf: Hidden feature dimension.
            num_layers_per_block: Number of TypeGCN layers inside each block.
            attention: Whether TypeGCN layers use attention gating.
            tanh_coord_updates: Whether PosGCN bounds coordinate updates with tanh.
            coords_range: Maximum coordinate update magnitude per block.
            aggr_type: Aggregation type for message passing.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate updated positions and type logits from noisy molecular inputs.

        Args:
            pos_noise: Noisy initial positions of shape [N, 3].
            type_noise: One-hot or embedding tensor for atom types.
            edge_index: Graph connectivity in COO format.
            return_change_in_pos: If True, also return intermediate position history.

        Returns:
            2-tuple (gen_pos, gen_types) when return_change_in_pos is False;
            3-tuple (gen_pos, gen_types, pos_list) when return_change_in_pos is True,
            where pos_list is a list of intermediate position tensors.
        """

        gen_feats = self.type_embedding(type_noise)
        gen_pos = pos_noise
        initial_squared_norm, _ = compute_edge_properties(gen_pos, edge_index)

        pos_list = []
        if return_change_in_pos:
            pos_list.append(gen_pos.clone())

        for block in self.blocks:
            squared_norm, scaled_dir_vector = compute_edge_properties(gen_pos, edge_index)
            edge_attr = torch.cat(
                [initial_squared_norm.unsqueeze(-1), squared_norm.unsqueeze(-1)],
                dim=-1,
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