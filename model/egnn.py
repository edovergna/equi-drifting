from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GCL(MessagePassing):
    propagate_type = {"x": torch.Tensor, "edge_attr": torch.Tensor}

    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        edges_in_d: int = 1,
        act_fn: nn.Module = nn.SiLU(),
        attention: bool = False,
        aggregation_method: str = "add",
    ):
        super().__init__(aggr=aggregation_method)  # "add" = sum

        self.attention = attention

        input_edge = input_nf * 2 + edges_in_d

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # x_i = target, x_j = source

        if edge_attr is not None:
            m_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            m_ij = torch.cat([x_i, x_j], dim=-1)

        m_ij = self.edge_mlp(m_ij)

        if self.attention:
            att = self.att_mlp(m_ij)
            m_ij = m_ij * att

        return m_ij

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([x, aggr_out], dim=-1)
        return x + self.node_mlp(out)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


class EquivariantUpdate(MessagePassing):
    propagate_type = {
        "x": torch.Tensor,
        "coord_diff": torch.Tensor,
        "edge_attr": torch.Tensor,
    }

    def __init__(
        self, hidden_nf: int, edges_in_d: int = 1, act_fn: nn.Module = nn.SiLU()
    ):
        super().__init__(aggr="add")

        input_edge = hidden_nf * 2 + edges_in_d

        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False),
        )

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        input_tensor = torch.cat([x_i, x_j, edge_attr], dim=-1)
        weight = self.coord_mlp(input_tensor)  # scalar
        return coord_diff * weight

    def forward(
        self,
        pos: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.propagate(
            edge_index,
            x=x,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
        )
        return pos + delta


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf: int, n_layers: int = 2):
        super().__init__()

        self.gcls = nn.ModuleList(
            [GCL(hidden_nf, hidden_nf, hidden_nf) for _ in range(n_layers)]
        )

        self.coord_update = EquivariantUpdate(hidden_nf)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        coord_diff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for gcl in self.gcls:
            x = gcl(x=x, edge_index=edge_index, edge_attr=edge_attr)

        pos = self.coord_update(
            pos=pos,
            x=x,
            edge_index=edge_index,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
        )

        return x, pos


class EGNN(nn.Module):
    def __init__(self, in_node_nf: int, hidden_nf: int, n_layers: int = 4):
        super().__init__()

        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, in_node_nf)

        self.blocks = nn.ModuleList(
            [EquivariantBlock(hidden_nf) for _ in range(n_layers)]
        )

    def compute_edge_features(
        self, pos: torch.Tensor, edge_index: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        # Get edge directions
        src, dst = edge_index
        # Compute relative distance
        coord_diff = pos[src] - pos[dst]
        radial = (coord_diff**2).sum(dim=-1, keepdim=True)
        norm = radial.sqrt() + eps
        return radial, coord_diff / norm

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x)

        for block in self.blocks:
            edge_attr, coord_diff = self.compute_edge_features(pos, edge_index)
            x, pos = block(x, pos, edge_index, edge_attr, coord_diff)

        x = self.embedding_out(x)
        return x, pos
