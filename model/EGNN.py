import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class AtomTypeEncoder(nn.Module):
    """
    Encodes atom type A.

    Accepts either:
    - LongTensor [num_nodes] with type indices in {0, ..., num_atom_types-1}
    - FloatTensor [num_nodes, num_atom_types] as one-hot / soft one-hot
    """

    def __init__(self, num_atom_types: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_atom_types = num_atom_types
        self.emb = nn.Embedding(num_atom_types, hidden_dim)
        self.lin = nn.Linear(num_atom_types, hidden_dim)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        if A.dtype == torch.long and A.dim() == 1:
            return self.emb(A)
        if A.dim() == 2 and A.size(-1) == self.num_atom_types:
            return self.lin(A.float())
        raise ValueError(
            f"A must be [num_nodes] long indices or [num_nodes, {self.num_atom_types}] one-hot; "
            f"got shape {tuple(A.shape)} and dtype {A.dtype}."
        )


class BondEncoder(nn.Module):
    """
    Encodes bond type E.

    Accepts either:
    - LongTensor [num_edges] with bond type indices in {0, ..., num_bond_types-1}
    - FloatTensor [num_edges, num_bond_types] as one-hot / soft one-hot
    """

    def __init__(self, num_bond_types: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_bond_types = num_bond_types
        self.emb = nn.Embedding(num_bond_types, hidden_dim)
        self.lin = nn.Linear(num_bond_types, hidden_dim)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        if E.dtype == torch.long and E.dim() == 1:
            return self.emb(E)
        if E.dim() == 2 and E.size(-1) == self.num_bond_types:
            return self.lin(E.float())
        raise ValueError(
            f"E must be [num_edges] long indices or [num_edges, {self.num_bond_types}] one-hot; "
            f"got shape {tuple(E.shape)} and dtype {E.dtype}."
        )


class EGNNLayer(MessagePassing):
    """
    Single sparse EGNN update over a PyG graph.

    Messages are computed only on the edges in edge_index.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_hidden_dim: int,
        space_dim: int = 3,
        atom_dist_step: float = 0.1,
        atom_feature_step: float = 0.5,
    ) -> None:
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.space_dim = space_dim
        self.atom_dist_step = atom_dist_step
        self.atom_feature_step = atom_feature_step

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:         [num_nodes, hidden_dim]
            X:         [num_nodes, space_dim]
            edge_index:[2, num_edges] with source=edge_index[0], target=edge_index[1]
            edge_feat: [num_edges, edge_hidden_dim]

        Returns:
            new_h:     [num_nodes, hidden_dim]
            new_X:     [num_nodes, space_dim]
        """
        aggr = self.propagate(edge_index=edge_index, h=h, X=X, edge_feat=edge_feat)

        aggr_messages = aggr[:, : self.hidden_dim]  # [num_nodes, hidden_dim]
        aggr_displacement = aggr[:, self.hidden_dim :]  # [num_nodes, space_dim]

        new_X = X + self.atom_dist_step * aggr_displacement

        node_inputs = torch.cat([h, aggr_messages], dim=-1)
        new_h = h + self.atom_feature_step * self.node_mlp(node_inputs)
        new_h = self.norm(new_h)

        return new_h, new_X

    def message(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        X_i: torch.Tensor,
        X_j: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Constructs edge messages from j -> i.
        """
        relative_vectors = X_i - X_j  # [num_edges, space_dim]
        squared_distances = (relative_vectors**2).sum(dim=-1, keepdim=True)

        edge_inputs = torch.cat(
            [h_i, h_j, edge_feat, squared_distances],
            dim=-1,
        )
        messages = self.edge_mlp(edge_inputs)  # [num_edges, hidden_dim]

        push_pull_weights = self.coord_mlp(messages)  # [num_edges, 1]
        coord_messages = relative_vectors * push_pull_weights  # [num_edges, space_dim]

        # Aggregate both node-feature messages and coordinate displacements
        return torch.cat([messages, coord_messages], dim=-1)


class EGNNVelocity(nn.Module):
    """
    EGNN-based velocity field on PyG graphs.

    Inputs:
        X: coordinates                [num_nodes, 3]
        A: atom type                  [num_nodes] or [num_nodes, num_atom_types]
        C: atom charge / atomic num   [num_nodes] or [num_nodes, 1]
        edge_index: COO edges         [2, num_edges]
        E: bond type                  [num_edges] or [num_edges, num_bond_types]
        time: scalar or [num_graphs]
        batch: graph ids              [num_nodes]

    Output:
        {
            "velocity":   [num_nodes, 3],
            "type_logits":[num_nodes, num_atom_types],
        }
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_atom_types: int = 5,
        num_bond_types: int = 4,
        space_dim: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.space_dim = space_dim

        self.atom_encoder = AtomTypeEncoder(num_atom_types, hidden_dim)
        self.charge_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bond_encoder = BondEncoder(num_bond_types, hidden_dim)

        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_hidden_dim=hidden_dim,
                    space_dim=space_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.type_out = nn.Linear(hidden_dim, num_atom_types)

        # QM9 atomic-number -> type-index mapping:
        # H=1 -> 0, C=6 -> 1, N=7 -> 2, O=8 -> 3, F=9 -> 4
        qm9_map = torch.full((10,), -1, dtype=torch.long)
        qm9_map[1] = 0  # Hydrogen
        qm9_map[6] = 1  # Carbon
        qm9_map[7] = 2  # Nitrogen
        qm9_map[8] = 3  # Oxygen
        qm9_map[9] = 4  # Fluor
        self.register_buffer("qm9_z_to_type", qm9_map)

    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        C: torch.Tensor,
        edge_index: torch.Tensor,
        E: torch.Tensor,
        time: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if X.dim() != 2 or X.size(-1) != self.space_dim:
            raise ValueError(
                f"X must have shape [num_nodes, {self.space_dim}], got {tuple(X.shape)}"
            )

        num_nodes = X.size(0)
        device = X.device

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Normalize charge shape to [num_nodes, 1]
        if C.dim() == 1:
            C = C.unsqueeze(-1)
        elif C.dim() != 2 or C.size(-1) != 1:
            raise ValueError(
                f"C must have shape [num_nodes] or [num_nodes, 1], got {tuple(C.shape)}"
            )

        # time: scalar or [num_graphs]
        if time.dim() == 0:
            time = time.view(1).expand(num_graphs)
        elif time.dim() == 1 and time.numel() == 1:
            time = time.expand(num_graphs)
        elif time.dim() == 1 and time.numel() == num_graphs:
            pass
        else:
            raise ValueError(
                f"time must be a scalar or [num_graphs]={num_graphs}, got shape {tuple(time.shape)}"
            )

        h = self.atom_encoder(A)
        h = h + self.charge_encoder(C.float())
        h = h + self.time_encoder(time[batch].unsqueeze(-1).float())

        edge_feat = self.bond_encoder(E)

        X0 = X
        Xcur = X
        hcur = h

        for layer in self.layers:
            hcur, Xcur = layer(hcur, Xcur, edge_index, edge_feat)

        velocity = Xcur - X0

        # Zero center-of-mass velocity per graph
        mean_velocity = scatter(
            velocity, batch, dim=0, reduce="mean"
        )  # [num_graphs, 3]
        velocity = velocity - mean_velocity[batch]

        type_logits = self.type_out(hcur)

        return {
            "velocity": velocity,
            "type_logits": type_logits,
        }

    def qm9_to_inputs(
        self,
        data,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Converts a torch_geometric.datasets.QM9 Data object to (X, A, C, edge_index, E, batch).

        Uses:
            X = data.pos
            A = type index derived from data.z
            C = data.z as scalar charge / atomic number
            E = data.edge_attr
        """
        if not hasattr(data, "pos"):
            raise ValueError("QM9 data object must have .pos")
        if not hasattr(data, "z"):
            raise ValueError("QM9 data object must have .z")
        if not hasattr(data, "edge_index"):
            raise ValueError("QM9 data object must have .edge_index")
        if not hasattr(data, "edge_attr"):
            raise ValueError("QM9 data object must have .edge_attr")

        z = data.z.long()
        if z.max().item() >= self.qm9_z_to_type.numel():
            raise ValueError(f"Unexpected atomic number in z: max={z.max().item()}")

        A = self.qm9_z_to_type[z]
        if (A < 0).any():
            bad = z[A < 0].unique().tolist()
            raise ValueError(f"Unsupported QM9 atomic numbers: {bad}")

        X = data.pos.float()
        C = z.float().unsqueeze(-1)
        edge_index = data.edge_index.long()
        E = data.edge_attr.float()
        batch = getattr(data, "batch", None)

        return X, A, C, edge_index, E, batch
