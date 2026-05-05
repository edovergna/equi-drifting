import torch
import torch.nn as nn

# EPT atom-vocab indices for the 5 QM9 atom types (H, C, N, O, F).
# VOCAB.idx2atom = [pad, mask, global] + periodic_table_uppercase, so:
#   H=3, He=4, Li=5, Be=6, B=7, C=8, N=9, O=10, F=11
# These must match the column order of a_soft: col 0=H, 1=C, 2=N, 3=O, 4=F.
_QM9_EPT_ATOM_INDICES = [3, 8, 9, 10, 11]


class EPTFeatureExtractor(nn.Module):
    def __init__(self, ckpt_path, device):
        super().__init__()
        self.device = device
        print(f"Loading EPT weights from {ckpt_path}...")

        # Load the checkpoint
        self.ept_model = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.ept_model.eval()

        # Patch version mismatches in the Transformer
        encoder_target = (
            self.ept_model.encoder.encoder
            if hasattr(self.ept_model.encoder, "encoder")
            else self.ept_model.encoder
        )
        if not hasattr(encoder_target, "use_ieconv"):
            encoder_target.use_ieconv = False
        if not hasattr(encoder_target, "zero_conv"):
            encoder_target.zero_conv = False

        for module in self.ept_model.modules():
            if hasattr(module, "efficient"):
                module.efficient = False

        # Freeze weights for use as an invariant extractor
        for param in self.ept_model.parameters():
            param.requires_grad = False

        # Locate the continuous embedding weights
        self.embed_weights = None
        for module in self.ept_model.graph_constructor.node_modules:
            if (
                type(module).__name__ == "ContinuousEmbedding"
                and module.level == "unit"
            ):
                self.embed_weights = module.embedding.weight
                break

        if self.embed_weights is None:
            raise ValueError("Could not find the atom embedding layer!")

        # Extract edge embedding and RadialEdge cutoffs from the checkpoint so we can
        # compute proper distance-based edge types at forward time.
        # EPT edge types are assigned purely by distance (no bond topology needed):
        #   type 0: dist >= topo_cutoff  (non-bonded)
        #   type 1: dist <  topo_cutoff  (bonded-range)
        #   type 2: cross-segment (only when scope='both'; not applicable here)
        gc = self.ept_model.graph_constructor
        self.edge_embed = gc.edge_embed if gc.edge_embed_size > 0 else None
        self.edge_embed_size = gc.edge_embed_size if gc.edge_embed_size > 0 else 64

        self.topo_cutoff = None
        for edge_module in gc.edge_modules:
            if type(edge_module).__name__ == "RadialEdge":
                self.topo_cutoff = edge_module.topo_cutoff
                break

        self.register_buffer(
            "qm9_ept_indices",
            torch.tensor(_QM9_EPT_ATOM_INDICES, dtype=torch.long),
        )

        print(
            f"EPT Initialized. "
            f"edge_embed_size={self.edge_embed_size}, topo_cutoff={self.topo_cutoff}"
        )

    def _compute_edge_attr(
        self, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Assign distance-based edge types and embed them using the checkpoint's
        edge_embed, matching how GraphConstructor builds edge_attr at training time.

        If edge_embed is not available (edge_embed_size=0) or topo_cutoff was not
        found, falls back to zeros.
        """
        if self.edge_embed is None or self.topo_cutoff is None or self.topo_cutoff <= 0:
            return torch.zeros(
                edge_index.shape[1], self.edge_embed_size, device=pos.device
            )

        src, dst = edge_index
        dist = torch.norm(pos[src] - pos[dst], dim=-1)  # [E]
        # Type 0: non-bonded (dist >= topo_cutoff), Type 1: bonded-range (dist < topo_cutoff)
        edge_types = (dist < self.topo_cutoff).long()  # [E]
        return self.edge_embed(edge_types)  # [E, edge_embed_size]

    def forward(
        self,
        pos: torch.Tensor,
        a_soft: torch.Tensor,
        block_id: torch.Tensor,
        batch_id: torch.Tensor,
        dense_edge_index: torch.Tensor,
    ):
        """
        pos: [N, 3] 3D coordinates
        a_soft: [N, 5] Continuous atom probabilities (H, C, N, O, F)
        block_id: [N] atom i -> block i (arange, each atom is its own block)
        batch_id: [N] block/atom i -> graph index
        dense_edge_index: [2, E] Fully connected edges (globally indexed)
        """
        # Bypass the non-differentiable nn.Embedding: soft-embed by taking the weighted
        # sum of the EPT atom embeddings for the 5 QM9 types (H, C, N, O, F).
        h_continuous = a_soft @ self.embed_weights[self.qm9_ept_indices, :]

        # Distance-based edge type embedding, matching EPT's RadialEdge training scheme.
        edge_attr = self._compute_edge_attr(pos, dense_edge_index)

        _, _, graph_repr, _ = self.ept_model.encoder(
            H=h_continuous,
            Z=pos,
            block_id=block_id,
            batch_id=batch_id,
            edges=dense_edge_index,
            edge_attr=edge_attr,
        )

        return graph_repr
