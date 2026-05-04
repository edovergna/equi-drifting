import torch
import torch.nn as nn


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

        print("EPT Initialized.")

    def forward(self, pos, a_soft, batch_vec, dense_edge_index):
        """
        pos: [N, 3] 3D coordinates
        a_soft: [N, 5] Continuous atom probabilities
        batch_vec: [N] PyG batch assignments
        dense_edge_index: [2, E] Fully connected edges
        """
        N_total = pos.shape[0]
        current_device = pos.device

        # This is to bypass the non-differentiable nn.Embedding layer (need for self.embed_weigths)
        h_continuous = a_soft @ self.embed_weights[:5, :]

        # block_vec needed to split
        block_vec = torch.arange(N_total, device=current_device)

        # dummy_edge_attr fills the 64-dim requirement the EPT expects for bonds
        dummy_edge_attr = torch.zeros(
            (dense_edge_index.shape[1], 64), device=current_device
        )

        _, _, graph_repr, _ = self.ept_model.encoder(
            H=h_continuous,
            Z=pos,
            block_id=block_vec,
            batch_id=batch_vec,
            edges=dense_edge_index,
            edge_attr=dummy_edge_attr,
        )

        # Returns a [Batch_Size, 512] vector purely representing 3D structure and chemistry
        return graph_repr
