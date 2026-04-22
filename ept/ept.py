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
        if not hasattr(
            encoder_target, "use_ieconv"
        ):  # because we use fully connected graph
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
            ):  # prep for gradient flow
                self.embed_weights = module.embedding.weight
                break

        if self.embed_weights is None:
            raise ValueError("Could not find the atom (unit) embedding layer!")

        print("EPT Initialized.")

    def encode(self, x_pred, a_soft, c_pred, batch_vec, fc_edges):
        """
        Universal Encoding Pass:
        Works for a single molecule or a large batch.
        """
        N_total = x_pred.shape[0]

        # 1. Differentiable Continuous Node Embeddings
        h_continuous = a_soft @ self.embed_weights[:5, :]

        # 2. Assign unique Block IDs for the attention hierarchy
        block_vec = torch.arange(N_total, device=self.device)

        # 3. 64-dim dummy edges to satisfy the Transformer MLP
        dummy_edge_attr = torch.zeros((fc_edges.shape[1], 64), device=self.device)

        # 4. Forward pass through EPT Backbone
        _, _, graph_repr, _ = self.ept_model.encoder(
            H=h_continuous,
            Z=x_pred,
            block_id=block_vec,
            batch_id=batch_vec,
            edges=fc_edges,
            edge_attr=dummy_edge_attr,
        )

        # 5. Charge Polarity Feature (Std Dev per molecule)
        num_mols = batch_vec.max().item() + 1
        c_std = torch.zeros(num_mols, 1, device=self.device)
        for i in range(num_mols):
            mask = batch_vec == i
            if mask.sum() > 1:
                c_std[i] = c_pred[mask].std()

        # Final phi vector [Batch_Size, 513]
        return torch.cat([graph_repr, c_std], dim=-1)
