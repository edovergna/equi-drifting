import torch
import torch.nn as nn
import math
 

# Embedding the input sequence
class NoiseProjection(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


# Self-attention layer
class SelfAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        attn = torch.matmul(query / math.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            # Accept mask shapes: [B, S], [B, 1, S], or [B, 1, 1, S]
            # Convert to boolean and expand to broadcast to attn: [B, num_heads, Q_len, K_len]
            m = mask
            if m.dim() == 2:
                # [B, S] -> [B, 1, 1, S]
                m = m.unsqueeze(1).unsqueeze(1)
            elif m.dim() == 3:
                # [B, 1, S] -> [B, 1, 1, S]
                m = m.unsqueeze(1)
            # ensure boolean (device placement is managed externally)
            m = m.to(dtype=torch.bool)
            attn = attn.masked_fill(~m, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, value)

        return output
        

# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        # Apply the linear projection
        output = self.out(output)
        return output



# Norm layer
class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)


# Transformer encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        # Add and Muti-head attention
        x = x + self.dropout1(self.self_attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))
        return x


# Encoder transformer
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, max_num_atoms, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = NoiseProjection(input_dim, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
        self.norm = Norm(embedding_dim)

    
    def forward(self, noise, noise_mask):
        # Embed the source
        x = self.embedding(noise)

        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, noise_mask)
        # Normalize
        x = self.norm(x)
        return x



# Transformers
class Transformer(nn.Module):
    def __init__(self, input_dim, max_num_atoms, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.max_num_atoms = max_num_atoms
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
    
        self.encoder = Encoder(input_dim, embedding_dim, max_num_atoms, num_heads, num_layers, dropout)
        self.final_linear = nn.Linear(embedding_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, noise, noise_mask):
        # Encoder forward pass
        output = self.encoder(noise, noise_mask)

        # Final linear layer
        output = self.dropout(output)
        output = self.final_linear(output)
        return output
    

def make_noise_mask(max_num_atoms, num_atoms, batch_size=None):
    """
    Create a boolean mask of shape `[batch_size, max_num_atoms]` where the first
    `num_atoms` positions are True for every batch element.

    Args:
        max_num_atoms (int): maximum number of atoms (S).
        num_atoms (int): number of valid atoms per example in the batch.
        batch_size (int, optional): batch size B. Required.

    Returns:
        torch.BoolTensor: mask of shape `[B, S]` with True for valid positions.
    """
    if batch_size is None:
        raise ValueError("batch_size must be provided")

    # allow 0-dim tensors for ints
    if torch.is_tensor(max_num_atoms) and max_num_atoms.dim() == 0:
        max_num_atoms = int(max_num_atoms.item())
    if torch.is_tensor(num_atoms) and num_atoms.dim() == 0:
        num_atoms = int(num_atoms.item())

    if not isinstance(max_num_atoms, int) or not isinstance(num_atoms, int):
        raise ValueError("max_num_atoms and num_atoms must be ints")

    if num_atoms < 0 or num_atoms > max_num_atoms:
        raise ValueError(f"num_atoms must be in [0, {max_num_atoms}], got {num_atoms}")
    idx = torch.arange(max_num_atoms)
    mask = (idx.unsqueeze(0) < num_atoms).expand(batch_size, max_num_atoms).to(torch.bool)
    return mask
