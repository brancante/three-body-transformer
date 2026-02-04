"""
Transformer Model for 3-Body Problem Prediction

Uses self-attention to model interactions between bodies and across time.
The hypothesis: attention mechanisms can learn gravitational interactions
and predict future states.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BodyEmbedding(nn.Module):
    """
    Embeds each body's state (x, y, z, vx, vy, vz) into a higher dimensional space.
    Also adds learnable body-specific embeddings.
    """
    
    def __init__(self, input_dim=6, embed_dim=64, n_bodies=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.body_embeddings = nn.Embedding(n_bodies, embed_dim)
        self.n_bodies = n_bodies
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, n_bodies, 6)
        
        Returns:
            Tensor of shape (batch, seq_len, n_bodies, embed_dim)
        """
        batch_size, seq_len, n_bodies, _ = x.shape
        
        # Linear projection
        embedded = self.linear(x)  # (batch, seq_len, n_bodies, embed_dim)
        
        # Add body-specific embeddings
        body_ids = torch.arange(n_bodies, device=x.device)
        body_emb = self.body_embeddings(body_ids)  # (n_bodies, embed_dim)
        embedded = embedded + body_emb.unsqueeze(0).unsqueeze(0)
        
        return embedded


class ThreeBodyTransformer(nn.Module):
    """
    Transformer model for predicting 3-body trajectories.
    
    Architecture:
    1. Body Embedding: Project 6D state to embed_dim
    2. Temporal Position Encoding: Add time information
    3. Body Attention: Self-attention across bodies (learn interactions)
    4. Temporal Attention: Self-attention across timesteps
    5. Prediction Head: Predict next state for all bodies
    """
    
    def __init__(
        self,
        input_dim=6,
        embed_dim=128,
        n_heads=8,
        n_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        n_bodies=3,
        seq_len=10
    ):
        super().__init__()
        
        self.n_bodies = n_bodies
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Body state embedding
        self.body_embedding = BodyEmbedding(input_dim, embed_dim, n_bodies)
        
        # Positional encoding for time (dimension matches concatenated body features)
        self.pos_encoding = PositionalEncoding(embed_dim * n_bodies, max_len=seq_len + 10, dropout=dropout)
        
        # Body interaction transformer (attention across bodies)
        self.body_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers // 2
        )
        
        # Temporal transformer (attention across time)
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim * n_bodies,
                nhead=n_heads,
                dim_feedforward=dim_feedforward * n_bodies,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers // 2
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim * n_bodies, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_bodies * input_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, n_bodies, 6)
               Contains 10 timesteps of body states
        
        Returns:
            Predicted next state: (batch, n_bodies, 6)
        """
        batch_size, seq_len, n_bodies, state_dim = x.shape
        
        # Embed body states
        embedded = self.body_embedding(x)  # (batch, seq_len, n_bodies, embed_dim)
        
        # Apply body-wise attention for each timestep
        # Reshape to process all timesteps at once
        embedded_flat = embedded.view(batch_size * seq_len, n_bodies, self.embed_dim)
        body_attended = self.body_transformer(embedded_flat)
        body_attended = body_attended.view(batch_size, seq_len, n_bodies, self.embed_dim)
        
        # Concatenate body features for temporal processing
        temporal_input = body_attended.view(batch_size, seq_len, n_bodies * self.embed_dim)
        
        # Add positional encoding
        temporal_input = self.pos_encoding(temporal_input)
        
        # Apply temporal attention
        temporal_output = self.temporal_transformer(temporal_input)  # (batch, seq_len, n_bodies * embed_dim)
        
        # Use last timestep for prediction
        last_hidden = temporal_output[:, -1, :]  # (batch, n_bodies * embed_dim)
        
        # Predict next state
        prediction = self.prediction_head(last_hidden)  # (batch, n_bodies * 6)
        prediction = prediction.view(batch_size, n_bodies, state_dim)
        
        return prediction


class ThreeBodyTransformerV2(nn.Module):
    """
    Alternative architecture: Joint space-time attention.
    Treats each (timestep, body) pair as a token.
    """
    
    def __init__(
        self,
        input_dim=6,
        embed_dim=128,
        n_heads=8,
        n_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        n_bodies=3,
        seq_len=10
    ):
        super().__init__()
        
        self.n_bodies = n_bodies
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_tokens = seq_len * n_bodies
        
        # State projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Learnable embeddings for body identity and time position
        self.body_embeddings = nn.Embedding(n_bodies, embed_dim)
        self.time_embeddings = nn.Embedding(seq_len, embed_dim)
        
        # Main transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Prediction head: from all tokens to next states
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim * self.n_tokens, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_bodies * input_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_bodies, 6)
        
        Returns:
            (batch, n_bodies, 6)
        """
        batch_size, seq_len, n_bodies, state_dim = x.shape
        
        # Flatten to tokens: (batch, seq_len * n_bodies, 6)
        x_flat = x.view(batch_size, seq_len * n_bodies, state_dim)
        
        # Project states
        embedded = self.input_projection(x_flat)  # (batch, n_tokens, embed_dim)
        
        # Add body and time embeddings
        body_ids = torch.arange(n_bodies, device=x.device).repeat(seq_len)  # [0,1,2,0,1,2,...]
        time_ids = torch.arange(seq_len, device=x.device).repeat_interleave(n_bodies)  # [0,0,0,1,1,1,...]
        
        body_emb = self.body_embeddings(body_ids)  # (n_tokens, embed_dim)
        time_emb = self.time_embeddings(time_ids)  # (n_tokens, embed_dim)
        
        embedded = embedded + body_emb.unsqueeze(0) + time_emb.unsqueeze(0)
        
        # Apply transformer
        output = self.transformer(embedded)  # (batch, n_tokens, embed_dim)
        
        # Flatten and predict
        output_flat = output.view(batch_size, -1)  # (batch, n_tokens * embed_dim)
        prediction = self.prediction_head(output_flat)
        prediction = prediction.view(batch_size, n_bodies, state_dim)
        
        return prediction


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    batch_size = 4
    seq_len = 10
    n_bodies = 3
    state_dim = 6
    
    x = torch.randn(batch_size, seq_len, n_bodies, state_dim)
    
    print("Testing ThreeBodyTransformer (V1)...")
    model_v1 = ThreeBodyTransformer()
    out_v1 = model_v1(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_v1.shape}")
    print(f"  Parameters: {count_parameters(model_v1):,}")
    
    print("\nTesting ThreeBodyTransformerV2...")
    model_v2 = ThreeBodyTransformerV2()
    out_v2 = model_v2(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_v2.shape}")
    print(f"  Parameters: {count_parameters(model_v2):,}")
