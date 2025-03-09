import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super(TransformerEncoder, self).__init__()
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # Transformer layers: Each with Multi-Head Self-Attention and Feedforward layer
        self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        
        # Layer normalization at the end of the encoder
        self.layer_norm = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Get the input shape (batch size, sequence length)
        b, t = x.size()
        
        # Token and positional embeddings
        token_emb = self.token_embedding(x)  # Shape: (b, t, n_embd)
        position_ids = torch.arange(t, dtype=torch.long, device=x.device)  # Shape: (t,)
        pos_emb = self.position_embedding(position_ids).unsqueeze(0).expand(b, t, -1)  # Shape: (b, t, n_embd)
        
        # Add embeddings
        x = token_emb + pos_emb  # Shape: (b, t, n_embd)
        
        # Pass through each transformer block and collect attention maps
        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_maps.append(attn_weights)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Mean pooling to provide embeddings for the classifier
        mean_emb = x.mean(dim=1)  # Shape: (b, n_embd)
        
        return mean_emb, attn_maps  # mean_emb is for the classifier, attn_maps is for sanity checking

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super(TransformerBlock, self).__init__()
        
        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(n_embd, n_head)
        
        # Feedforward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_output, attn_weights = self.attention(x, mask=None)  # No mask for encoder
        x = self.layer_norm1(x + attn_output)
        
        # Feedforward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x, attn_weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super(MultiHeadSelfAttention, self).__init__()
        
        # Ensure n_embd is divisible by n_head
        assert n_embd % n_head == 0
        self.d_k = n_embd // n_head  # Dimension per head
        self.n_head = n_head
        
        # Define linear layers for query, key, and value projections
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        # Output projection
        self.fc_out = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        b, t, d = x.size()
        
        # Linear transformations and split into heads
        q = self.query(x).view(b, t, self.n_head, self.d_k).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_head, self.d_k).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask for decoder to prevent attending to future tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and apply final linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, d)
        output = self.fc_out(attn_output)
        
        return output, attn_weights.mean(dim=1)
   


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super(TransformerDecoder, self).__init__()
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Decoder layers
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_embd, n_head) for _ in range(n_layer)])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(n_embd)
        
        # Output projection to vocab size
        self.fc_out = nn.Linear(n_embd, vocab_size)  # This maps from embedding dim to vocab size
        
    def forward(self, x, y=None):
            b, t = x.size()
            # Token and positional embeddings
            token_emb = self.token_embedding(x)  # Shape: (b, t, n_embd)
            position_ids = torch.arange(t, dtype=torch.long, device=x.device)  # Shape: (t,)
            pos_emb = self.position_embedding(position_ids).unsqueeze(0).expand(b, t, -1)  # Shape: (b, t, n_embd)

            # Add embeddings
            x = token_emb + pos_emb  # Shape: (b, t, n_embd)

            # Masked attention mechanism to prevent peeking at future tokens
            mask = torch.tril(torch.ones((t, t), device=x.device)).unsqueeze(0).unsqueeze(0)

            # Pass through each decoder block
            attn_maps = []
            for layer in self.layers:
                x, attn_weights = layer(x, mask=mask)
                attn_maps.append(attn_weights)

            # Final layer normalization and projection to vocab size
            x = self.layer_norm(x)
            logits = self.fc_out(x)  # Shape: (b, t, vocab_size)
            
            return logits, attn_maps

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super(TransformerDecoderBlock, self).__init__()
        
        # Masked Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(n_embd, n_head)
        
        # Feedforward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 100),  # Hidden dimension of 100 as specified
            nn.ReLU(),
            nn.Linear(100, n_embd),
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        # Apply masked self-attention with the provided mask
        attn_output, attn_weights = self.attention(x, mask)
        x = self.layer_norm1(x + attn_output)
        
        # Feedforward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x, attn_weights
