import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate the attention weights.
    Args:
        q: query shape == (batch, num_heads, seq_len_q, depth)
        k: key shape == (batch, num_heads, seq_len_k, depth)
        v: value shape == (batch, num_heads, seq_len_v, depth_v)
        mask: Optional tensor of shape broadcastable to (batch, num_heads, seq_len_q, seq_len_k)
    Returns:
        output, attention_weights
    """
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)

    # Scale by square root of depth
    dk = q.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    # Add the mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)

    # Compute output
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=True)
        self.wv = nn.Linear(d_model, d_model, bias=True)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose to shape (batch_size, num_heads, seq_len, depth).
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(self.wk(k), batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(self.wv(v), batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Apply scaled dot-product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # Combine heads back into one
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """
    Create a feed-forward network.
    """
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, seq_len, d_model)

        # Feed-forward network
        ffn_output = self.ffn(out1)  # (batch_size, seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, seq_len, d_model)

        return out2
