

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class SensorAttention(nn.Module):
    def __init__(self, n_filters, kernel_size, dilation_rate,embd):
        super(SensorAttention, self).__init__()
        self.conv_1 = nn.Conv2d(1, n_filters, kernel_size=kernel_size, dilation=dilation_rate, padding='same')
        self.conv_f = nn.Conv2d(n_filters, 1, kernel_size=1, padding='same')
        self.ln = nn.LayerNorm(embd)  # Adjust normalized_shape as needed
    def forward(self, x):
        x = self.ln(x)
        x1 = x.unsqueeze(1)
        x1 = F.relu(self.conv_1(x1))
        x1 = self.conv_f(x1)
        x1 = F.softmax(x1, dim=2)
        x1 = x1.view(x.shape)
        return x * x1, x1



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    


class AttentionWithContext(nn.Module):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports variable-length sequences and masks.
    Input shape:
        3D tensor with shape `(batch, steps, features)`.
    Output shape:
        2D tensor with shape `(batch, features)`.
    """

    def __init__(self, feature_dim, bias=True, return_attention=False):
        super(AttentionWithContext, self).__init__()
        self.return_attention = return_attention

        # Linear transformation layers
        self.linear_uit = nn.Linear(feature_dim, feature_dim, bias=bias)  # Equivalent to W and b
        self.context_vector = nn.Linear(feature_dim, 1, bias=False)  # Equivalent to u

        # Xavier Uniform Initialization
        nn.init.xavier_uniform_(self.linear_uit.weight)
        nn.init.xavier_uniform_(self.context_vector.weight)

        if bias:
            nn.init.zeros_(self.linear_uit.bias)  # Initialize bias as zeros

    def forward(self, x, mask=None):
        """
        x: Input tensor with shape `(batch, steps, features)`
        mask: Optional mask tensor with shape `(batch, steps)`. Values should be 1 for valid steps and 0 for padding.
        """
        # Apply linear transformation and tanh activation
        uit = torch.tanh(self.linear_uit(x))  # Shape: (batch, steps, features)

        # Compute attention scores using the context vector
        ait = self.context_vector(uit).squeeze(-1)  # Shape: (batch, steps)

        # Apply mask (if provided)
        if mask is not None:
            ait = ait.masked_fill(mask == 0, -1e9)  # Mask invalid steps

        # Compute normalized attention scores
        attention = F.softmax(ait, dim=-1)  # Shape: (batch, steps)

        # Compute weighted sum of input features
        attention = attention.unsqueeze(-1)  # Shape: (batch, steps, 1)
        weighted_input = x * attention  # Shape: (batch, steps, features)
        output = weighted_input.sum(dim=1)  # Shape: (batch, features)

        if self.return_attention:
            return output, attention.squeeze(-1)
        return output
