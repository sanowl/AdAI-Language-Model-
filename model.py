import torch
import torch.nn as nn
import math

class MambaSSM(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout_rate=0.3):
        super(MambaSSM, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.input_proj = nn.Linear(d_model, d_state * expand)
        self.conv1d = nn.Conv1d(d_state * expand, d_state * expand, kernel_size=3, padding=1, groups=expand)
        self.output_proj = nn.Linear(d_state * expand, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        x_res = x  # Save residual connection
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.output_proj(x)
        x = self.layer_norm1(x)
        
        x = self.dropout(x)
        x_residual = self.residual(x_res)
        x = x + x_residual
        x = self.layer_norm2(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AdAI(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, d_conv, expand, num_layers, dropout_rate=0.3):
        super(AdAI, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.ssms = nn.ModuleList([MambaSSM(d_model, d_state, d_conv, expand, dropout_rate) for _ in range(num_layers)])
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout_rate), num_layers=num_layers
        )
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for ssm in self.ssms:
            x = ssm(x)

        x = x.permute(1, 0, 2)  # Prepare for attention: (seq_length, batch_size, d_model)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.permute(1, 0, 2)  # Restore original shape: (batch_size, seq_length, d_model)
        x = self.transformer_layers(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        x = self.output_layer(x)
        return x
