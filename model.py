import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdvancedMambaSSM(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout_rate=0.3):
        super(AdvancedMambaSSM, self).__init__()
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
        
        # Improved residual block
        self.residual = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_rate)
        )
        
        # Gating mechanism
        self.gate = nn.Linear(d_model * 2, d_model)

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
        
        # Apply gating mechanism
        gate = torch.sigmoid(self.gate(torch.cat([x, x_residual], dim=-1)))
        x = gate * x + (1 - gate) * x_residual
        
        x = self.layer_norm2(x)
        return x

class AdvancedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(AdvancedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
        # Learnable scale parameter
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class AdvancedAdAI(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, d_conv, expand, num_layers, dropout_rate=0.3):
        super(AdvancedAdAI, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = AdvancedPositionalEncoding(d_model)
        self.ssms = nn.ModuleList([AdvancedMambaSSM(d_model, d_state, d_conv, expand, dropout_rate) for _ in range(num_layers)])
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=4*d_model, dropout=dropout_rate, activation='gelu')
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Memory mechanism
        self.memory = nn.Parameter(torch.randn(1, 1, d_model))
        self.memory_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Expand memory to match batch size
        batch_size = x.size(0)
        memory = self.memory.expand(batch_size, -1, -1)
        
        for ssm in self.ssms:
            x = ssm(x)
        
        x = torch.cat([memory, x], dim=1)  # Prepend memory to sequence
        x = x.permute(1, 0, 2)  # Prepare for attention: (seq_length, batch_size, d_model)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.permute(1, 0, 2)  # Restore original shape: (batch_size, seq_length, d_model)
        
        # Update memory
        memory = self.memory_proj(x[:, 0, :]).unsqueeze(1)
        x = x[:, 1:, :]  # Remove memory from sequence
        
        x = self.transformer_layers(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)
        return x, memory

    def generate_text(self, input_ids, vocab, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
        generated = input_ids
        memory = self.memory.expand(1, -1, -1)  # Initialize memory
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs, memory = self.forward(generated)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k filtering
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat((generated, next_token), dim=1)
                
                if next_token.item() == vocab.stoi['<eos>']:
                    break
        
        return generated

# Example usage
vocab_size = 30000
d_model = 512
d_state = 64
d_conv = 3
expand = 2
num_layers = 6

model = AdvancedAdAI(vocab_size, d_model, d_state, d_conv, expand, num_layers)