import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

# Check if MPS (Metal Performance Shaders) is available and set device
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')

# Define the model (same as before)
class AdvancedTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_length):
        super(AdvancedTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, activation='gelu', batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings = self.embedding(input_ids) + self.position_embedding(position_ids)
        embeddings = self.dropout(embeddings)

        transformer_output = embeddings
        for layer in self.layers:
            transformer_output = layer(transformer_output)

        transformer_output = self.layer_norm(transformer_output)
        logits = self.fc(transformer_output)
        return logits

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Hyperparameters (ensure they match what was used during training)
vocab_size = tokenizer.vocab_size  # Use the vocabulary size of the tokenizer
embedding_dim = 256
num_heads = 8
hidden_dim = 1024  # Increased hidden dimension for more parameters
num_layers = 6
max_length = 128

# Instantiate the model
model = AdvancedTransformerModel(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_length).to(device)

# Load the model weights
model.load_state_dict(torch.load('advanced_transformer_model.pth'))

# Set the model to evaluation mode
model.eval()

# Function to generate text
def generate_text(prompt, max_length=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# Generate text
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)
