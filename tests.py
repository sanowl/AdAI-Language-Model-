import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer
from datasets import load_dataset
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu')
logging.info(f"Using device: {device}")

# Load the tokenizer and add a padding token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load a text dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
logging.info(f"Dataset loaded. Size: {len(dataset)}")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'], num_proc=4)
logging.info("Dataset tokenized")

# Convert to PyTorch dataset format
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

train_dataset = TextDataset(tokenized_datasets)
logging.info("PyTorch dataset created")

# Define the model with smaller hyperparameters
class SmallTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_length):
        super(SmallTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, 
                                       dim_feedforward=hidden_dim, activation='gelu', 
                                       batch_first=True),
            num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        embeddings = self.embedding(input_ids) + self.position_embedding(position_ids)
        embeddings = self.dropout(embeddings)

        transformer_output = self.layers(embeddings)
        transformer_output = self.layer_norm(transformer_output)
        logits = self.fc(transformer_output)
        return logits

# Hyperparameters
vocab_size = len(tokenizer)
embedding_dim = 64
num_heads = 2
hidden_dim = 256
num_layers = 2
max_length = 128
batch_size = 32
num_epochs = 5
learning_rate = 0.001

# Data loader
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
logging.info(f"Data loader created. Batch size: {batch_size}")

# Instantiate and move the model to the device
model = SmallTransformerModel(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_length).to(device)
logging.info(f"Model created. Parameters: {sum(p.numel() for p in model.parameters())}")

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

# Initialize TensorBoard writer
writer = SummaryWriter()

# Training loop
logging.info("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, vocab_size), input_ids.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 99:  # Log every 100 batches
            avg_loss = running_loss / 100
            writer.add_scalar('training_loss', avg_loss, epoch * len(data_loader) + i)
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {avg_loss:.4f}')
            running_loss = 0.0

    epoch_time = time.time() - start_time
    logging.info(f'Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds')
    scheduler.step()

# Save the trained model
torch.save(model.state_dict(), 'small_transformer_model.pth')
logging.info("Model saved")

writer.close()

# Text generation function
@torch.no_grad()
def generate_text(prompt, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token = torch.argmax(outputs[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Generate text
prompt = "Once upon a time"
generated_text = generate_text(prompt)
logging.info(f"Generated Text: {generated_text}")