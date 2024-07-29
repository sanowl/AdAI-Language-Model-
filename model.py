import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer
from datasets import load_dataset
import time
import logging
from tqdm import tqdm
import argparse
import os
from typing import List, Dict, Any

# Set environment variable to fall back to CPU for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class TextDataset(Dataset):
    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long)
        }

class SmallTransformerModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, hidden_dim: int, num_layers: int, max_length: int, dropout: float = 0.1):
        super(SmallTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            activation='gelu', 
            batch_first=True,
            dropout=dropout
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        embeddings = self.embedding(input_ids) + self.pos_embedding(position_ids)
        embeddings = self.dropout(embeddings)

        padding_mask = attention_mask.logical_not() if attention_mask is not None else None

        transformer_output = self.layers(embeddings, src_key_padding_mask=padding_mask)
        transformer_output = self.layer_norm(transformer_output)
        logits = self.fc(transformer_output)
        return logits

def tokenize_function(examples: Dict[str, List[str]], tokenizer: GPT2Tokenizer, max_length: int) -> Dict[str, List[int]]:
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

@torch.no_grad()
def generate_text(model: nn.Module, tokenizer: GPT2Tokenizer, prompt: str, device: torch.device, max_length: int = 50, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()
    
    for _ in range(max_length):
        outputs = model(generated)
        next_token_logits = outputs[:, -1, :] / temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probabilities = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0, filter_value: float = -float('Inf')) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, scheduler: OneCycleLR, device: torch.device, num_epochs: int, tokenizer: GPT2Tokenizer, writer: SummaryWriter) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        val_loss = validate_model(model, val_loader, criterion, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_small_transformer_model.pth')
            logging.info("New best model saved")
        
        sample_prompt = "Once upon a time"
        sample_text = generate_text(model, tokenizer, sample_prompt, device)
        logging.info(f"Sample generated text: {sample_text}")
    
    return model

def validate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def chat_with_model(model: nn.Module, tokenizer: GPT2Tokenizer, device: torch.device):
    print("Chat with the model (type 'quit' to exit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        try:
            response = generate_text(model, tokenizer, user_input, device, max_length=100)
            print("Model:", response)
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            print("Model: I'm sorry, I couldn't generate a response. Please try again.")

def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        logging.info(f"Dataset loaded. Size: {len(dataset)}")
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return

    max_length = args.max_length
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length), 
        batched=True, 
        remove_columns=['text']
    )
    logging.info("Dataset tokenized")

    full_dataset = TextDataset(tokenized_datasets)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    logging.info("PyTorch datasets created")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logging.info(f"Data loaders created. Batch size: {args.batch_size}")

    model = SmallTransformerModel(len(tokenizer), args.embedding_dim, args.num_heads, args.hidden_dim, args.num_layers, max_length).to(device)
    logging.info(f"Model created. Parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=len(train_loader) * args.num_epochs)

    writer = SummaryWriter()

    logging.info("Starting training...")
    try:
        trained_model = train_model(model, train_loader, val_loader, optimizer, scheduler, device, args.num_epochs, tokenizer, writer)
        torch.save(trained_model.state_dict(), 'final_small_transformer_model.pth')
        logging.info("Final model saved")
        
        # Validate the model before starting the chat
        val_loss = validate_model(trained_model, val_loader, nn.CrossEntropyLoss(), device)
        logging.info(f"Final validation loss: {val_loss:.4f}")
        
        chat_with_model(trained_model, tokenizer, device)
    except Exception as e:
        logging.error(f"An error occurred during training or chat: {str(e)}")
    finally:
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a small transformer model")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()
    
    main(args)