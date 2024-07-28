import os
import logging
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from model import AdvancedAdAI  # Ensure the model is correctly imported

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdsDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

def collate_fn(batch, padding_value, device):
    return pad_sequence(batch, batch_first=True, padding_value=padding_value).to(device)

def load_data(vocab_path, tensors_path, device):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    ad_tensors = torch.load(tensors_path)
    ad_tensors = [tensor.to(device) for tensor in ad_tensors]
    return vocab, ad_tensors

def train_one_epoch(model, train_dataloader, criterion, optimizer, scaler, vocab_size, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output, _ = model(batch)
            loss = criterion(output.view(-1, vocab_size), batch.view(-1))
        
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(output, -1)
        correct_predictions += (predicted == batch).sum().item()
        total_predictions += batch.numel()

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def validate(model, val_dataloader, criterion, vocab_size, device):
    model.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in val_dataloader:
            output, _ = model(batch)
            loss = criterion(output.view(-1, vocab_size), batch.view(-1))
            total_val_loss += loss.item()
            _, predicted = torch.max(output, -1)
            correct_predictions += (predicted == batch).sum().item()
            total_predictions += batch.numel()

    avg_val_loss = total_val_loss / len(val_dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_val_loss, accuracy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Define parameters directly
    vocab_path = 'vocab.pkl'
    tensors_path = 'ad_tensors.pt'
    d_model = 512
    d_state = 256
    d_conv = 3
    expand = 4
    num_layers = 8
    dropout_rate = 0.3
    batch_size = 32
    learning_rate = 0.0001
    weight_decay = 1e-5
    max_lr = 0.001
    epochs = 20
    patience = 7
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'
    save_interval = 5

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    vocab, ad_tensors = load_data(vocab_path, tensors_path, device)
    train_tensors, val_tensors = train_test_split(ad_tensors, test_size=0.2, random_state=42)

    train_dataset = AdsDataset(train_tensors)
    val_dataset = AdsDataset(val_tensors)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, vocab['<pad>'], device))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, vocab['<pad>'], device))

    model = AdvancedAdAI(len(vocab), d_model, d_state, d_conv, expand, num_layers, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dataloader), epochs=epochs)
    scaler = torch.cuda.amp.GradScaler()

    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logging.info(f"Model Parameters: {count_parameters(model)}")

    # Hyperparameter logging
    writer.add_hparams({
        'd_model': d_model,
        'd_state': d_state,
        'd_conv': d_conv,
        'expand': expand,
        'num_layers': num_layers,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'max_lr': max_lr,
        'epochs': epochs,
        'patience': patience
    }, {})

    best_val_loss = float('inf')
    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        avg_train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, scaler, len(vocab), device)
        avg_val_loss, val_acc = validate(model, val_dataloader, criterion, len(vocab), device)

        logging.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'ad_language_model_best.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                logging.info("Early stopping triggered due to no improvement in validation loss.")
                break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'ad_language_model_best_accuracy.pth'))

        scheduler.step()

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'ad_language_model_epoch_{epoch + 1}.pth'))

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'ad_language_model_final.pth'))
    logging.info("Training complete.")

if __name__ == "__main__":
    main()
