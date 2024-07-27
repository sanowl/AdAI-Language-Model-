import argparse
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
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
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

def collate_fn(batch, padding_value):
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

def validate(model, val_dataloader, criterion, vocab_size):
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

def main(args):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")

    vocab, ad_tensors = load_data(args.vocab_path, args.tensors_path, device)
    train_tensors, val_tensors = train_test_split(ad_tensors, test_size=0.2, random_state=42)

    train_dataset = AdsDataset(train_tensors)
    val_dataset = AdsDataset(val_tensors)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, vocab['<pad>']))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, vocab['<pad>']))

    model = AdvancedAdAI(len(vocab), args.d_model, args.d_state, args.d_conv, args.expand, args.num_layers, args.dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    log_dir = os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logging.info(f"Model Parameters: {count_parameters(model)}")

    # Hyperparameter logging
    writer.add_hparams({
        'd_model': args.d_model,
        'd_state': args.d_state,
        'd_conv': args.d_conv,
        'expand': args.expand,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_lr': args.max_lr,
        'epochs': args.epochs,
        'patience': args.patience
    }, {})

    best_val_loss = float('inf')
    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        avg_train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, scaler, len(vocab), device)
        avg_val_loss, val_acc = validate(model, val_dataloader, criterion, len(vocab))

        logging.info(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'ad_language_model_best.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve == args.patience:
                logging.info("Early stopping triggered due to no improvement in validation loss.")
                break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'ad_language_model_best_accuracy.pth'))

        scheduler.step()

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'ad_language_model_epoch_{epoch + 1}.pth'))

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'ad_language_model_final.pth'))
    logging.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an AdvancedAdAI language model.")
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the vocabulary file.')
    parser.add_argument('--tensors_path', type=str, required=True, help='Path to the tensor data file.')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('--d_state', type=int, default=256, help='Dimension of the state.')
    parser.add_argument('--d_conv', type=int, default=3, help='Dimension of the convolution.')
    parser.add_argument('--expand', type=int, default=4, help='Expansion factor.')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer.')
    parser.add_argument('--max_lr', type=float, default=0.001, help='Maximum learning rate for OneCycleLR scheduler.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for TensorBoard logs.')
    parser.add_argument('--save_interval', type=int, default=5, help='Interval of epochs to save checkpoints.')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
