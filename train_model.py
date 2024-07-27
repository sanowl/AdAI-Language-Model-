import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from model import AdAI  # Import the model

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Load the tensors
ad_tensors = torch.load('ad_tensors.pt')

# Move tensors to device
ad_tensors = [tensor.to(device) for tensor in ad_tensors]

# Split data into training and validation sets
train_tensors, val_tensors = train_test_split(ad_tensors, test_size=0.2, random_state=42)

# Dataset and DataLoader
class AdsDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=vocab['<pad>']).to(device)

# Create dataset and dataloader
train_dataset = AdsDataset(train_tensors)
val_dataset = AdsDataset(val_tensors)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Hyperparameters
vocab_size = len(vocab)
d_model = 512
d_state = 256
d_conv = 3
expand = 4
num_layers = 8
learning_rate = 0.0001
epochs = 20
patience = 7  # Early stopping patience

# Initialize model
model = AdAI(vocab_size, d_model, d_state, d_conv, expand, num_layers).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay for regularization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)  # Learning rate scheduler

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Early stopping variables
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()

        # Forward pass
        with torch.cuda.amp.autocast():
            output = model(batch)
            loss = criterion(output.view(-1, vocab_size), batch.view(-1))
        
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)

    # Validation step
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            output = model(batch)
            loss = criterion(output.view(-1, vocab_size), batch.view(-1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_dataloader)

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
    
    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'ad_language_model_best.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping triggered")
            break
    
    scheduler.step(avg_val_loss)  # Adjust the learning rate based on validation loss

# Save the final model
torch.save(model.state_dict(), 'ad_language_model_final.pth')
