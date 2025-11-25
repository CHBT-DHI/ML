import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- Your Dataset and Model ---
from dataset import TimeSeriesNH4Dataset
from model import Transformer
from utils import plot_nh4_predictions

# --- Hyperparameters ---
csv_path = "data/raw_normalized/normalized.csv"
seq_len = 1
batch_size = 256
lr = 5e-4
num_epochs = 50
train_ratio = 0.8
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load dataset ---
dataset = TimeSeriesNH4Dataset(csv_path, seq_len=seq_len)

# Train/val split
train_size = int(len(dataset) * train_ratio)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# --- Initialize model ---
input_dim = dataset.features.shape[1]
model = Transformer(input_dim=input_dim, seq_len=seq_len, n_layers=4).to(device)

# --- Loss and optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Training loop ---
for epoch in range(1, num_epochs+1):
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for x_batch, y_batch in train_bar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)

    train_loss /= len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for x_batch, y_batch in val_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * x_batch.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    plot_nh4_predictions(
        model,
        val_loader
    )
