import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import DeepEYEModel  # assuming your model is saved in model.py
import os

# === CONFIG ===
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "training.npy"

# === Custom Dataset ===
class VideoDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        # data shape: (227, 227, N)
        frames = self.data.shape[2]
        frames = frames - frames % 10  # make divisible by 10
        self.data = self.data[:, :, :frames]
        self.data = self.data.reshape(227, 227, -1, 10)  # shape: (227, 227, num_samples, 10)
        self.data = np.transpose(self.data, (2, 3, 0, 1))  # (num_samples, 10, 227, 227)
        self.data = np.expand_dims(self.data, axis=1)     # (num_samples, 1, 10, 227, 227)

        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # input == target for reconstruction

# === Load Data ===
dataset = VideoDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model, Loss, Optimizer ===
model = DeepEYEModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Training Loop ===
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(DEVICE)  # (B, 1, 10, 227, 227)
        y = y.to(DEVICE)

        x = x.permute(0, 1, 3, 4, 2)  # (B, 1, 227, 227, 10) -> match model input
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

# === Save Model ===
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/deepeye_model.pth")
print("Model saved to model/deepeye_model.pth")
