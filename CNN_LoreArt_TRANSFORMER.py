# All necessary imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from pytorch_msssim import ssim
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Transformer-based denoiser
class TransformerDenoiser(nn.Module):
    def __init__(self, img_size=64, patch_size=4, emb_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size

        self.linear_proj = nn.Linear(self.patch_dim, emb_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(emb_dim, self.patch_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, self.patch_dim)

        x = self.linear_proj(x) + self.positional_embedding
        x = self.transformer(x)
        x = self.output_proj(x)

        x = x.view(B, int(self.img_size / self.patch_size), int(self.img_size / self.patch_size),
                   self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, 1, H, W)
        x = self.refine(x)
        return x

# Normalization
def normalize_data(data):
    data_min, data_max = data.min(), data.max()
    return (data - data_min) / (data_max - data_min)

# Hybrid loss function
l1 = nn.L1Loss()
def hybrid_loss(pred, target):
    return 0.85 * l1(pred, target) + 0.15 * (1 - ssim(pred, target, data_range=1.0, size_average=True))

# Train function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = hybrid_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += hybrid_loss(pred, y).item()
        val_losses.append(val_loss / len(val_loader))
    return model, train_losses, val_losses

# Plot predictions
def plot_comparison(noisy, clean, pred, index, tag):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(noisy[index, 0], cmap='gray')
    axs[0].set_title("Noisy")
    axs[1].imshow(clean[index, 0], cmap='gray')
    axs[1].set_title("Clean")
    axs[2].imshow(pred[index, 0], cmap='gray')
    axs[2].set_title("Denoised")
    plt.savefig(f"transformer_comparison_{tag}_sample_{index:02d}.png")
    plt.close()

# Run k-fold cross-validation and plot results
def cross_validate_transformer(noisy_data, clean_data, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_train, all_val = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(noisy_data)):
        print(f"Fold {fold + 1}/{k}")
        train_ds = TensorDataset(
            torch.from_numpy(noisy_data[train_idx]).unsqueeze(1).float(),
            torch.from_numpy(clean_data[train_idx]).unsqueeze(1).float()
        )
        val_ds = TensorDataset(
            torch.from_numpy(noisy_data[val_idx]).unsqueeze(1).float(),
            torch.from_numpy(clean_data[val_idx]).unsqueeze(1).float()
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)

        model = TransformerDenoiser().to(device)
        _, train_loss, val_loss = train_model(model, train_loader, val_loader, num_epochs=10)
        all_train.append(train_loss)
        all_val.append(val_loss)

        if fold == 0:
            model.eval()
            val_samples = next(iter(val_loader))[0].to(device)
            preds = model(val_samples).detach().cpu().numpy()
            plot_comparison(noisy_data[val_idx], clean_data[val_idx], preds, 0, tag="val")
            plot_comparison(noisy_data[val_idx], clean_data[val_idx], preds, 1, tag="val")
            train_samples = next(iter(train_loader))[0].to(device)
            preds_train = model(train_samples).detach().cpu().numpy()
            plot_comparison(noisy_data[train_idx], clean_data[train_idx], preds_train, 0, tag="train")
            plot_comparison(noisy_data[train_idx], clean_data[train_idx], preds_train, 1, tag="train")

    avg_train = np.mean(all_train, axis=0)
    avg_val = np.mean(all_val, axis=0)
    plt.plot(avg_train, label="Train Loss")
    plt.plot(avg_val, label="Validation Loss")
    plt.title("Transformer Model Performance")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("transformer_loss_comparison.png")
    plt.close()

# MAIN
if __name__ == "__main__":
    print("Using device =", device)
    noisy_data = np.load("noisy_train_19k.npy").astype(np.float32)
    clean_data = np.load("clean_train_19k.npy").astype(np.float32)

    noisy_data = normalize_data(noisy_data)
    clean_data = normalize_data(clean_data)

    start = time.time()
    cross_validate_transformer(noisy_data, clean_data, k=5)
    end = time.time()
    print("Total training time:", end - start, "seconds")
