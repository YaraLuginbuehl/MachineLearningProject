# All necessary imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import os
import time
from pytorch_msssim import ssim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Transformer Denoiser with local CNN encoder and decoder ---
class HybridTransformerDenoiser(nn.Module):
    def __init__(self, img_size=64, patch_size=4, emb_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size

        # Local CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

        self.linear_proj = nn.Linear(self.patch_dim, emb_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(emb_dim, self.patch_dim)

        # Local CNN Decoder
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.encoder(x)
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


# Data normalization
def normalize_data(data):
    data_min, data_max = data.min(), data.max()
    return (data - data_min) / (data_max - data_min)

# SSIM-enhanced loss
l1 = nn.L1Loss()
def hybrid_loss(pred, target):
    return 0.85 * l1(pred, target) + 0.15 * (1 - ssim(pred, target, data_range=1.0, size_average=True))

# Train a single model
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

# Visualize prediction vs ground truth
def plot_comparison(noisy, clean, pred, index, tag):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(noisy[index, 0], cmap='gray')
    axs[0].set_title("Noisy")
    axs[1].imshow(clean[index, 0], cmap='gray')
    axs[1].set_title("Clean")
    axs[2].imshow(pred[index, 0], cmap='gray')
    axs[2].set_title("Denoised")
    plt.savefig(f"comparison_{tag}_sample_{index:02d}.png")
    plt.close()

# Run k-fold cross-validation and compare configs
def cross_validate_and_compare(noisy_data, clean_data, configs, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}

    for config_name, config_params in configs.items():
        print(f"\n=== Training configuration: {config_name} ===")
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

            model = HybridTransformerDenoiser(**config_params).to(device)
            _, train_loss, val_loss = train_model(model, train_loader, val_loader, num_epochs=10)
            all_train.append(train_loss)
            all_val.append(val_loss)

            # Visualize 2 images from fold 0 only
            if fold == 0 and config_name == list(configs.keys())[0]:
                model.eval()
                samples = next(iter(val_loader))[0].to(device)
                preds = model(samples).detach().cpu().numpy()
                plot_comparison(noisy_data[val_idx], clean_data[val_idx], preds, 0, tag="val")
                plot_comparison(noisy_data[val_idx], clean_data[val_idx], preds, 1, tag="val")
                train_samples = next(iter(train_loader))[0].to(device)
                preds_train = model(train_samples).detach().cpu().numpy()
                plot_comparison(noisy_data[train_idx], clean_data[train_idx], preds_train, 0, tag="train")
                plot_comparison(noisy_data[train_idx], clean_data[train_idx], preds_train, 1, tag="train")

        results[config_name] = (all_train, all_val)

    # Plot comparison of loss curves
    for config_name, (tr, vl) in results.items():
        avg_tr = np.mean(tr, axis=0)
        avg_vl = np.mean(vl, axis=0)
        plt.plot(avg_tr, label=f"{config_name} - Train")
        plt.plot(avg_vl, label=f"{config_name} - Val")
    plt.title("Configuration Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("config_comparison.png")
    plt.close()

# === MAIN ===
if __name__ == "__main__":
    print("Using device =", device)

    noisy_data = np.load("noisy_train_19k.npy").astype(np.float32)
    clean_data = np.load("clean_train_19k.npy").astype(np.float32)
    noisy_data = normalize_data(noisy_data)
    clean_data = normalize_data(clean_data)

    configs = {
        "ConfigA": {"img_size": 64, "patch_size": 4, "emb_dim": 128},
        "ConfigB": {"img_size": 64, "patch_size": 4, "emb_dim": 256},
        "ConfigC": {"img_size": 64, "patch_size": 8, "emb_dim": 256}
    }

    start = time.time()
    cross_validate_and_compare(noisy_data, clean_data, configs, k=5)
    end = time.time()
    print("Total training time:", end - start, "seconds")
