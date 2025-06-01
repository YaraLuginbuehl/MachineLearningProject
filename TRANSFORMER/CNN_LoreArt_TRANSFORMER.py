# -*- coding: utf-8 -*-
# === Imports ===
import numpy as np
import numpy.ma as ma
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim_metric, mean_squared_error as mse
from datetime import datetime
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Transformer Model Definition ===
class TransformerDenoiser(nn.Module):
    def __init__(self, img_size=128, patch_size=8, emb_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.linear_proj = nn.Linear(self.patch_dim, emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(emb_dim, self.patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5).reshape(B, -1, self.patch_dim)

        emb = self.linear_proj(patches) + self.pos_embedding[:, :patches.shape[1], :]
        encoded = self.transformer(emb)
        decoded = self.output_proj(encoded)

        decoded = decoded.view(B, H // self.patch_size, W // self.patch_size, 1, self.patch_size, self.patch_size)
        decoded = decoded.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = decoded.view(B, 1, H, W)

        return torch.sigmoid(out)

# === Helper Functions ===
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

def hybrid_loss(pred, target):
    l1 = nn.L1Loss()
    return 0.85 * l1(pred, target) + 0.15 * (1 - ssim(pred, target, data_range=1.0, size_average=True))

def save_prediction_data(noisy, clean, pred, fold, index):
    os.makedirs("transformer_predictions", exist_ok=True)
    np.save(f"transformer_predictions/noisy_fold{fold}_idx{index}.npy", noisy[index, 0])
    np.save(f"transformer_predictions/clean_fold{fold}_idx{index}.npy", clean[index, 0])
    np.save(f"transformer_predictions/pred_fold{fold}_idx{index}.npy", pred[index, 0])

def plot_comparison(noisy, clean, pred, index, tag, fold):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(noisy[index, 0], cmap='gray'); axs[0].set_title("Noisy")
    axs[1].imshow(clean[index, 0], cmap='gray'); axs[1].set_title("Clean")
    axs[2].imshow(pred[index, 0], cmap='gray'); axs[2].set_title("Denoised")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"transformer_{tag}_sample_{index:02d}_fold{fold}_{timestamp}.png"
    plt.savefig(fname)
    plt.close()

# === Training Routine ===
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

# === Cross-Validation Routine ===
def cross_validate_model(noisy_data, clean_data, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_train, all_val = [], []
    loss_records = []
    mse_list, psnr_list, ssim_list = [], [], []

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
        _, train_loss, val_loss = train_model(model, train_loader, val_loader)
        all_train.append(train_loss)
        all_val.append(val_loss)

        for epoch in range(len(train_loss)):
            loss_records.append({
                'Fold': fold + 1,
                'Epoch': epoch + 1,
                'Train Loss': train_loss[epoch],
                'Val Loss': val_loss[epoch]
            })

        if fold == 0:
            model.eval()
            val_inputs, val_targets = next(iter(val_loader))
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            preds_val = model(val_inputs).detach().cpu().numpy()
            val_inputs_cpu = val_inputs.cpu().numpy()
            val_targets_cpu = val_targets.cpu().numpy()
            for i in range(4):
                save_prediction_data(val_inputs_cpu, val_targets_cpu, preds_val, fold, i)
                plot_comparison(val_inputs_cpu, val_targets_cpu, preds_val, i, tag="val", fold=fold)

        model.eval()
        val_inputs, val_targets = next(iter(val_loader))
        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
        preds = model(val_inputs).detach().cpu().numpy()
        val_targets = val_targets.cpu().numpy()
        mse_score = np.mean([mse(val_targets[i, 0], preds[i, 0]) for i in range(min(4, len(preds)))])
        psnr_score = np.mean([psnr(val_targets[i, 0], preds[i, 0], data_range=1.0) for i in range(min(4, len(preds)))])
        ssim_score = np.mean([ssim_metric(val_targets[i, 0], preds[i, 0], data_range=1.0) for i in range(min(4, len(preds)))])
        mse_list.append(mse_score)
        psnr_list.append(psnr_score)
        ssim_list.append(ssim_score)

        print(f"Fold {fold} MSE: {mse_score}")
        print(f"Fold {fold} PSNR: {psnr_score}")
        print(f"Fold {fold} SSIM: {ssim_score}")

    print("\n=== Transformer Model Final Metrics ===")
    print(f"Avg MSE: {np.mean(mse_list):.6f} +/- {np.std(mse_list):.6f}")
    print(f"Avg PSNR: {np.mean(psnr_list):.2f} +/- {np.std(psnr_list):.2f}")
    print(f"Avg SSIM: {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}")

    max_epochs = max(len(l) for l in all_train)
    def pad(l): return l + [np.nan] * (max_epochs - len(l))
    train_matrix = np.array([pad(l) for l in all_train])
    val_matrix = np.array([pad(l) for l in all_val])
    mean_train = ma.masked_invalid(train_matrix).mean(axis=0)
    mean_val = ma.masked_invalid(val_matrix).mean(axis=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.plot(mean_train, label="Train Loss")
    plt.plot(mean_val, label="Validation Loss")
    plt.title("Transformer Loss (Mean across folds)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"transformer_loss_mean_{timestamp}.png")
    plt.close()

# === Main ===
if __name__ == "__main__":
    print("==== Script started ====")
    noisy_data = np.load("noisy_train_19k.npy").astype(np.float32)
    clean_data = np.load("clean_train_19k.npy").astype(np.float32)
    noisy_data = normalize_data(noisy_data)
    clean_data = normalize_data(clean_data)
    start = time.time()
    cross_validate_model(noisy_data, clean_data, k=5)
    print("Total time:", time.time() - start, "seconds")
