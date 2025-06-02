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


# === Hybrid U-Net Transformer Model ===
class UNetTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=8, emb_dim=256, num_layers=2, num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_dim = 128 * patch_size * patch_size

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.linear_proj = nn.Linear(self.patch_dim, emb_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(emb_dim, self.patch_dim)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        t_input = self.pool2(e2)
        B, C_t, H2, W2 = t_input.shape
        num_patches_h = H2 // self.patch_size
        num_patches_w = W2 // self.patch_size

        patches = t_input.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().reshape(B, -1, self.patch_dim)

        emb = self.linear_proj(patches)
        t_output = self.transformer(emb)
        decoded = self.output_proj(t_output)

        decoded = decoded.view(B, num_patches_h, num_patches_w, 128, self.patch_size, self.patch_size)
        decoded = decoded.permute(0, 3, 1, 4, 2, 5).contiguous()
        t_output = decoded.view(B, 128, H2, W2)

        x_up = self.up2(t_output)
        x_up = self.up1(x_up)
        d1 = self.dec1(torch.cat([x_up, e1], dim=1))

        return torch.sigmoid(self.final(d1))


def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())


def hybrid_loss(pred, target):
    l1 = nn.L1Loss()
    return 0.85 * l1(pred, target) + 0.15 * (1 - ssim(pred, target, data_range=1.0, size_average=True))


def plot_and_save_comparison(noisy, clean, pred, idx, fold, tag):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(noisy[idx, 0], cmap='gray'); axs[0].set_title("Noisy")
    axs[1].imshow(clean[idx, 0], cmap='gray'); axs[1].set_title("Clean")
    axs[2].imshow(pred[idx, 0], cmap='gray'); axs[2].set_title("Denoised")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"hybrid_predictions/hybrid_{tag}_sample_{idx}_fold{fold}_{timestamp}.png")
    plt.close()

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = hybrid_loss(model(x), y)
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
                val_loss += hybrid_loss(model(x), y).item()
        val_losses.append(val_loss / len(val_loader))

    return model, train_losses, val_losses

def cross_validate_model(noisy_data, clean_data, k=5):
    os.makedirs("hybrid_predictions", exist_ok=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_train, all_val = [], []
    fold_metrics = []
    loss_records = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(noisy_data)):
        print(f"\n=== Fold {fold + 1}/{k} ===")
        train_loader = DataLoader(TensorDataset(
            torch.from_numpy(noisy_data[train_idx]).unsqueeze(1).float(),
            torch.from_numpy(clean_data[train_idx]).unsqueeze(1).float()
        ), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(
            torch.from_numpy(noisy_data[val_idx]).unsqueeze(1).float(),
            torch.from_numpy(clean_data[val_idx]).unsqueeze(1).float()
        ), batch_size=64)

        model = UNetTransformer().to(device)
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

        model.eval()
        inputs, targets = next(iter(val_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).detach().cpu().numpy()
        inputs_np, targets_np = inputs.cpu().numpy(), targets.cpu().numpy()

        # Save visual and raw data for 4 samples
        for i in range(min(4, len(preds))):
            plot_and_save_comparison(inputs_np, targets_np, preds, i, fold, tag="val")

            # Save raw arrays for subtraction heatmaps
            np.save(f"hybrid_predictions/noisy_fold{fold}_idx{i}.npy", inputs_np[i, 0])
            np.save(f"hybrid_predictions/clean_fold{fold}_idx{i}.npy", targets_np[i, 0])
            np.save(f"hybrid_predictions/pred_hybrid_fold{fold}_idx{i}.npy", preds[i, 0])

        # Fold metrics
        mse_fold = np.mean([mse(targets_np[i, 0], preds[i, 0]) for i in range(len(preds))])
        psnr_fold = np.mean([psnr(targets_np[i, 0], preds[i, 0], data_range=1.0) for i in range(len(preds))])
        ssim_fold = np.mean([ssim_metric(targets_np[i, 0], preds[i, 0], data_range=1.0) for i in range(len(preds))])
        fold_metrics.append((mse_fold, psnr_fold, ssim_fold))
        print(f"MSE: {mse_fold:.6f} | PSNR: {psnr_fold:.2f} | SSIM: {ssim_fold:.4f}")

    # Final summary
    mse_vals, psnr_vals, ssim_vals = zip(*fold_metrics)
    print("\n=== Final Hybrid Metrics ===")
    print(f"Avg MSE:  {np.mean(mse_vals):.6f} +/- {np.std(mse_vals):.6f}")
    print(f"Avg PSNR: {np.mean(psnr_vals):.2f} +/- {np.std(psnr_vals):.2f}")
    print(f"Avg SSIM: {np.mean(ssim_vals):.4f} +/- {np.std(ssim_vals):.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(loss_records)
    df.to_csv(f"hybrid_predictions/hybrid_loss_folds_{timestamp}.csv", index=False)


# === Main ===
if __name__ == "__main__":
    print("==== Hybrid Model Evaluation Started ====")
    noisy_data = np.load("noisy_train_19k.npy").astype(np.float32)
    clean_data = np.load("clean_train_19k.npy").astype(np.float32)
    noisy_data = normalize_data(noisy_data)
    clean_data = normalize_data(clean_data)

    start = time.time()
    cross_validate_model(noisy_data, clean_data, k=5)
    print("Total time:", time.time() - start, "seconds")
