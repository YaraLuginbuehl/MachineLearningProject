# All necessary imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Improved U-Net-like CNN model
class UNetSmall(nn.Module):
    def __init__(self):
        super(UNetSmall, self).__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64)
        )

        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.act(self.out(d1))


def normalize_data(data_whole):
    normalized_data = np.zeros(data_whole.shape)
    for i in range(data_whole.shape[0]):
        data = data_whole[i]
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min
        normalized_data[i] = (data - data_min) / data_range
    return normalized_data


def train_model(train_loader, val_loader, model):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    num_epochs = 10

    loss_history = []
    loss_history_val = []
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        print("Epoch:", epoch)
        running_loss = 0
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            y_pred = model(noisy)
            loss = criterion(y_pred, clean)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()

        test_loss /= len(val_loader)
        loss_history_val.append(test_loss)

        scheduler.step(test_loss)

    return model, loss_history, loss_history_val


def plot_noisy_clean_predicted(noisy_profiles_norm, clean_profiles_norm, predicted_data, index):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(noisy_profiles_norm[index, 0], cmap='gray')
    axs[0].set_title(f"Noisy (Sample {index})")
    axs[0].axis('off')

    axs[1].imshow(clean_profiles_norm[index, 0], cmap='gray')
    axs[1].set_title(f"Clean (Sample {index})")
    axs[1].axis('off')

    axs[2].imshow(predicted_data[index, 0], cmap='gray')
    axs[2].set_title(f"Predicted (Sample {index})")
    axs[2].axis('off')

    plt.savefig(f"denoised_sample_{index:04d}.png")
    plt.close()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device =", device)

    noisy_data = np.load("noisy_train_19k.npy")
    clean_data = np.load("clean_train_19k.npy")

    noisy_profiles_norm = normalize_data(noisy_data)
    clean_profiles_norm = normalize_data(clean_data)

    noisy_profiles_tensor = torch.from_numpy(noisy_profiles_norm).unsqueeze(1).float()
    clean_profiles_tensor = torch.from_numpy(clean_profiles_norm).unsqueeze(1).float()

    dataset = TensorDataset(noisy_profiles_tensor, clean_profiles_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = UNetSmall().to(device)

    start = time.time()
    model, loss_history, loss_history_val = train_model(train_loader, val_loader, model)
    end = time.time()
    print("Training took =", end - start)

    plt.figure()
    plt.plot(loss_history, label="Train Loss")
    plt.plot(loss_history_val, label="Validation Loss")
    plt.legend()
    plt.savefig("Loss_plot.png")

    model.eval()
    predicted_profiles = []
    with torch.no_grad():
        for noisy, _ in val_loader:
            noisy = noisy.to(device)
            predicted = model(noisy)
            predicted_profiles.append(predicted)

    predicted_profiles = torch.cat(predicted_profiles, dim=0).cpu().numpy()
    plot_noisy_clean_predicted(val_dataset[:][0], val_dataset[:][1], predicted_profiles, index=30)
