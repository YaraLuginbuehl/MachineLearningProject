import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import optuna
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class CNN(nn.Module):
    def __init__(self, num_layers=3, filters=None, kernels=None):
        super(CNN, self).__init__()
        # Defaults
        if filters is None:
            filters = [64, 128, 1]
        if kernels is None:
            kernels = [3] * num_layers

        layers = []
        in_channels = 1
        for i in range(num_layers):
            out_channels = filters[i]
            kernel_size = kernels[i]
            padding = kernel_size // 2  # to keep spatial size

            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            # For all but last layer add BatchNorm + ReLU, last layer only Sigmoid
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

            in_channels = out_channels

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def normalize_data(data_whole):
    normalized_data = np.zeros(data_whole.shape)
    for i in range(data_whole.shape[0]):
        data = data_whole[i] 
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min
        normalized_data[i] = (data - data_min) / data_range
    return normalized_data

def train_model_for_optuna(train_loader, val_loader, model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            y_pred = model(noisy)
            loss = criterion(y_pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation loss after training
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()

    test_loss /= len(val_loader)
    return test_loss

def objective(trial):
    print("Starting trial:", trial.number)
    # Sample architectural hyperparameters
    num_layers = trial.suggest_int('num_layers', 2, 5)

    # For each layer sample number of filters (except last fixed to 1 output channel)
    filters = []
    for i in range(num_layers - 1):
        filters.append(trial.suggest_categorical(f'filters_l{i}', [32, 64, 128, 256]))
    filters.append(1)  # last layer always output 1 channel

    kernels = []
    for i in range(num_layers):
        kernels.append(trial.suggest_categorical(f'kernel_l{i}', [3, 5, 7]))

    model = CNN(num_layers=num_layers, filters=filters, kernels=kernels).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    val_loss = train_model_for_optuna(train_loader, val_loader, model, criterion, optimizer, num_epochs=10)
    return val_loss

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

    plt.savefig("Denoising_2D")
    plt.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device =", device)

    noisy_data = np.load("noisy_images_small_1k.npy").astype(np.float32)
    clean_data = np.load("clean_images_small_1k.npy").astype(np.float32)

    noisy_profiles_norm = normalize_data(noisy_data)
    clean_profiles_norm = normalize_data(clean_data)

    noisy_profiles_tensor = torch.from_numpy(noisy_profiles_norm).unsqueeze(1).float()
    clean_profiles_tensor = torch.from_numpy(clean_profiles_norm).unsqueeze(1).float()

    dataset = TensorDataset(noisy_profiles_tensor, clean_profiles_tensor)
    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Run Optuna optimization for architecture
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

    print("Best architecture hyperparameters:")
    print(study.best_params)

    # Train final model with best architecture
    best_params = study.best_params
    num_layers = best_params['num_layers']
    filters = [best_params[f'filters_l{i}'] for i in range(num_layers - 1)] + [1]
    kernels = [best_params[f'kernel_l{i}'] for i in range(num_layers)]

    final_model = CNN(num_layers=num_layers, filters=filters, kernels=kernels).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("Training final model with best architecture...")
    start = time.time()
    train_model_for_optuna(train_loader, val_loader, final_model, criterion, optimizer, num_epochs=20)
    end = time.time()
    print(f"Final training took {end - start:.2f} seconds")

    final_model.eval()
    predicted_profiles = []
    with torch.no_grad():
        for noisy, _ in val_loader:
            noisy = noisy.to(device)
            predicted = final_model(noisy)
            predicted_profiles.append(predicted)

    predicted_profiles = torch.cat(predicted_profiles, dim=0).cpu().numpy()

    plot_noisy_clean_predicted(noisy_profiles_norm, clean_profiles_norm, predicted_profiles, index=30)
