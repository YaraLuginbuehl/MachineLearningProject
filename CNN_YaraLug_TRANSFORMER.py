#TODO: all the neccessary imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

import os
import time
# os.chdir('C:\\Users\yslug\Desktop\SpaceSystems\SpaceData\HORUS\HW2')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the MLP model
class TransformerDenoiser(nn.Module):
    def __init__(self, img_size=64, patch_size=8, emb_dim=256, num_layers=4, num_heads=8):
        super(TransformerDenoiser, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size  # 1 channel, so no *channels

        self.linear_proj = nn.Linear(self.patch_dim, emb_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(emb_dim, self.patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Divide into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, self.patch_dim)  # (B, num_patches, patch_dim)

        x = self.linear_proj(x) + self.positional_embedding  # Add position
        x = self.transformer(x)  # (B, num_patches, emb_dim)

        x = self.output_proj(x)  # (B, num_patches, patch_dim)
        x = x.view(B, int(self.img_size / self.patch_size), int(self.img_size / self.patch_size),
                   self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, 1, H, W)
        return x

#TODO: define and fill the normalize_data function
def normalize_data(data_whole):
    normalized_data = np.zeros(data_whole.shape)
    for i in range(data_whole.shape[0]):
        data = data_whole[i] 
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min
        normalized_data[i] = (data - data_min) / data_range

    return normalized_data #(data - np.mean(data)) / np.std(data)

#TODO define and fill the MSE function
def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

#TODO fill the main training function
def train_model(train_loader, val_loader, model):
    # --------------------
    # 1) Set your paramters
    # --------------------

    # --------------------
    # 2) Create DataLoaders
    #    (not strictly necessary but it's the de facto standard)
    # --------------------
  
    # --------------------
    # 3) Define model, loss, optimizer
    # --------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # --------------------
    # 4) Training loop over epochs
    # --------------------

    loss_history = []
    loss_history_val = []
    for epoch in range(num_epochs):
        model.train()
        print("epoch:", epoch)
        running_loss = 0
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            y_pred = model(noisy)
            loss = criterion(y_pred, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # VALIDATION ERROR
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()

        # Average validation loss for the epoch
        test_loss /= len(val_loader)
        loss_history_val.append(test_loss)



    # Return the trained model and loss curves)
    return model, loss_history, loss_history_val



def visualize_patches(image_tensor, patch_size):
    img = image_tensor.squeeze().numpy()
    fig, ax = plt.subplots()
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            ax.add_patch(plt.Rectangle((j, i), patch_size, patch_size,
                                       fill=False, edgecolor='red', lw=1))
    ax.imshow(img, cmap='gray')
    plt.title("Image patches")
    plt.savefig("Patches_Visualisation")


#TODO: fill the function to plot a clean & noisy sample as well as the corresponding NN prediction
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
    return


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'CPU')
    print("Using device = ", device)

    # (A) Load data (example: shape (N, H, W))
    noisy_data = np.load("noisy_train_19k.npy")
    clean_data = np.load("clean_train_19k.npy")

    # Convert to row-wise brightness profiles, shape (N, H)
    # noisy_profiles = noisy_data.mean(axis=2)
    # clean_profiles = clean_data.mean(axis=2)

    #TODO: call the normalize function
    noisy_profiles_norm = normalize_data(noisy_data)
    clean_profiles_norm = normalize_data(clean_data)

    #TODO convert to torch tensors and create the TensorDataset
    noisy_profiles_tensor = torch.from_numpy(noisy_profiles_norm).unsqueeze(1).float()
    clean_profiles_tensor = torch.from_numpy(clean_profiles_norm).unsqueeze(1).float()
    # (batch_size, channels, height, width) = (1000, 1 (grayscale), 64 (H), 64 (W))

    #TODO split into train and validation set (eg. with torches random_split function)
    dataset = TensorDataset(noisy_profiles_tensor, clean_profiles_tensor)
    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle = True) 
    val_loader = DataLoader(val_dataset, batch_size=64) 


    #TODO: call your train_model function
    model = TransformerDenoiser(img_size=64, patch_size=8, emb_dim=256).to(device)

    start = time.time()
    model, loss_history, loss_history_val = train_model(train_loader, val_loader, model)
    end = time.time()

    print("Took =",end-start)

    plt.figure()
    plt.plot(loss_history, label = "loss")
    plt.plot(loss_history_val[1:], label = "Val loss")
    plt.legend()
    plt.savefig("Loss_plot")


    #TODO set the model to eval mode
    model.eval()

    #TODO: call your noisy_clean_predicted function with a sample from the training and a sample from the testing dataset
    predicted_profiles = []
    with torch.no_grad():
        for noisy, _ in val_loader:
            noisy = noisy.to(device)

            predicted = model(noisy)
            predicted_profiles.append(predicted)

    predicted_profiles = torch.cat(predicted_profiles, dim=0).cpu().numpy()
    plot_noisy_clean_predicted(val_dataset[:][0], val_dataset[:][1], predicted_profiles, index=30)

    visualize_patches(noisy_profiles_tensor[0], patch_size=8)


