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
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = self.sigmoid(x)
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
    noisy_data = np.load("noisy_images_small_1k.npy").astype(np.float32)
    clean_data = np.load("clean_images_small_1k.npy").astype(np.float32)

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
    model = CNN().to(device)

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



