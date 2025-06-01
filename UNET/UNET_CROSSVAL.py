def UNET_FUNCTION(
    noisy_data_path,
    clean_data_path,
    num_layers=3,
    activation='relu',
    optimizer_type='adamw',
    learning_rate=0.0005,
    n_splits=5  # New argument for number of folds
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import time
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = 32
    bottleneck_size = 128

    def get_activation(act):
        return {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }.get(act.lower(), nn.ReLU())

    def get_optimizer(model, optimizer_type, learning_rate):
        opt = optimizer_type.lower()
        if opt == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif opt == 'adamw':
            return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif opt == 'sgd':
            return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    class UNetDynamic(nn.Module):
        def __init__(self, features, num_layers, bottleneck_size, activation):
            super().__init__()
            act = get_activation(activation)
            self.encoders = nn.ModuleList()
            self.pools = nn.ModuleList()
            in_channels = 1
            for i in range(num_layers):
                out_channels = features * (2 ** i)
                self.encoders.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        act,
                        nn.BatchNorm2d(out_channels),
                    )
                )
                self.pools.append(nn.MaxPool2d(2))
                in_channels = out_channels

            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels, bottleneck_size, 3, padding=1),
                act,
                nn.BatchNorm2d(bottleneck_size),
            )

            self.ups = nn.ModuleList()
            self.decoders = nn.ModuleList()
            for i in reversed(range(num_layers)):
                up_out_channels = features * (2 ** i)
                self.ups.append(nn.ConvTranspose2d(bottleneck_size, up_out_channels, 2, stride=2))
                self.decoders.append(
                    nn.Sequential(
                        nn.Conv2d(up_out_channels * 2, up_out_channels, 3, padding=1),
                        act,
                        nn.BatchNorm2d(up_out_channels),
                    )
                )
                bottleneck_size = up_out_channels

            self.final = nn.Sequential(
                nn.Conv2d(features, 16, 3, padding=1), act, nn.BatchNorm2d(16),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            skips = []
            for enc, pool in zip(self.encoders, self.pools):
                x = enc(x)
                skips.append(x)
                x = pool(x)

            x = self.bottleneck(x)

            for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
                x = up(x)
                x = torch.cat([x, skip], dim=1)
                x = dec(x)

            return self.final(x)

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
        optimizer = get_optimizer(model, optimizer_type, learning_rate)
        num_epochs = 50
        loss_history = []
        loss_history_val = []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}")
            model.train()
            running_loss = 0
            for noisy, clean in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            loss_history.append(epoch_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_loader)
            loss_history_val.append(val_loss)

        return model, loss_history, loss_history_val

    def validation(clean, predicted):
        psnr_list, ssim_list, mse_list = [], [], []
        for i in range(len(clean)):
            gt = clean[i, 0].astype(np.float32)
            pred = predicted[i, 0].astype(np.float32)
            psnr_list.append(psnr(gt, pred, data_range=1.0))
            ssim_list.append(ssim(gt, pred, data_range=1.0))
            mse_list.append(mean_squared_error(gt, pred))
        return np.mean(psnr_list), np.mean(ssim_list), np.mean(mse_list)

    noisy_data = np.load(noisy_data_path)
    clean_data = np.load(clean_data_path)
    noisy_profiles_norm = normalize_data(noisy_data)
    clean_profiles_norm = normalize_data(clean_data)

    noisy_tensor = torch.from_numpy(noisy_profiles_norm).unsqueeze(1).float()
    clean_tensor = torch.from_numpy(clean_profiles_norm).unsqueeze(1).float()
    dataset = TensorDataset(noisy_tensor, clean_tensor)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32)

        model = UNetDynamic(features, num_layers, bottleneck_size, activation).to(device)

        start_time = time.time()
        model, loss_history, val_loss_history = train_model(train_loader, val_loader, model)
        runtime = time.time() - start_time

        model.eval()
        predicted_profiles = []
        with torch.no_grad():
            for noisy, _ in val_loader:
                noisy = noisy.to(device)
                predicted = model(noisy)
                predicted_profiles.append(predicted.cpu())
        predicted_profiles = torch.cat(predicted_profiles, dim=0).cpu().numpy()

        val_noisy, val_clean = [], []
        for i in range(len(val_subset)):
            n, c = val_subset[i]
            val_noisy.append(n.numpy())
            val_clean.append(c.numpy())
        val_noisy = np.stack(val_noisy)
        val_clean = np.stack(val_clean)

        avg_psnr, avg_ssim, avg_mse = validation(val_clean, predicted_profiles)

        idx = min(30, len(val_subset)-1)
        noisy_sample = val_noisy[idx, 0]
        clean_sample = val_clean[idx, 0]
        pred_sample = predicted_profiles[idx, 0]

        results.append({
            "fold": fold+1,
            "predicted_image": pred_sample,
            "clean_image": clean_sample,
            "noisy_image": noisy_sample,
            "runtime_sec": runtime,
            "validation": {
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "mse": avg_mse,
            },
            "loss_history": loss_history,
            "val_loss_history": val_loss_history
        })

    return results
