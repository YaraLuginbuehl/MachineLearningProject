
from CNN_FUNCTION_CROSSVAL import CNN_FUNCTION

results = CNN_FUNCTION(
    noisy_data_path="/cluster/home/yluginbuehl/Denoising_2D/noisy_train_19k.npy",
    clean_data_path="/cluster/home/yluginbuehl/Denoising_2D/clean_train_19k.npy",        
    activation='relu',
    num_layers=2, 
    optimizer_type='adam',
    learning_rate=0.0005,
    n_splits=5  # Enable cross-validation
)

import matplotlib.pyplot as plt
import numpy as np

# Plot and print results for each fold
for fold_result in results:
    fold = fold_result["fold"]
    print(f"\n=== Fold {fold} ===")
    print(f"Runtime: {fold_result['runtime_sec']:.2f} seconds")
    print(f"Validation PSNR: {fold_result['validation']['psnr']:.2f}")
    print(f"Validation SSIM: {fold_result['validation']['ssim']:.4f}")
    print(f"Validation MSE: {fold_result['validation']['mse']:.6f}")

    plt.figure(figsize=(12, 4))
    titles = ["Noisy", "Clean", "Predicted"]
    images = [
        fold_result["noisy_image"],
        fold_result["clean_image"],
        fold_result["predicted_image"]
    ]
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.suptitle(f"Model 1 - Fold {fold}")
    plt.savefig(f"/cluster/home/yluginbuehl/Denoising_2D/CNN/Model1/Model1_CNN_Image_Fold{fold}")

    plt.figure(figsize=(8,5))
    plt.plot(fold_result['loss_history'], label='Train Loss')
    plt.plot(fold_result['val_loss_history'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training and Validation Loss - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/cluster/home/yluginbuehl/Denoising_2D/CNN/Model1/Model1_CNN_Loss_Fold{fold}")

# Calculate mean and std for metrics across folds
psnr_list = [fold_result['validation']['psnr'] for fold_result in results]
ssim_list = [fold_result['validation']['ssim'] for fold_result in results]
mse_list = [fold_result['validation']['mse'] for fold_result in results]
runtime_list = [fold_result['runtime_sec'] for fold_result in results]

print("\n=== Cross-Validation Summary ===")
print(f"PSNR:  mean = {np.mean(psnr_list):.2f}, std = {np.std(psnr_list):.2f}")
print(f"SSIM:  mean = {np.mean(ssim_list):.4f}, std = {np.std(ssim_list):.4f}")
print(f"MSE:   mean = {np.mean(mse_list):.6f}, std = {np.std(mse_list):.6f}")
print(f"Runtime (s): mean = {np.mean(runtime_list):.2f}, std = {np.std(runtime_list):.2f}")