# MODEL 1

from Denoising_2D.UNET.CNN_YaraLug_UNET_FUNCTION import UNET_FUNCTION


results = UNET_FUNCTION(
    noisy_data_path="/cluster/home/yluginbuehl/Denoising_2D/UNET/noisy_train_19k.npy",
    clean_data_path="/cluster/home/yluginbuehl/Denoising_2D/UNET/clean_train_19k.npy",
    features=32,           # Base number of filters
    num_layers=3,          # U-Net depth
    activation='relu',     # Activation function
    bottleneck_size=128,   # Bottleneck feature size
)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
titles = ["Noisy", "Clean", "Predicted"]
images = [
    results["noisy_image"],
    results["clean_image"],
    results["predicted_image"]
]

for i, (title, img) in enumerate(zip(titles, images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.title("Model 1")
plt.savefig("/cluster/home/yluginbuehl/Denoising_2D/UNET/Model1/Model 1 Image")


plt.figure(figsize=(8,5))
plt.plot(results['loss_history'], label='Train Loss')
plt.plot(results['val_loss_history'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("/cluster/home/yluginbuehl/Denoising_2D/UNET/Model1/Model 1 Loss")

print(f"Runtime: {results['runtime_sec']:.2f} seconds")
print(f"Validation PSNR: {results['validation']['psnr']:.2f}")
print(f"Validation SSIM: {results['validation']['ssim']:.4f}")
print(f"Validation MSE: {results['validation']['mse']:.6f}")