import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def calculate_ssim_psnr(fake_img_path, real_img_path):
    """Calculate SSIM and PSNR between two images."""
    fake_img = np.array(Image.open(fake_img_path).convert("L"))  # Convert to grayscale
    real_img = np.array(Image.open(real_img_path).convert("L"))  # Convert to grayscale
    
    # SSIM calculation
    ssim = compare_ssim(fake_img, real_img)

    # PSNR calculation
    psnr = compare_psnr(real_img, fake_img)

    return ssim, psnr

def plot_comparison(real_A_path, fake_B_path, real_B_path):
    """
    Visualize and compare Real T1, Fake FLAIR, and Real FLAIR.
    Automatically calculate SSIM and PSNR.
    """
    # Calculate SSIM and PSNR
    ssim, psnr = calculate_ssim_psnr(fake_B_path, real_B_path)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Real T1", "Fake FLAIR", "Real FLAIR"]

    for i, (img_path, title) in enumerate(zip([real_A_path, fake_B_path, real_B_path], titles)):
        img = Image.open(img_path)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(title)

    plt.suptitle(f"SSIM: {ssim:.4f}, PSNR: {psnr:.2f}")
    plt.show()

# Example usage
real_A_path = "C:/Users/pmk/Desktop/cyclegan_medicianai/pytorch-CycleGAN-and-pix2pix_mk/datasets/mrDataset/testA/Singapore_52_T1_20.png"  # Path to Real T1 image
fake_B_path = "C:/Users/pmk/Desktop/cyclegan_medicianai/pytorch-CycleGAN-and-pix2pix_mk/datasets/results/mri_cyclegan_tune/test_latest/images/Singapore_52_T1_20_fake_B.png"  # Path to Fake FLAIR image
real_B_path = "C:/Users/pmk/Desktop/cyclegan_medicianai/pytorch-CycleGAN-and-pix2pix_mk/datasets/give_val_set_REAL_FLAIR/Singapore_52_FLAIR_20.png"  # Path to Real FLAIR image

plot_comparison(real_A_path, fake_B_path, real_B_path)


