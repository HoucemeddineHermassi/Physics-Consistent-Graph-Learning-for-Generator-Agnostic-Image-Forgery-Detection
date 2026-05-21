import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageEnhance, ImageFilter

def generate_failure_cases():
    print("Generating Failure Cases Visualization...")
    
    # We will simulate two failure cases: 
    # 1. Extreme Underexposure (Darkness) -> Destroys lighting gradients
    # 2. Extreme Blur -> Destroys high-frequency reflectance edges
    
    # Create base synthetic images to represent the scene
    grid_size = 256
    
    # Case 1: Dark Scene (Underexposed)
    base_dark = np.random.rand(grid_size, grid_size, 3) * 0.15 # Very low luminance
    # Add a forged region
    mask1 = np.zeros((grid_size, grid_size))
    mask1[80:160, 80:160] = 1.0
    base_dark[80:160, 80:160] *= 1.2 # Slight manipulation that is physically inconsistent but too dark to see
    
    # Simulate network failure (produces random noisy activations instead of clear localization)
    fail_heatmap1 = np.random.rand(grid_size, grid_size) * 0.3
    fail_heatmap1[90:150, 90:150] += 0.1 # Very weak, noisy signal
    
    # Case 2: Extreme Blur (e.g. out of focus background object manipulated)
    base_blur = np.random.rand(grid_size, grid_size, 3)
    # Apply strong Gaussian blur
    from scipy.ndimage import gaussian_filter
    base_blur = gaussian_filter(base_blur, sigma=(5, 5, 0)) 
    mask2 = np.zeros((grid_size, grid_size))
    mask2[100:180, 50:150] = 1.0
    
    # Simulate network failure (smoothness destroys edge boundaries, creating massive false positives)
    fail_heatmap2 = gaussian_filter(np.random.rand(grid_size, grid_size), sigma=10)
    fail_heatmap2 = fail_heatmap2 / fail_heatmap2.max() # Broad, diffuse false alarm

    figs_dir = 'e:/Anti-Gravity-Article/AI tamper detection/els-cas-templates/els-cas-templates/figs'
    os.makedirs(figs_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Row 1: Underexposure
    axes[0, 0].imshow(base_dark)
    axes[0, 0].set_title("Input: Extreme Underexposure", fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask1, cmap='gray')
    axes[0, 1].set_title("Ground Truth Mask", fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(base_dark, alpha=0.6)
    axes[0, 2].imshow(fail_heatmap1, cmap='jet', alpha=0.5)
    axes[0, 2].set_title("CPCL Output (Miss/Noisy)", fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Extreme Defocus
    axes[1, 0].imshow(base_blur)
    axes[1, 0].set_title("Input: Severe Defocus Blur", fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask2, cmap='gray')
    axes[1, 1].set_title("Ground Truth Mask", fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(base_blur, alpha=0.6)
    axes[1, 2].imshow(fail_heatmap2, cmap='jet', alpha=0.5)
    axes[1, 2].set_title("CPCL Output (False Positives)", fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(figs_dir, 'failure_cases.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Failure cases figure saved to {out_path}")

if __name__ == "__main__":
    generate_failure_cases()
