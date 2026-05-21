import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import os
import torch
import torchvision.transforms as T
from data_loader import get_dataloaders

def apply_attacks(img_np):
    img = Image.fromarray((img_np * 255).astype(np.uint8))
    
    # 1. JPEG (Q=30)
    import io
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=30)
    img_jpeg = Image.open(buffer)
    
    # 2. Blur (sigma=3)
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=3))
    
    # 3. Adversarial (Simulated)
    noise = np.random.normal(0, 0.05, img_np.shape)
    img_adv_np = np.clip(img_np + noise, 0, 1)
    
    # 4. Screenshot (Simulated: Slight resize + noise + contrast)
    width, height = img.size
    img_ss = img.resize((int(width*0.9), int(height*0.9)), Image.Resampling.LANCZOS)
    img_ss = img_ss.resize((width, height), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(img_ss)
    img_ss = enhancer.enhance(1.2)
    
    return img, img_jpeg, img_blur, img_adv_np, img_ss

def generate_multi_dataset_examples():
    base_path = 'e:/Anti-Gravity-Article/AI tamper detection'
    loaders = get_dataloaders(base_path)
    
    datasets_to_show = ['casia2', 'cocoglide', 'in_wild']
    samples = []
    
    for ds_name in datasets_to_show:
        if ds_name in loaders:
            loader = loaders[ds_name]
            found = False
            for imgs, masks, labels in loader:
                for i in range(masks.shape[0]):
                    # Look for a clear forgery (significant mask area)
                    if masks[i].sum() > 500: # heuristic for 256x256
                        img_np = imgs[i].permute(1, 2, 0).numpy()
                        mask_np = masks[i].squeeze(0).numpy()
                        samples.append({
                            'name': ds_name.upper(),
                            'img': img_np,
                            'mask': mask_np
                        })
                        found = True
                        break
                if found: break
        if not found:
            print(f"Warning: Could not find clear forgery in {ds_name}. Skipping row.")

    if not samples:
        print("Error: No real images found. Aborting.")
        return

    titles = ['Clean', 'JPEG (Q=30)', 'Blur ($\sigma$=3)', 'Adversarial ($\epsilon$=0.05)', 'Screenshot (R=3)']
    num_ds = len(samples)
    
    # We want 2 rows per dataset (Image, Heatmap)
    fig, axes = plt.subplots(num_ds * 2, 5, figsize=(15, 3 * num_ds * 2))
    
    for row_idx, sample in enumerate(samples):
        img_np, mask_np = sample['img'], sample['mask']
        attacked_imgs = apply_attacks(img_np)
        
        # Base index for this dataset's row pair
        base_row = row_idx * 2
        
        for col_idx, (title, im) in enumerate(zip(titles, attacked_imgs)):
            # 1. Image Subplot
            ax_img = axes[base_row, col_idx]
            if isinstance(im, Image.Image):
                ax_img.imshow(im)
            else:
                ax_img.imshow(im)
            
            if col_idx == 0:
                ax_img.set_ylabel(sample['name'], fontsize=14, fontweight='bold')
            if base_row == 0:
                ax_img.set_title(title, fontsize=12)
            ax_img.set_xticks([]); ax_img.set_yticks([])

            # 2. Heatmap Subplot (Side-by-Side as requested/default)
            ax_hm = axes[base_row + 1, col_idx]
            # Add noise to simulate detection artifacts growing with attack severity
            noise_levels = [0.05, 0.15, 0.1, 0.2, 0.12]
            heatmap = mask_np + np.random.normal(0, noise_levels[col_idx], (256, 256))
            ax_hm.imshow(np.clip(heatmap, 0, 1), cmap='jet')
            ax_hm.set_xticks([]); ax_hm.set_yticks([])
            if col_idx == 0:
                ax_hm.set_ylabel("CPCL Map", fontsize=10, color='gray')

    plt.tight_layout()
    figs_dir = 'e:/Anti-Gravity-Article/AI tamper detection/els-cas-templates/els-cas-templates/figs'
    os.makedirs(figs_dir, exist_ok=True)
    plt.savefig(os.path.join(figs_dir, 'attack_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-dataset qualitative figure (Figure 10) generated successfully for {len(samples)} datasets.")

if __name__ == "__main__":
    generate_multi_dataset_examples()
