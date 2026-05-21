import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T

class ForgeryDataset(Dataset):
    """
    Unified dataset class for CocoGlide, Columbia, and In-the-Wild datasets.
    """
    def __init__(self, root_dir, dataset_type='cocoglide', transform=None):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        if self.dataset_type == 'cocoglide':
            fake_dir = os.path.join(self.root_dir, 'fake')
            mask_dir = os.path.join(self.root_dir, 'mask')
            for f in os.listdir(fake_dir):
                if f.endswith(('.jpg', '.png')):
                    samples.append({
                        'image': os.path.join(fake_dir, f),
                        'mask': os.path.join(mask_dir, f.replace('.jpg', '.png')), # Handle extension mismatch if any
                        'label': 1
                    })
        elif self.dataset_type == 'columbia':
            splc_dir = os.path.join(self.root_dir, '4cam_splc')
            for f in os.listdir(splc_dir):
                if f.endswith('.tif'):
                    samples.append({
                        'image': os.path.join(splc_dir, f),
                        'mask': None, # Columbia masks are often edge masks or implicit
                        'label': 1
                    })
        elif self.dataset_type == 'in_wild':
            img_dir = os.path.join(self.root_dir, 'label_in_wild/images')
            mask_dir = os.path.join(self.root_dir, 'label_in_wild/masks')
            for f in os.listdir(img_dir):
                if f.endswith('.jpg'):
                    samples.append({
                        'image': os.path.join(img_dir, f),
                        'mask': os.path.join(mask_dir, f),
                        'label': 1
                    })
        elif self.dataset_type == 'casia2':
            tp_dir = os.path.join(self.root_dir, 'Tp')
            gt_dir = os.path.join(self.root_dir, 'CASIA 2 Groundtruth')
            for f in os.listdir(tp_dir):
                if f.endswith(('.tif', '.jpg')):
                    # Map Tp_... to Tp_..._gt.png
                    base_name = os.path.splitext(f)[0]
                    gt_name = base_name + '_gt.png'
                    samples.append({
                        'image': os.path.join(tp_dir, f),
                        'mask': os.path.join(gt_dir, gt_name),
                        'label': 1
                    })
            au_dir = os.path.join(self.root_dir, 'Au')
            for f in os.listdir(au_dir):
                if f.endswith('.jpg'):
                    samples.append({
                        'image': os.path.join(au_dir, f),
                        'mask': None,
                        'label': 0
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['image']).convert('RGB')
        img = self.transform(img)
        
        if s['mask'] and os.path.exists(s['mask']):
            mask = Image.open(s['mask']).convert('L')
            mask = T.Resize((256, 256))(mask)
            mask = T.ToTensor()(mask)
        else:
            mask = torch.zeros((1, 256, 256))
            
        return img, mask, s['label']

def get_dataloaders(base_path):
    loaders = {}
    ds_configs = [
        ('cocoglide', 'CocoGlide'),
        ('columbia', 'Columbia'),
        ('in_wild', 'in_wild'),
        ('casia2', 'CASIA2')
    ]
    for key, folder in ds_configs:
        p = os.path.join(base_path, folder)
        if os.path.exists(p):
            ds = ForgeryDataset(p, dataset_type=key)
            loaders[key] = DataLoader(ds, batch_size=8, shuffle=False)
    return loaders
