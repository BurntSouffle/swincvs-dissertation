"""
Dataset class for frames with GT segmentation masks.
Only loads the 493 frames that have ground truth semantic segmentation.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Add parent paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWINCVS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SWINCVS_ROOT)
sys.path.insert(0, os.path.join(SWINCVS_ROOT, 'scripts'))


class GTMaskDataset(Dataset):
    """
    Dataset for frames with ground truth segmentation masks.

    Returns:
        If use_masks=False: (rgb_tensor, label_tensor)
        If use_masks=True: (rgb_mask_tensor, label_tensor)

    Where:
        rgb_tensor: [3, H, W] or [5, H, W] if masks included
        label_tensor: [3] binary CVS labels
    """

    def __init__(
        self,
        dataset_dir,
        split='train',
        use_masks=False,
        image_size=384,
        center_crop=480,
        threshold=0.5,  # Threshold for CVS labels
        seed=42
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.use_masks = use_masks
        self.image_size = image_size
        self.center_crop = center_crop
        self.threshold = threshold

        # Paths
        self.image_dir = os.path.join(dataset_dir, 'all')
        self.semseg_dir = os.path.join(dataset_dir, 'semseg')
        self.metadata_path = os.path.join(dataset_dir, 'all_metadata.csv')

        # Load metadata
        df = pd.read_csv(self.metadata_path)

        # Get frames with semseg masks
        semseg_files = os.listdir(self.semseg_dir)
        semseg_names = set([f.replace('.png', '') for f in semseg_files])
        df['vid_frame'] = df['vid'].astype(str) + '_' + df['frame'].astype(str)
        df_semseg = df[df['vid_frame'].isin(semseg_names)].copy()

        # Split by video (deterministic based on seed)
        np.random.seed(seed)
        unique_vids = sorted(df_semseg['vid'].unique())
        np.random.shuffle(unique_vids)

        n_vids = len(unique_vids)
        n_train = int(0.7 * n_vids)
        n_val = int(0.15 * n_vids)

        train_vids = set(unique_vids[:n_train])
        val_vids = set(unique_vids[n_train:n_train + n_val])
        test_vids = set(unique_vids[n_train + n_val:])

        # Filter by split
        if split == 'train':
            self.df = df_semseg[df_semseg['vid'].isin(train_vids)].reset_index(drop=True)
        elif split == 'val':
            self.df = df_semseg[df_semseg['vid'].isin(val_vids)].reset_index(drop=True)
        elif split == 'test':
            self.df = df_semseg[df_semseg['vid'].isin(test_vids)].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown split: {split}")

        # Transforms for RGB
        self.rgb_transform = transforms.Compose([
            transforms.CenterCrop(center_crop),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Transforms for masks (no normalization, just resize)
        self.mask_transform = transforms.Compose([
            transforms.CenterCrop(center_crop),
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        print(f"[GTMaskDataset] Split: {split}, Samples: {len(self.df)}, Use masks: {use_masks}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = int(row['vid'])
        frame = int(row['frame'])

        # Load RGB image
        # Note: all/ folder contains text files with paths, need to resolve
        img_name = f"{vid}_{frame}.jpg"
        img_path = os.path.join(self.image_dir, img_name)

        # Check if file is a reference (small size) and resolve
        if os.path.getsize(img_path) < 1000:
            with open(img_path, 'r') as f:
                rel_path = f.read().strip()
            img_path = os.path.join(self.image_dir, rel_path)

        rgb_img = Image.open(img_path).convert('RGB')
        rgb_tensor = self.rgb_transform(rgb_img)

        if self.use_masks:
            # Load segmentation mask
            mask_name = f"{vid}_{frame}.png"
            mask_path = os.path.join(self.semseg_dir, mask_name)
            mask_img = Image.open(mask_path)

            # Apply spatial transform (center crop + resize)
            mask_img = self.mask_transform(mask_img)
            mask_array = np.array(mask_img)

            # Extract gallbladder (class 5) and tool (class 6) masks
            gb_mask = (mask_array == 5).astype(np.float32)
            tool_mask = (mask_array == 6).astype(np.float32)

            # Normalize masks to [-1, 1] to match RGB scale
            gb_mask = (gb_mask - 0.5) / 0.5
            tool_mask = (tool_mask - 0.5) / 0.5

            # Stack: [5, H, W] = [RGB(3) + GB(1) + Tool(1)]
            gb_tensor = torch.from_numpy(gb_mask).unsqueeze(0)
            tool_tensor = torch.from_numpy(tool_mask).unsqueeze(0)
            input_tensor = torch.cat([rgb_tensor, gb_tensor, tool_tensor], dim=0)
        else:
            input_tensor = rgb_tensor

        # Get CVS labels (binarize with threshold)
        c1 = 1.0 if row['C1'] >= self.threshold else 0.0
        c2 = 1.0 if row['C2'] >= self.threshold else 0.0
        c3 = 1.0 if row['C3'] >= self.threshold else 0.0
        label_tensor = torch.tensor([c1, c2, c3], dtype=torch.float32)

        return input_tensor, label_tensor


def get_gt_mask_dataloaders(dataset_dir, use_masks=False, batch_size=8, num_workers=4, seed=42):
    """Create train/val/test dataloaders for GT mask frames."""

    train_ds = GTMaskDataset(dataset_dir, split='train', use_masks=use_masks, seed=seed)
    val_ds = GTMaskDataset(dataset_dir, split='val', use_masks=use_masks, seed=seed)
    test_ds = GTMaskDataset(dataset_dir, split='test', use_masks=use_masks, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    dataset_dir = r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes'

    print("Testing RGB-only dataset:")
    train_ds = GTMaskDataset(dataset_dir, split='train', use_masks=False)
    val_ds = GTMaskDataset(dataset_dir, split='val', use_masks=False)
    test_ds = GTMaskDataset(dataset_dir, split='test', use_masks=False)

    x, y = train_ds[0]
    print(f"  RGB shape: {x.shape}, Label shape: {y.shape}")

    print("\nTesting RGB+Mask dataset:")
    train_ds_mask = GTMaskDataset(dataset_dir, split='train', use_masks=True)
    x, y = train_ds_mask[0]
    print(f"  RGB+Mask shape: {x.shape}, Label shape: {y.shape}")
    print(f"  RGB channels range: [{x[:3].min():.2f}, {x[:3].max():.2f}]")
    print(f"  Mask channels range: [{x[3:].min():.2f}, {x[3:].max():.2f}]")
