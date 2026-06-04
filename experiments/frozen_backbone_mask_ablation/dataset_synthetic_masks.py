"""
Dataset class for frames with SAM2-generated synthetic segmentation masks.
Uses all 1,933 BBox201 frames with the official Endoscapes train/val/test split.
"""

import os
import sys
import json
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


class SyntheticMaskDataset(Dataset):
    """
    Dataset for frames with SAM2-generated synthetic segmentation masks.
    Uses the official Endoscapes BBox201 split (1212 train, 409 val, 312 test).

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
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.use_masks = use_masks
        self.image_size = image_size
        self.center_crop = center_crop
        self.threshold = threshold

        # Paths - use official split directories
        self.image_dir = os.path.join(dataset_dir, split)
        self.mask_dir = os.path.join(dataset_dir, 'synthetic_masks', split, 'semantic')
        self.metadata_path = os.path.join(dataset_dir, 'all_metadata.csv')

        # Load COCO annotations for this split to get frame list
        coco_path = os.path.join(self.image_dir, 'annotation_coco.json')
        with open(coco_path, 'r') as f:
            coco = json.load(f)

        # Build frame list from COCO images
        self.frames = []
        for img in coco['images']:
            # Parse filename: {vid}_{frame}.jpg
            fname = img['file_name'].replace('.jpg', '')
            parts = fname.split('_')
            vid = int(parts[0])
            frame = int(parts[1])
            self.frames.append({'vid': vid, 'frame': frame, 'filename': img['file_name']})

        # Load metadata for CVS labels
        df = pd.read_csv(self.metadata_path)
        df['vid_frame'] = df['vid'].astype(str) + '_' + df['frame'].astype(str)
        self.metadata = df.set_index('vid_frame')

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

        print(f"[SyntheticMaskDataset] Split: {split}, Samples: {len(self.frames)}, Use masks: {use_masks}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_info = self.frames[idx]
        vid = frame_info['vid']
        frame = frame_info['frame']
        vid_frame_key = f"{vid}_{frame}"

        # Load RGB image from split directory
        img_path = os.path.join(self.image_dir, frame_info['filename'])
        rgb_img = Image.open(img_path).convert('RGB')
        rgb_tensor = self.rgb_transform(rgb_img)

        if self.use_masks:
            # Load synthetic segmentation mask
            mask_name = f"{vid}_{frame}.png"
            mask_path = os.path.join(self.mask_dir, mask_name)
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

        # Get CVS labels from metadata (binarize with threshold)
        if vid_frame_key in self.metadata.index:
            row = self.metadata.loc[vid_frame_key]
            c1 = 1.0 if row['C1'] >= self.threshold else 0.0
            c2 = 1.0 if row['C2'] >= self.threshold else 0.0
            c3 = 1.0 if row['C3'] >= self.threshold else 0.0
        else:
            # Default to all zeros if not found
            c1, c2, c3 = 0.0, 0.0, 0.0

        label_tensor = torch.tensor([c1, c2, c3], dtype=torch.float32)

        return input_tensor, label_tensor


def get_synthetic_mask_dataloaders(dataset_dir, use_masks=False, batch_size=8, num_workers=4):
    """Create train/val/test dataloaders for synthetic mask frames."""

    train_ds = SyntheticMaskDataset(dataset_dir, split='train', use_masks=use_masks)
    val_ds = SyntheticMaskDataset(dataset_dir, split='val', use_masks=use_masks)
    test_ds = SyntheticMaskDataset(dataset_dir, split='test', use_masks=use_masks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    dataset_dir = r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes'

    print("Testing RGB-only dataset:")
    train_ds = SyntheticMaskDataset(dataset_dir, split='train', use_masks=False)
    val_ds = SyntheticMaskDataset(dataset_dir, split='val', use_masks=False)
    test_ds = SyntheticMaskDataset(dataset_dir, split='test', use_masks=False)

    x, y = train_ds[0]
    print(f"  RGB shape: {x.shape}, Label shape: {y.shape}")

    print("\nTesting RGB+Mask dataset:")
    train_ds_mask = SyntheticMaskDataset(dataset_dir, split='train', use_masks=True)
    x, y = train_ds_mask[0]
    print(f"  RGB+Mask shape: {x.shape}, Label shape: {y.shape}")
    print(f"  RGB channels range: [{x[:3].min():.2f}, {x[:3].max():.2f}]")
    print(f"  Mask channels range: [{x[3:].min():.2f}, {x[3:].max():.2f}]")

    # Check CVS label distribution
    print("\nCVS label distribution in train set:")
    c1_pos = sum(1 for i in range(len(train_ds)) if train_ds[i][1][0] == 1.0)
    c2_pos = sum(1 for i in range(len(train_ds)) if train_ds[i][1][1] == 1.0)
    c3_pos = sum(1 for i in range(len(train_ds)) if train_ds[i][1][2] == 1.0)
    print(f"  C1 positive: {c1_pos}/{len(train_ds)}")
    print(f"  C2 positive: {c2_pos}/{len(train_ds)}")
    print(f"  C3 positive: {c3_pos}/{len(train_ds)}")
