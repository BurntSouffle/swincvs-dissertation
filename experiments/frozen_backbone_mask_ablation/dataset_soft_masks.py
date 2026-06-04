"""
Dataset class for frames with SAM2 soft masks (confidence scores).
Uses confidence maps instead of binary masks to preserve uncertainty information.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import cv2

# Add parent paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWINCVS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SWINCVS_ROOT)
sys.path.insert(0, os.path.join(SWINCVS_ROOT, 'scripts'))


class SoftMaskDataset(Dataset):
    """
    Dataset using SAM2 confidence scores as soft masks.

    Unlike binary masks (0/1), soft masks preserve uncertainty:
    - High confidence regions: values close to 1.0
    - Uncertain edges: values around 0.5
    - Background: values close to 0.0

    Returns:
        If use_masks=False: (rgb_tensor, label_tensor)
        If use_masks=True: (rgb_soft_mask_tensor, label_tensor)
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

        # Paths
        self.image_dir = os.path.join(dataset_dir, split)
        self.confidence_dir = os.path.join(dataset_dir, 'synthetic_masks', split, 'confidence')
        self.metadata_path = os.path.join(dataset_dir, 'all_metadata.csv')

        # Load COCO annotations for this split to get frame list
        coco_path = os.path.join(self.image_dir, 'annotation_coco.json')
        with open(coco_path, 'r') as f:
            coco = json.load(f)

        # Build frame list from COCO images
        self.frames = []
        for img in coco['images']:
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

        print(f"[SoftMaskDataset] Split: {split}, Samples: {len(self.frames)}, Use masks: {use_masks}")

    def __len__(self):
        return len(self.frames)

    def _load_soft_mask(self, vid, frame, class_name):
        """
        Load and combine all confidence maps for a given class.
        Multiple instances are combined using max (union).
        """
        # Find all matching confidence files
        pattern = os.path.join(self.confidence_dir, f"{vid}_{frame}_{class_name}_*.npy")
        files = glob.glob(pattern)

        if not files:
            # No mask for this class - return zeros
            return np.zeros((480, 854), dtype=np.float32)

        # Load and combine using max (for overlapping instances)
        combined = None
        for f in files:
            conf = np.load(f).astype(np.float32)
            if combined is None:
                combined = conf
            else:
                combined = np.maximum(combined, conf)

        return combined

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
            # Load soft masks (confidence maps)
            gb_conf = self._load_soft_mask(vid, frame, 'gallbladder')
            tool_conf = self._load_soft_mask(vid, frame, 'tool')

            # Apply center crop and resize to match RGB
            # First center crop
            h, w = gb_conf.shape
            crop_size = self.center_crop
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2

            gb_cropped = gb_conf[start_h:start_h+crop_size, start_w:start_w+crop_size]
            tool_cropped = tool_conf[start_h:start_h+crop_size, start_w:start_w+crop_size]

            # Resize to target size (bilinear for soft masks)
            gb_resized = cv2.resize(gb_cropped, (self.image_size, self.image_size),
                                    interpolation=cv2.INTER_LINEAR)
            tool_resized = cv2.resize(tool_cropped, (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_LINEAR)

            # Normalize to [-1, 1] range (same as binary mask normalization)
            gb_norm = (gb_resized - 0.5) / 0.5
            tool_norm = (tool_resized - 0.5) / 0.5

            # Convert to tensors
            gb_tensor = torch.from_numpy(gb_norm).unsqueeze(0)
            tool_tensor = torch.from_numpy(tool_norm).unsqueeze(0)

            # Stack: [5, H, W] = [RGB(3) + GB(1) + Tool(1)]
            input_tensor = torch.cat([rgb_tensor, gb_tensor, tool_tensor], dim=0)
        else:
            input_tensor = rgb_tensor

        # Get CVS labels from metadata
        if vid_frame_key in self.metadata.index:
            row = self.metadata.loc[vid_frame_key]
            c1 = 1.0 if row['C1'] >= self.threshold else 0.0
            c2 = 1.0 if row['C2'] >= self.threshold else 0.0
            c3 = 1.0 if row['C3'] >= self.threshold else 0.0
        else:
            c1, c2, c3 = 0.0, 0.0, 0.0

        label_tensor = torch.tensor([c1, c2, c3], dtype=torch.float32)

        return input_tensor, label_tensor


def get_soft_mask_dataloaders(dataset_dir, use_masks=False, batch_size=8, num_workers=4):
    """Create train/val/test dataloaders for soft mask frames."""

    train_ds = SoftMaskDataset(dataset_dir, split='train', use_masks=use_masks)
    val_ds = SoftMaskDataset(dataset_dir, split='val', use_masks=use_masks)
    test_ds = SoftMaskDataset(dataset_dir, split='test', use_masks=use_masks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    dataset_dir = r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes'

    print("Testing soft mask dataset...")
    ds = SoftMaskDataset(dataset_dir, split='train', use_masks=True)

    x, y = ds[0]
    print(f"\nInput shape: {x.shape}")
    print(f"Label: {y}")
    print(f"\nChannel stats:")
    print(f"  RGB (0-2): min={x[:3].min():.3f}, max={x[:3].max():.3f}")
    print(f"  GB  (3):   min={x[3].min():.3f}, max={x[3].max():.3f}, mean={x[3].mean():.3f}")
    print(f"  Tool(4):   min={x[4].min():.3f}, max={x[4].max():.3f}, mean={x[4].mean():.3f}")

    # Check distribution of soft mask values
    print(f"\nSoft mask distribution (GB channel):")
    gb_flat = x[3].flatten()
    print(f"  Values < -0.5: {(gb_flat < -0.5).sum().item()} ({100*(gb_flat < -0.5).float().mean():.1f}%)")
    print(f"  Values [-0.5, 0.5]: {((gb_flat >= -0.5) & (gb_flat <= 0.5)).sum().item()}")
    print(f"  Values > 0.5: {(gb_flat > 0.5).sum().item()} ({100*(gb_flat > 0.5).float().mean():.1f}%)")
