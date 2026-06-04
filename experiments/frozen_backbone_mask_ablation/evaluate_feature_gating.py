"""
Evaluate the Feature Gating checkpoint (E4_FeatureGating_v1_best.pt) on test set.

Reconstructs the model architecture based on checkpoint structure:
- mask_encoder: CNN encoder for 2-channel masks -> 1024-dim gating signal
- SwinV2 backbone (frozen during training)
- LSTM classifier
- Gating: features * sigmoid(mask_encoder(masks))
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import glob
import pandas as pd


def average_precision_score_manual(y_true, y_score):
    """Calculate Average Precision without sklearn."""
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Get distinct values for thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate([[0], distinct_value_indices + 1])

    # Calculate precision and recall at each threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (threshold_idxs + 1) - tps
    precision = tps / (tps + fps)

    # Recall
    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0
    recall = tps / total_positives

    # Add sentinel values
    precision = np.concatenate([[1], precision])
    recall = np.concatenate([[0], recall])

    # Calculate AP using trapezoidal rule
    ap = np.sum(np.diff(recall) * precision[1:])
    return ap

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWINCVS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SWINCVS_ROOT)
sys.path.insert(0, os.path.join(SWINCVS_ROOT, 'scripts'))

from m_swinv2 import SwinTransformerV2


class MaskEncoder(nn.Module):
    """
    CNN encoder for mask inputs (2 channels: gallbladder + tool).
    Reconstructed from checkpoint structure:
    - conv.0: Conv2d(2, 32, 7x7) + conv.1: BatchNorm2d(32)
    - conv.3: Conv2d(32, 64, 5x5) + conv.4: BatchNorm2d(64)
    - conv.6: Conv2d(64, 128, 3x3) + conv.7: BatchNorm2d(128)
    - fc.0: Linear(128, 256) + ReLU + Dropout
    - fc.3: Linear(256, 1024)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=4, padding=3),   # 0
            nn.BatchNorm2d(32),                                      # 1
            nn.ReLU(inplace=True),                                   # 2
            nn.Conv2d(32, 64, kernel_size=5, stride=4, padding=2),  # 3
            nn.BatchNorm2d(64),                                      # 4
            nn.ReLU(inplace=True),                                   # 5
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 6
            nn.BatchNorm2d(128),                                     # 7
            nn.ReLU(inplace=True),                                   # 8
            nn.AdaptiveAvgPool2d(1),                                 # 9
            nn.Flatten(),                                            # 10
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 256),     # 0
            nn.ReLU(inplace=True),   # 1
            nn.Dropout(0.3),         # 2
            nn.Linear(256, 1024),    # 3
        )

    def forward(self, masks):
        x = self.conv(masks)
        return self.fc(x)


class SwinCVSWithGating(nn.Module):
    """
    SwinCVS with feature gating mechanism.

    Architecture:
    1. Extract features from frames using frozen SwinV2
    2. Encode masks to gating signal using MaskEncoder
    3. Gate features: gated_features = features * sigmoid(gate)
    4. Classify using LSTM
    """

    def __init__(self, swinv2_model, lstm_hidden=256, lstm_layers=2, num_classes=3):
        super().__init__()
        self.swinv2_model = swinv2_model
        self.mask_encoder = MaskEncoder()

        # LSTM for temporal modeling (if using sequences)
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0  # No dropout between layers based on checkpoint
        )

        # Final classifier
        self.fc_lstm = nn.Linear(lstm_hidden, num_classes)

    def forward(self, frames, masks):
        """
        Args:
            frames: [B, T, 3, H, W] or [B, 3, H, W] for single frames
            masks: [B, T, 2, H, W] or [B, 2, H, W] for single frames
        """
        # Handle single frame vs sequence
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)  # [B, 1, 3, H, W]
            masks = masks.unsqueeze(1)    # [B, 1, 2, H, W]

        B, T, C, H, W = frames.shape

        # Flatten batch and time for backbone
        frames_flat = frames.view(B * T, C, H, W)
        masks_flat = masks.view(B * T, 2, H, W)

        # Get SwinV2 features (frozen)
        with torch.no_grad():
            features = self.swinv2_model.forward_features(frames_flat)  # [B*T, 1024]

        # Get gating signal from masks
        gate = self.mask_encoder(masks_flat)  # [B*T, 1024]
        gate_sigmoid = torch.sigmoid(gate)

        # Apply gating
        gated_features = features * gate_sigmoid  # [B*T, 1024]

        # Reshape for LSTM
        gated_features = gated_features.view(B, T, -1)  # [B, T, 1024]

        # LSTM
        lstm_out, _ = self.lstm(gated_features)  # [B, T, hidden]

        # Take last timestep
        last_out = lstm_out[:, -1, :]  # [B, hidden]

        # Classify
        logits = self.fc_lstm(last_out)  # [B, num_classes]

        return logits, gate_sigmoid


class SoftMaskDatasetSeparate(Dataset):
    """
    Dataset that returns RGB and masks separately for feature gating evaluation.
    """

    def __init__(self, dataset_dir, split='test', image_size=384, center_crop=480):
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_size = image_size
        self.center_crop = center_crop

        self.image_dir = os.path.join(dataset_dir, split)
        self.confidence_dir = os.path.join(dataset_dir, 'synthetic_masks', split, 'confidence')
        self.metadata_path = os.path.join(dataset_dir, 'all_metadata.csv')

        # Load COCO annotations
        coco_path = os.path.join(self.image_dir, 'annotation_coco.json')
        with open(coco_path, 'r') as f:
            coco = json.load(f)

        self.frames = []
        for img in coco['images']:
            fname = img['file_name'].replace('.jpg', '')
            parts = fname.split('_')
            vid = int(parts[0])
            frame = int(parts[1])
            self.frames.append({'vid': vid, 'frame': frame, 'filename': img['file_name']})

        # Load metadata
        df = pd.read_csv(self.metadata_path)
        df['vid_frame'] = df['vid'].astype(str) + '_' + df['frame'].astype(str)
        self.metadata = df.set_index('vid_frame')

        # RGB transform
        self.rgb_transform = transforms.Compose([
            transforms.CenterCrop(center_crop),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"[SoftMaskDatasetSeparate] Split: {split}, Samples: {len(self.frames)}")

    def __len__(self):
        return len(self.frames)

    def _load_soft_mask(self, vid, frame, class_name):
        pattern = os.path.join(self.confidence_dir, f"{vid}_{frame}_{class_name}_*.npy")
        files = glob.glob(pattern)

        if not files:
            return np.zeros((480, 854), dtype=np.float32)

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

        # Load RGB
        img_path = os.path.join(self.image_dir, frame_info['filename'])
        rgb_img = Image.open(img_path).convert('RGB')
        rgb_tensor = self.rgb_transform(rgb_img)

        # Load soft masks
        gb_conf = self._load_soft_mask(vid, frame, 'gallbladder')
        tool_conf = self._load_soft_mask(vid, frame, 'tool')

        # Center crop and resize masks
        h, w = gb_conf.shape
        crop_size = self.center_crop
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2

        gb_cropped = gb_conf[start_h:start_h+crop_size, start_w:start_w+crop_size]
        tool_cropped = tool_conf[start_h:start_h+crop_size, start_w:start_w+crop_size]

        # Resize using torch interpolate
        gb_tensor_tmp = torch.from_numpy(gb_cropped).unsqueeze(0).unsqueeze(0)
        tool_tensor_tmp = torch.from_numpy(tool_cropped).unsqueeze(0).unsqueeze(0)

        gb_resized = F.interpolate(gb_tensor_tmp, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze().numpy()
        tool_resized = F.interpolate(tool_tensor_tmp, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze().numpy()

        # Normalize masks to [-1, 1]
        gb_norm = (gb_resized - 0.5) / 0.5
        tool_norm = (tool_resized - 0.5) / 0.5

        # Stack masks [2, H, W]
        mask_tensor = torch.from_numpy(np.stack([gb_norm, tool_norm], axis=0)).float()

        # Get labels
        if vid_frame_key in self.metadata.index:
            row = self.metadata.loc[vid_frame_key]
            c1 = 1.0 if row['C1'] >= 0.5 else 0.0
            c2 = 1.0 if row['C2'] >= 0.5 else 0.0
            c3 = 1.0 if row['C3'] >= 0.5 else 0.0
        else:
            c1, c2, c3 = 0.0, 0.0, 0.0

        label_tensor = torch.tensor([c1, c2, c3], dtype=torch.float32)

        return rgb_tensor, mask_tensor, label_tensor


def build_model(checkpoint_path, device):
    """Build model and load weights."""

    # Build SwinV2 backbone
    backbone = SwinTransformerV2(
        img_size=384,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=24,
        pretrained_window_sizes=[12, 12, 12, 6],
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )

    # Remove backbone head
    backbone.head = nn.Identity()

    # Build full model
    model = SwinCVSWithGating(backbone)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Checkpoint epoch: {ckpt['epoch']}")
    print(f"Checkpoint val_map: {ckpt['val_map']:.4f}")

    # Load state dict
    state_dict = ckpt['model_state_dict']

    # Check for missing/unexpected keys
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"Missing keys: {len(missing)}")
        for k in list(missing)[:5]:
            print(f"  {k}")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for k in list(unexpected)[:5]:
            print(f"  {k}")

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


def evaluate(model, dataloader, device):
    """Evaluate model on test set."""

    all_preds = []
    all_targets = []
    all_gate_stats = []

    with torch.no_grad():
        for batch_idx, (rgb, masks, targets) in enumerate(dataloader):
            rgb = rgb.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            # Forward pass
            logits, gate_sigmoid = model(rgb, masks)

            # Collect predictions
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Collect gate statistics
            gate_stats = {
                'min': gate_sigmoid.min().item(),
                'max': gate_sigmoid.max().item(),
                'mean': gate_sigmoid.mean().item(),
                'std': gate_sigmoid.std().item(),
            }
            all_gate_stats.append(gate_stats)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}", end='\r')

    print()

    # Calculate metrics
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    aps = []
    for i in range(3):
        if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
            ap = average_precision_score_manual(targets[:, i], preds[:, i])
        else:
            ap = 0.0
        aps.append(ap)

    mAP = np.mean(aps)

    # Aggregate gate statistics
    gate_summary = {
        'min': np.mean([g['min'] for g in all_gate_stats]),
        'max': np.mean([g['max'] for g in all_gate_stats]),
        'mean': np.mean([g['mean'] for g in all_gate_stats]),
        'std': np.mean([g['std'] for g in all_gate_stats]),
    }

    return mAP, aps, gate_summary


def main():
    print("="*60)
    print("FEATURE GATING CHECKPOINT EVALUATION")
    print("="*60)

    # Paths
    checkpoint_path = os.path.join(SWINCVS_ROOT, 'results', 'E4_FeatureGating_v1_best.pt')
    dataset_dir = r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build model
    print("\nBuilding model...")
    model = build_model(checkpoint_path, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    mask_encoder_params = sum(p.numel() for p in model.mask_encoder.parameters())
    print(f"Total params: {total_params:,}")
    print(f"Mask encoder params: {mask_encoder_params:,}")

    # Load test data
    print("\nLoading test data...")
    test_ds = SoftMaskDatasetSeparate(dataset_dir, split='test')
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # Evaluate
    print("\nEvaluating on test set...")
    test_mAP, test_aps, gate_stats = evaluate(model, test_loader, device)

    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nTest mAP: {test_mAP:.4f} (Baseline: 0.6882)")
    print(f"  C1 AP: {test_aps[0]:.4f}")
    print(f"  C2 AP: {test_aps[1]:.4f}")
    print(f"  C3 AP: {test_aps[2]:.4f}")

    print(f"\nGate Statistics:")
    print(f"  Min:  {gate_stats['min']:.4f}")
    print(f"  Max:  {gate_stats['max']:.4f}")
    print(f"  Mean: {gate_stats['mean']:.4f}")
    print(f"  Std:  {gate_stats['std']:.4f}")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if gate_stats['std'] < 0.05:
        print("WARNING: Gate values have very low variance!")
        print("  -> The mask encoder is not learning discriminative gating.")
        print("  -> Gates are essentially constant (not using mask information).")
    elif gate_stats['mean'] < 0.3 or gate_stats['mean'] > 0.7:
        print("WARNING: Gate values are biased!")
        print(f"  -> Mean gate value of {gate_stats['mean']:.3f} suggests the model")
        print("     is either suppressing too much (mean~0) or gating is ineffective (mean~1).")
    else:
        print("Gate statistics look reasonable.")
        print("  -> The mask encoder appears to be learning meaningful gating.")

    if test_mAP < 0.60:
        print("\nRECOMMENDATION: Abandon feature gating approach.")
        print("  -> Test mAP is significantly below baseline.")
    elif test_mAP < 0.68:
        print("\nRECOMMENDATION: Consider modifications to gating mechanism.")
        print("  -> Performance gap suggests potential but current approach is suboptimal.")
    else:
        print("\nRECOMMENDATION: Continue training or try hyperparameter tuning.")

    # Save results
    results = {
        'test_mAP': test_mAP,
        'test_aps': test_aps,
        'gate_stats': gate_stats,
        'checkpoint_epoch': 6,
        'checkpoint_val_map': 0.597,
    }

    results_path = os.path.join(SCRIPT_DIR, 'results', 'E4_FeatureGating_v1_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
