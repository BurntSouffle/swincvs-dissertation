"""
Training script for frozen backbone mask ablation experiment.

Run A: RGB only (3ch) -> Frozen SwinV2 -> Trainable LSTM + Head
Run B: RGB + Masks (5ch) -> Projection (5->3) -> Frozen SwinV2 -> Trainable LSTM + Head

Usage:
    python train_frozen.py --config config_rgb.yaml
    python train_frozen.py --config config_rgb_masks.yaml
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWINCVS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SWINCVS_ROOT)
sys.path.insert(0, os.path.join(SWINCVS_ROOT, 'scripts'))

import yaml
from dataset_gt_masks import get_gt_mask_dataloaders
from dataset_synthetic_masks import get_synthetic_mask_dataloaders
from dataset_soft_masks import get_soft_mask_dataloaders


class MaskProjection(nn.Module):
    """Projects 5-channel input (RGB + GB + Tool) to 3-channel for frozen backbone."""

    def __init__(self, in_channels=5, out_channels=3):
        super().__init__()
        # Simple 1x1 conv to project mask channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        # Initialize: RGB channels pass through, mask channels start small
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, :3, :, :] = torch.eye(3).view(3, 3, 1, 1)  # Identity for RGB
            self.proj.weight[:, 3:, :, :] = torch.randn(3, 2, 1, 1) * 0.02  # Small random for masks
            self.proj.bias.zero_()

    def forward(self, x):
        return self.proj(x)


class FrozenBackboneModel(nn.Module):
    """
    Model with frozen SwinV2 backbone and trainable head.

    For 5-channel input, adds a trainable projection layer before the frozen backbone.
    """

    def __init__(self, backbone, lstm_head, in_channels=3, use_mask_projection=False):
        super().__init__()
        self.in_channels = in_channels
        self.use_mask_projection = use_mask_projection

        if use_mask_projection:
            self.mask_proj = MaskProjection(in_channels=5, out_channels=3)
        else:
            self.mask_proj = None

        self.backbone = backbone
        self.lstm_head = lstm_head

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x shape: [B, C, H, W] for single frame

        if self.use_mask_projection and x.shape[1] == 5:
            x = self.mask_proj(x)  # [B, 5, H, W] -> [B, 3, H, W]

        # Get backbone features
        features = self.backbone.forward_features(x)  # [B, num_features]

        # Pass through LSTM head (treating single frame as sequence of 1)
        # Need to reshape for LSTM: [B, 1, features]
        features = features.unsqueeze(1)
        output = self.lstm_head(features)  # [B, 3]

        return output


class SimpleLSTMHead(nn.Module):
    """LSTM-based classification head."""

    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [B, seq_len, hidden_dim]
        # Take last timestep
        last_out = lstm_out[:, -1, :]  # [B, hidden_dim]
        out = self.dropout(last_out)
        out = self.fc(out)  # [B, num_classes]
        return out


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_frozen_model(config, checkpoint_path, device):
    """Build model with frozen backbone from checkpoint."""

    from m_swinv2 import SwinTransformerV2

    # Build backbone
    backbone = SwinTransformerV2(
        img_size=384,
        patch_size=4,
        in_chans=3,  # Always 3 for backbone
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

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # Filter backbone weights (remove LSTM/head weights)
    backbone_state = {}
    for k, v in state_dict.items():
        # Remove 'swinv2_model.' prefix if present
        if k.startswith('swinv2_model.'):
            new_k = k.replace('swinv2_model.', '')
            backbone_state[new_k] = v
        elif not k.startswith('lstm') and not k.startswith('fc') and not k.startswith('classifier'):
            backbone_state[k] = v

    # Load backbone weights (allow missing keys for head)
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    print(f"Backbone loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Get feature dimension from backbone
    feature_dim = backbone.num_features
    print(f"Backbone feature dim: {feature_dim}")

    # Build LSTM head
    lstm_head = SimpleLSTMHead(
        input_dim=feature_dim,
        hidden_dim=config.get('lstm_hidden', 256),
        num_layers=config.get('lstm_layers', 2),
        num_classes=3,
        dropout=config.get('dropout', 0.3)
    )

    # Build full model
    use_masks = config.get('use_masks', False)
    model = FrozenBackboneModel(
        backbone=backbone,
        lstm_head=lstm_head,
        in_channels=5 if use_masks else 3,
        use_mask_projection=use_masks
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params: {frozen_params:,}")

    return model


def calculate_metrics(preds, targets):
    """Calculate mAP and per-class AP."""
    preds = np.array(preds)
    targets = np.array(targets)

    aps = []
    for i in range(3):
        if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
            ap = average_precision_score(targets[:, i], preds[:, i])
        else:
            ap = 0.0  # Can't calculate AP if all same class
        aps.append(ap)

    mAP = np.mean(aps)
    return mAP, aps


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()

    # Only set trainable parts to train mode
    if hasattr(model, 'backbone'):
        model.backbone.eval()  # Keep frozen backbone in eval mode

    total_loss = 0
    all_preds = []
    all_targets = []

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    mAP, aps = calculate_metrics(all_preds, all_targets)

    return avg_loss, mAP, aps


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()

    total_loss = 0
    all_preds = []
    all_targets = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    mAP, aps = calculate_metrics(all_preds, all_targets)

    return avg_loss, mAP, aps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"\n{'='*60}")
    print(f"FROZEN BACKBONE MASK ABLATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Use masks: {config.get('use_masks', False)}")
    print()

    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    print("\nLoading data...")
    dataset_type = config.get('dataset_type', 'gt')  # 'gt', 'synthetic', or 'soft'

    if dataset_type == 'soft':
        print("Using SOFT mask dataset (SAM2 confidence, 1,933 frames)")
        train_loader, val_loader, test_loader = get_soft_mask_dataloaders(
            dataset_dir=config['dataset_dir'],
            use_masks=config.get('use_masks', False),
            batch_size=config.get('batch_size', 8),
            num_workers=config.get('num_workers', 0),
        )
    elif dataset_type == 'synthetic':
        print("Using SYNTHETIC mask dataset (binary, 1,933 frames)")
        train_loader, val_loader, test_loader = get_synthetic_mask_dataloaders(
            dataset_dir=config['dataset_dir'],
            use_masks=config.get('use_masks', False),
            batch_size=config.get('batch_size', 8),
            num_workers=config.get('num_workers', 0),
        )
    else:
        print("Using GT mask dataset (493 frames)")
        train_loader, val_loader, test_loader = get_gt_mask_dataloaders(
            dataset_dir=config['dataset_dir'],
            use_masks=config.get('use_masks', False),
            batch_size=config.get('batch_size', 8),
            num_workers=config.get('num_workers', 0),
            seed=seed
        )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Model
    print("\nBuilding model...")
    model = build_frozen_model(config, config['checkpoint_path'], device)

    # Loss with class weights
    class_weights = torch.tensor(config.get('class_weights', [1.0, 1.0, 1.0])).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('epochs', 30),
        eta_min=1e-6
    )

    scaler = GradScaler()

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    results = {
        'config': config,
        'train_losses': [],
        'val_losses': [],
        'train_maps': [],
        'val_maps': [],
        'val_aps': [],
        'best_epoch': 0,
        'best_val_map': 0,
    }

    best_val_map = 0
    patience_counter = 0
    patience = config.get('patience', 5)

    for epoch in range(1, config.get('epochs', 30) + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_map, train_aps = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_loss, val_map, val_aps = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{config.get('epochs', 30)} | "
              f"Train Loss: {train_loss:.4f} mAP: {train_map:.4f} | "
              f"Val Loss: {val_loss:.4f} mAP: {val_map:.4f} | "
              f"C1: {val_aps[0]:.3f} C2: {val_aps[1]:.3f} C3: {val_aps[2]:.3f} | "
              f"{epoch_time:.1f}s")

        # Save results
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['train_maps'].append(train_map)
        results['val_maps'].append(val_map)
        results['val_aps'].append(val_aps)

        # Check best
        if val_map > best_val_map:
            best_val_map = val_map
            results['best_epoch'] = epoch
            results['best_val_map'] = val_map
            patience_counter = 0

            # Save best model
            save_path = os.path.join(SCRIPT_DIR, 'results', f"{config['experiment_name']}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_map': val_map,
                'val_aps': val_aps
            }, save_path)
            print(f"  -> New best! Saved to {save_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
            break

    # Final test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")

    # Load best model
    best_path = os.path.join(SCRIPT_DIR, 'results', f"{config['experiment_name']}_best.pt")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_map, test_aps = validate(model, test_loader, criterion, device)

    print(f"Test mAP: {test_map:.4f}")
    print(f"Test C1 AP: {test_aps[0]:.4f}")
    print(f"Test C2 AP: {test_aps[1]:.4f}")
    print(f"Test C3 AP: {test_aps[2]:.4f}")

    results['test_map'] = test_map
    results['test_aps'] = test_aps
    results['test_loss'] = test_loss

    # Save results
    results_path = os.path.join(SCRIPT_DIR, 'results', f"{config['experiment_name']}_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in results.items()}
        results_json['val_aps'] = [list(x) for x in results['val_aps']]
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Best Val mAP: {results['best_val_map']:.4f}")
    print(f"Test mAP: {test_map:.4f}")


if __name__ == '__main__':
    main()
