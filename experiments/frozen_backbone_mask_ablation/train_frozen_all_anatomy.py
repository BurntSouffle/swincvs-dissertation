"""
Training script for frozen backbone mask ablation - ALL ANATOMY VERSION.

This experiment tests whether including ALL anatomy masks (not just GB + Tool)
improves CVS prediction, particularly for C1.

Input channels: 3 (RGB) + 6 (anatomy masks) = 9 channels
Anatomy masks:
  - cystic_plate (class 1)
  - calot_triangle (class 2)
  - cystic_artery (class 3) <- C1 relevant
  - cystic_duct (class 4)   <- C1 relevant
  - gallbladder (class 5)
  - tool (class 6)

Usage:
    python train_frozen_all_anatomy.py
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

from dataset_gt_masks_all_anatomy import get_gt_mask_all_anatomy_dataloaders, CLASS_NAMES


class MaskProjectionAllAnatomy(nn.Module):
    """Projects 9-channel input (RGB + 6 anatomy masks) to 3-channel for frozen backbone."""

    def __init__(self, in_channels=9, out_channels=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        # Initialize: RGB channels pass through, mask channels start small
        with torch.no_grad():
            self.proj.weight.zero_()
            # Identity for RGB (first 3 channels)
            self.proj.weight[:, :3, :, :] = torch.eye(3).view(3, 3, 1, 1)
            # Small random for mask channels (channels 3-8)
            self.proj.weight[:, 3:, :, :] = torch.randn(3, 6, 1, 1) * 0.02
            self.proj.bias.zero_()

    def forward(self, x):
        return self.proj(x)


class FrozenBackboneModelAllAnatomy(nn.Module):
    """
    Model with frozen SwinV2 backbone and trainable head.
    Supports 9-channel input (RGB + 6 anatomy masks).
    """

    def __init__(self, backbone, lstm_head, in_channels=9, use_mask_projection=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_mask_projection = use_mask_projection

        if use_mask_projection:
            self.mask_proj = MaskProjectionAllAnatomy(in_channels=in_channels, out_channels=3)
        else:
            self.mask_proj = None

        self.backbone = backbone
        self.lstm_head = lstm_head

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.use_mask_projection and x.shape[1] > 3:
            x = self.mask_proj(x)  # [B, 9, H, W] -> [B, 3, H, W]

        features = self.backbone.forward_features(x)
        features = features.unsqueeze(1)
        output = self.lstm_head(features)

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
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out)
        return out


def build_model(checkpoint_path, device, lstm_hidden=256, lstm_layers=2, dropout=0.3):
    """Build model with frozen backbone from checkpoint."""

    from m_swinv2 import SwinTransformerV2

    # Build backbone
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

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # Filter backbone weights
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith('swinv2_model.'):
            new_k = k.replace('swinv2_model.', '')
            backbone_state[new_k] = v
        elif not k.startswith('lstm') and not k.startswith('fc') and not k.startswith('classifier'):
            backbone_state[k] = v

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    print(f"Backbone loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    feature_dim = backbone.num_features
    print(f"Backbone feature dim: {feature_dim}")

    # Build LSTM head
    lstm_head = SimpleLSTMHead(
        input_dim=feature_dim,
        hidden_dim=lstm_hidden,
        num_layers=lstm_layers,
        num_classes=3,
        dropout=dropout
    )

    # Build full model with 9-channel input
    model = FrozenBackboneModelAllAnatomy(
        backbone=backbone,
        lstm_head=lstm_head,
        in_channels=9,
        use_mask_projection=True
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
            ap = 0.0
        aps.append(ap)

    mAP = np.mean(aps)
    return mAP, aps


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    if hasattr(model, 'backbone'):
        model.backbone.eval()

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
    # Configuration
    config = {
        'experiment_name': 'run_c_all_anatomy_masks',
        'seed': 42,
        'dataset_dir': r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes',
        'checkpoint_path': r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\SwinCVS\weights\SwinCVS_E2E_MC_IMNP_sd5_bestMAP.pt',
        'lstm_hidden': 256,
        'lstm_layers': 2,
        'dropout': 0.3,
        'epochs': 30,
        'batch_size': 8,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'patience': 7,
        'class_weights': [4.7, 6.5, 3.7],
        'num_workers': 0
    }

    print(f"\n{'='*70}")
    print("FROZEN BACKBONE MASK ABLATION - ALL ANATOMY MASKS")
    print(f"{'='*70}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Input: RGB (3ch) + ALL anatomy masks (6ch) = 9 channels")
    print(f"\nAnatomy mask channels:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Channel {i+3}: {name} (class {i+1})")
    print()

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_gt_mask_all_anatomy_dataloaders(
        dataset_dir=config['dataset_dir'],
        use_masks=True,  # Always use masks for this experiment
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        seed=config['seed']
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Model
    print("\nBuilding model...")
    model = build_model(
        config['checkpoint_path'],
        device,
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        dropout=config['dropout']
    )

    # Loss with class weights
    class_weights = torch.tensor(config['class_weights']).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )

    scaler = GradScaler()

    # Results storage
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

    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")

    best_val_map = 0
    patience_counter = 0

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()

        train_loss, train_map, train_aps = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_map, val_aps = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} mAP: {train_map:.4f} | "
              f"Val Loss: {val_loss:.4f} mAP: {val_map:.4f} | "
              f"C1: {val_aps[0]:.3f} C2: {val_aps[1]:.3f} C3: {val_aps[2]:.3f} | "
              f"{epoch_time:.1f}s")

        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['train_maps'].append(train_map)
        results['val_maps'].append(val_map)
        results['val_aps'].append(val_aps)

        if val_map > best_val_map:
            best_val_map = val_map
            results['best_epoch'] = epoch
            results['best_val_map'] = val_map
            patience_counter = 0

            # Save best model
            os.makedirs(os.path.join(SCRIPT_DIR, 'results'), exist_ok=True)
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

        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch} (patience={config['patience']})")
            break

    # Final test evaluation
    print(f"\n{'='*70}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*70}")

    # Load best model
    best_path = os.path.join(SCRIPT_DIR, 'results', f"{config['experiment_name']}_best.pt")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_map, test_aps = validate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  mAP:    {test_map:.4f}")
    print(f"  C1 AP:  {test_aps[0]:.4f}")
    print(f"  C2 AP:  {test_aps[1]:.4f}")
    print(f"  C3 AP:  {test_aps[2]:.4f}")

    results['test_map'] = test_map
    results['test_aps'] = test_aps
    results['test_loss'] = test_loss

    # Save results
    results_path = os.path.join(SCRIPT_DIR, 'results', f"{config['experiment_name']}_results.json")
    with open(results_path, 'w') as f:
        results_json = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in results.items()}
        results_json['val_aps'] = [list(x) for x in results['val_aps']]
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Comparison with previous results
    print(f"\n{'='*70}")
    print("COMPARISON WITH PREVIOUS EXPERIMENTS")
    print(f"{'='*70}")
    print(f"\n{'Experiment':<30} {'mAP':>10} {'C1':>10} {'C2':>10} {'C3':>10}")
    print("-" * 70)
    print(f"{'RGB Baseline':<30} {'87.62%':>10} {'83.22%':>10} {'96.17%':>10} {'83.48%':>10}")
    print(f"{'GB + Tool masks':<30} {'90.33%':>10} {'87.52%':>10} {'96.17%':>10} {'87.28%':>10}")
    print(f"{'ALL anatomy masks (this)':<30} {f'{test_map*100:.2f}%':>10} {f'{test_aps[0]*100:.2f}%':>10} {f'{test_aps[1]*100:.2f}%':>10} {f'{test_aps[2]*100:.2f}%':>10}")

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
