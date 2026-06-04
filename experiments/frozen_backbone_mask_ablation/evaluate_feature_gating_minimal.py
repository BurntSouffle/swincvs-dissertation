"""
Minimal evaluation script for Feature Gating checkpoint.
Avoids problematic imports (pandas, sklearn, cv2, timm).
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
import csv

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWINCVS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SWINCVS_ROOT)
sys.path.insert(0, os.path.join(SWINCVS_ROOT, 'scripts'))


def average_precision_score(y_true, y_score):
    """Calculate Average Precision."""
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate([[0], distinct_value_indices + 1])

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (threshold_idxs + 1) - tps
    precision = tps / (tps + fps)

    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0
    recall = tps / total_positives

    precision = np.concatenate([[1], precision])
    recall = np.concatenate([[0], recall])

    ap = np.sum(np.diff(recall) * precision[1:])
    return ap


# ============================================================================
# Inline SwinV2 model (minimal version needed for forward_features)
# ============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def to_2tuple(x):
    return (x, x) if not isinstance(x, tuple) else x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        l = (1. + torch.erf(torch.tensor((a - mean) / std / np.sqrt(2)))) / 2.
        u = (1. + torch.erf(torch.tensor((b - mean) / std / np.sqrt(2)))) / 2.
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * np.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).item()).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ============================================================================
# Model Architecture
# ============================================================================

class MaskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1024),
        )

    def forward(self, masks):
        x = self.conv(masks)
        return self.fc(x)


class SwinCVSWithGating(nn.Module):
    def __init__(self, swinv2_model, lstm_hidden=256, lstm_layers=2, num_classes=3):
        super().__init__()
        self.swinv2_model = swinv2_model
        self.mask_encoder = MaskEncoder()

        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0
        )
        self.fc_lstm = nn.Linear(lstm_hidden, num_classes)

    def forward(self, frames, masks):
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
            masks = masks.unsqueeze(1)

        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        masks_flat = masks.view(B * T, 2, H, W)

        with torch.no_grad():
            features = self.swinv2_model.forward_features(frames_flat)

        gate = self.mask_encoder(masks_flat)
        gate_sigmoid = torch.sigmoid(gate)
        gated_features = features * gate_sigmoid

        gated_features = gated_features.view(B, T, -1)
        lstm_out, _ = self.lstm(gated_features)
        last_out = lstm_out[:, -1, :]
        logits = self.fc_lstm(last_out)

        return logits, gate_sigmoid


# ============================================================================
# Dataset
# ============================================================================

class SimpleDataset(Dataset):
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

        # Load metadata using csv module
        self.metadata = {}
        with open(self.metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['vid']}_{row['frame']}"
                self.metadata[key] = row

        self.rgb_transform = transforms.Compose([
            transforms.CenterCrop(center_crop),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"[Dataset] Split: {split}, Samples: {len(self.frames)}")

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

        # Center crop and resize
        h, w = gb_conf.shape
        crop_size = self.center_crop
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2

        gb_cropped = gb_conf[start_h:start_h+crop_size, start_w:start_w+crop_size]
        tool_cropped = tool_conf[start_h:start_h+crop_size, start_w:start_w+crop_size]

        # Resize using torch
        gb_tensor_tmp = torch.from_numpy(gb_cropped).unsqueeze(0).unsqueeze(0)
        tool_tensor_tmp = torch.from_numpy(tool_cropped).unsqueeze(0).unsqueeze(0)

        gb_resized = F.interpolate(gb_tensor_tmp, size=(self.image_size, self.image_size),
                                   mode='bilinear', align_corners=False).squeeze().numpy()
        tool_resized = F.interpolate(tool_tensor_tmp, size=(self.image_size, self.image_size),
                                     mode='bilinear', align_corners=False).squeeze().numpy()

        # Normalize to [-1, 1]
        gb_norm = (gb_resized - 0.5) / 0.5
        tool_norm = (tool_resized - 0.5) / 0.5

        mask_tensor = torch.from_numpy(np.stack([gb_norm, tool_norm], axis=0)).float()

        # Get labels
        if vid_frame_key in self.metadata:
            row = self.metadata[vid_frame_key]
            c1 = 1.0 if float(row['C1']) >= 0.5 else 0.0
            c2 = 1.0 if float(row['C2']) >= 0.5 else 0.0
            c3 = 1.0 if float(row['C3']) >= 0.5 else 0.0
        else:
            c1, c2, c3 = 0.0, 0.0, 0.0

        label_tensor = torch.tensor([c1, c2, c3], dtype=torch.float32)

        return rgb_tensor, mask_tensor, label_tensor


# ============================================================================
# Main
# ============================================================================

def build_model(checkpoint_path, device):
    backbone = SwinTransformerV2(
        img_size=384, patch_size=4, in_chans=3, num_classes=1000, embed_dim=128,
        depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=24,
        pretrained_window_sizes=[12, 12, 12, 6], mlp_ratio=4, qkv_bias=True,
        drop_rate=0.0, drop_path_rate=0.2, ape=False, patch_norm=True
    )
    backbone.head = nn.Identity()

    model = SwinCVSWithGating(backbone)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Checkpoint epoch: {ckpt['epoch']}")
    print(f"Checkpoint val_map: {ckpt['val_map']:.4f}")

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    return model


def evaluate(model, dataloader, device):
    all_preds = []
    all_targets = []
    all_gate_stats = []

    with torch.no_grad():
        for batch_idx, (rgb, masks, targets) in enumerate(dataloader):
            rgb = rgb.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            logits, gate_sigmoid = model(rgb, masks)

            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

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

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    aps = []
    for i in range(3):
        if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
            ap = average_precision_score(targets[:, i], preds[:, i])
        else:
            ap = 0.0
        aps.append(ap)

    mAP = np.mean(aps)

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

    checkpoint_path = os.path.join(SWINCVS_ROOT, 'results', 'E4_FeatureGating_v1_best.pt')
    dataset_dir = r'C:\Users\sufia\Documents\Uni\Masters\DISSERTATION\endoscapes'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nBuilding model...")
    model = build_model(checkpoint_path, device)

    total_params = sum(p.numel() for p in model.parameters())
    mask_encoder_params = sum(p.numel() for p in model.mask_encoder.parameters())
    print(f"Total params: {total_params:,}")
    print(f"Mask encoder params: {mask_encoder_params:,}")

    print("\nLoading test data...")
    test_ds = SimpleDataset(dataset_dir, split='test')
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    print("\nEvaluating on test set...")
    test_mAP, test_aps, gate_stats = evaluate(model, test_loader, device)

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

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if gate_stats['std'] < 0.05:
        print("WARNING: Gate values have very low variance!")
        print("  -> The mask encoder is not learning discriminative gating.")
    elif gate_stats['mean'] < 0.3 or gate_stats['mean'] > 0.7:
        print("WARNING: Gate values are biased!")
        print(f"  -> Mean gate value of {gate_stats['mean']:.3f} suggests suboptimal gating.")
    else:
        print("Gate statistics look reasonable.")

    if test_mAP < 0.60:
        print("\nRECOMMENDATION: Abandon feature gating approach.")
    elif test_mAP < 0.68:
        print("\nRECOMMENDATION: Consider modifications to gating mechanism.")
    else:
        print("\nRECOMMENDATION: Continue training or try hyperparameter tuning.")

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
