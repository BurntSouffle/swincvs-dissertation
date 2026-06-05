"""Generic dataloader inference for a SwinCVS checkpoint.

Loads {WEIGHTS} into a SwinCVS model in INFERENCE mode, iterates the
chosen split's dataloader, writes predictions CSV in the schema:
  video_id, frame, c{1,2,3}_pred, c{1,2,3}_label

The labels in the CSV are whatever the SwinCVS dataset returns for that
split — MV-rounded for val and test, soft-or-MV for train depending on
config.TRAIN.SOFT_TRAIN_LABELS. F3-style threshold-fitting flows want
val predictions; baseline scoring flows want test predictions.

Usage:
  python -u /workspace/SwinCVS/scoring/infer_test.py \
    --config config/SwinCVS_baseline_sd1.yaml \
    --weights SwinCVS_E2E_MC_IMNP_sd1_bestMAP.pt \
    --split test \
    --out /workspace/experiment_outputs/baseline_sd1/predictions_sd1.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

# Run from /workspace/SwinCVS so relative paths in the codebase work.
SWINCVS = Path("/workspace/SwinCVS")
os.chdir(SWINCVS)
sys.path.insert(0, str(SWINCVS))

from scripts.f_environment import get_config, set_deterministic_behaviour
from scripts.f_dataset import get_datasets, get_dataloaders
from scripts.f_build import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True,
                   help="filename inside weights/ (NOT a path)")
    p.add_argument("--split", default="test", choices=("train", "val", "test"),
                   help="which dataloader to iterate (default: test)")
    p.add_argument("--out", required=True, help="output CSV path")
    args = p.parse_args()

    cfg, _ = get_config(args.config)
    cfg.defrost()
    cfg.MODEL.INFERENCE = True
    cfg.MODEL.INFERENCE_WEIGHTS = args.weights
    cfg.freeze()
    set_deterministic_behaviour(cfg.SEED)

    train_ds, val_ds, test_ds = get_datasets(cfg)
    train_dl, val_dl, test_dl = get_dataloaders(cfg, train_ds, val_ds, test_ds)
    ds_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    dl_map = {"train": train_dl, "val": val_dl, "test": test_dl}
    target_ds = ds_map[args.split]
    target_dl = dl_map[args.split]
    print(f"{args.split} dataset: {len(target_ds)} sequences")

    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(SWINCVS / "weights" / args.weights,
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    df = target_ds.image_dataframe
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # F1b detection: the model has a fc_lstm_dis head iff trained with
    # MODEL.F1B_DISAGREEMENT_HEAD=True. When present, dataset labels are
    # 6-vectors [MV, disagreement] and the model returns (mv_logits,
    # dis_logits). We write a second CSV alongside the MV one with the
    # disagreement predictions and labels, harness-compatible schema:
    #   video_id, frame, d1_pred, d2_pred, d3_pred, d1_label, d2_label, d3_label
    is_f1b = hasattr(model, "fc_lstm_dis")
    if is_f1b:
        dis_out = out.parent / out.name.replace("predictions_", "predictions_disagreement_", 1)
        dis_rows = []
        print(f"F1b detected — also writing disagreement-head predictions to {dis_out}")

    rows = []
    t0 = time.time()
    with torch.inference_mode():
        for idx, (samples, targets) in enumerate(target_dl):
            samples = samples.to(device)
            if is_f1b:
                mv_logits, dis_logits = model(samples)
                mv_probs = torch.sigmoid(mv_logits).squeeze(0).cpu().numpy()
                dis_probs = torch.sigmoid(dis_logits).squeeze(0).cpu().numpy()
            else:
                mv_probs = torch.sigmoid(model(samples)).squeeze(0).cpu().numpy()
            t = targets.squeeze(0).numpy()
            stem = Path(df.iloc[idx]["f4"]).stem
            vid, frame = stem.split("_", 1)
            rows.append({
                "video_id": int(vid), "frame": int(frame),
                "c1_pred": float(mv_probs[0]), "c2_pred": float(mv_probs[1]),
                "c3_pred": float(mv_probs[2]),
                "c1_label": int(t[0]), "c2_label": int(t[1]),
                "c3_label": int(t[2]),
            })
            if is_f1b:
                # F1b targets are 6-vec [MV, disagreement]; slice indices 3..6
                dis_rows.append({
                    "video_id": int(vid), "frame": int(frame),
                    "d1_pred": float(dis_probs[0]), "d2_pred": float(dis_probs[1]),
                    "d3_pred": float(dis_probs[2]),
                    "d1_label": int(t[3]), "d2_label": int(t[4]),
                    "d3_label": int(t[5]),
                })
            if (idx + 1) % 200 == 0 or idx == 0:
                elapsed = time.time() - t0
                eta = elapsed / (idx + 1) * (len(target_dl) - idx - 1)
                print(f"  {idx+1}/{len(target_dl)}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    print(f"inference done in {time.time()-t0:.1f}s ({len(rows)} samples)")
    fields = ["video_id", "frame", "c1_pred", "c2_pred", "c3_pred",
              "c1_label", "c2_label", "c3_label"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    if is_f1b:
        dis_fields = ["video_id", "frame", "d1_pred", "d2_pred", "d3_pred",
                      "d1_label", "d2_label", "d3_label"]
        with dis_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=dis_fields)
            w.writeheader()
            w.writerows(dis_rows)
        print(f"wrote {dis_out} ({len(dis_rows)} rows)")


if __name__ == "__main__":
    main()
