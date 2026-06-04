"""Generic test-set inference for a SwinCVS checkpoint.

Loads {WEIGHTS} into a SwinCVS model in INFERENCE mode, iterates the test
dataloader, writes predictions CSV in the predictions.csv schema:
  video_id, frame, c{1,2,3}_pred, c{1,2,3}_label

Used by launch_seed1.sh on the RunPod pod after training.

Usage:
  python -u /workspace/SwinCVS/scoring/infer_test.py \
    --config config/SwinCVS_baseline_sd1.yaml \
    --weights SwinCVS_E2E_MC_IMNP_sd1_bestMAP.pt \
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
    p.add_argument("--out", required=True, help="output CSV path")
    args = p.parse_args()

    cfg, _ = get_config(args.config)
    cfg.defrost()
    cfg.MODEL.INFERENCE = True
    cfg.MODEL.INFERENCE_WEIGHTS = args.weights
    cfg.freeze()
    set_deterministic_behaviour(cfg.SEED)

    _, _, test_ds = get_datasets(cfg)
    _, _, test_dl = get_dataloaders(cfg, test_ds, test_ds, test_ds)
    print(f"test dataset: {len(test_ds)} sequences")

    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(SWINCVS / "weights" / args.weights,
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    df = test_ds.image_dataframe
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    with torch.inference_mode():
        for idx, (samples, targets) in enumerate(test_dl):
            samples = samples.to(device)
            probs = torch.sigmoid(model(samples)).squeeze(0).cpu().numpy()
            t = targets.squeeze(0).numpy()
            stem = Path(df.iloc[idx]["f4"]).stem
            vid, frame = stem.split("_", 1)
            rows.append({
                "video_id": int(vid), "frame": int(frame),
                "c1_pred": float(probs[0]), "c2_pred": float(probs[1]),
                "c3_pred": float(probs[2]),
                "c1_label": int(t[0]), "c2_label": int(t[1]),
                "c3_label": int(t[2]),
            })
            if (idx + 1) % 200 == 0 or idx == 0:
                elapsed = time.time() - t0
                eta = elapsed / (idx + 1) * (len(test_dl) - idx - 1)
                print(f"  {idx+1}/{len(test_dl)}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    print(f"inference done in {time.time()-t0:.1f}s ({len(rows)} samples)")
    fields = ["video_id", "frame", "c1_pred", "c2_pred", "c3_pred",
              "c1_label", "c2_label", "c3_label"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
