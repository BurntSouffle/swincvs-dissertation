#!/bin/bash
# Re-run ONLY the soft arm with weights/ redirected to RAM, because MooseFS corrupted
# the best-val checkpoint WRITE on the first attempt (training was fine, best-val 0.6269).
# Deterministic (seed 1) -> reproduces the identical soft model. Then score it and gate
# against the known hard control (test mAP 59.75, ECE 11.71).
set -uo pipefail
cd /workspace/SwinCVS
export DATASET_DIR=/dev/shm NUM_WORKERS=4 SWINCVS_AUTO=1 PYTHONUNBUFFERED=1
ROOT=/workspace/experiment_outputs/stage1_pair
IMNET=swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth
ts(){ date -u +%FT%TZ; }; log(){ echo "[$(ts)] $*"; }

# --- redirect weights/ -> RAM (idempotent) so checkpoint saves can't be MooseFS-corrupted ---
if [ ! -L weights ]; then
  mkdir -p /dev/shm/weights
  log "copying imagenet init into RAM weights (verified loadable)"
  ok=0
  for a in 1 2 3 4; do
    cp "weights/$IMNET" /dev/shm/weights/ 2>/dev/null
    if python -c "import torch; torch.load('/dev/shm/weights/$IMNET', map_location='cpu')" 2>/dev/null; then ok=1; break; fi
    log "  imagenet copy attempt $a corrupt, retry"; sleep 2
  done
  [ "$ok" = 1 ] || { log "FATAL: could not get a clean imagenet init into RAM"; exit 1; }
  mv weights weights_real && ln -s /dev/shm/weights weights
  log "weights/ -> /dev/shm/weights (RAM); original preserved at weights_real/"
fi

# --- deterministic soft re-run (checkpoint now saves to RAM) ---
log "re-running soft arm (seed 1, identical recipe; checkpoint -> RAM)"
python -u SwinCVS.py --config_path config/_pair_soft.yaml > "$ROOT/soft/train.log" 2>&1; rc=$?
log "soft re-run done (exit $rc); best-val=$(grep -oE 'val mAP=[0-9.]+' "$ROOT/soft/train.log" | tail -1)"
CKPT=weights/SwinV2_backbone_IMNP_F1a_sd1_bestMAP.pt
[ -f "$CKPT" ] || { log "FATAL: soft checkpoint still missing"; exit 1; }
python -c "import torch; torch.load('$CKPT', map_location='cpu')" || { log "FATAL: soft checkpoint corrupt even in RAM"; exit 1; }
log "soft checkpoint OK ($(du -h "$CKPT" | cut -f1))"

# --- score soft single-frame test ---
python -u scoring/infer_test.py --config config/_pair_soft.yaml --weights "SwinV2_backbone_IMNP_F1a_sd1_bestMAP.pt" \
    --split test --out "$ROOT/soft/preds_test.csv" > "$ROOT/soft/infer.log" 2>&1
python -u scoring/score_model.py --predictions "$ROOT/soft/preds_test.csv" \
    --annotation /dev/shm/endoscapes/test/annotation_ds_coco.json --name soft --out_dir "$ROOT/soft" \
    --provenance "Stage-1 soft single-frame test (RAM weights, deterministic re-run)" > "$ROOT/soft/score.log" 2>&1
cp "$CKPT" "$ROOT/soft/" 2>/dev/null || true
cp "$CKPT" weights_real/ 2>/dev/null || true   # persist to MooseFS

# --- GATE: soft vs known hard control ---
SM=$(grep -oE "Mean mAP: [0-9.]+" "$ROOT/soft/score.log" | grep -oE "[0-9.]+$")
SE=$(grep -E "\*\*Mean\*\*" "$ROOT/soft"/metrics_soft.md | sed -n '3p' | grep -oE "[0-9.]+" | head -1)
HM=59.75; HE=11.71
DELTA=$(awk "BEGIN{printf \"%.2f\", ${SM:-0}-$HM}")
echo "=== soft per-criterion ==="; grep -E "C[123] mAP|Mean mAP" "$ROOT/soft/score.log"
echo "=== GATE: soft mAP=$SM ECE=$SE | hard mAP=$HM ECE=$HE | delta(soft-hard)=$DELTA ==="
echo "SOFT_RERUN_DONE soft_mAP=$SM soft_ECE=$SE delta=$DELTA $(ts)"
