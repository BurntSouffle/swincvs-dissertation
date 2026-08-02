#!/bin/bash
# Multi-seed the controlled soft-vs-hard Stage-1 pair (seeds 2 & 3) to CONSOLIDATE the
# seed-1 finding (+1.15pp, C1/C3 up, C2 down, ECE better). Same LOCKED recipe as seed 1
# (backbone 1e-5, head 1e-3, 8 ep), arms differ ONLY in labels. Reads + checkpoint-writes
# both in /dev/shm (RAM) -> reliable on the degraded MooseFS. Then runs consolidate.py.
set -uo pipefail
cd /workspace/SwinCVS
export DATASET_DIR=/dev/shm NUM_WORKERS=4 SWINCVS_AUTO=1 PYTHONUNBUFFERED=1
ROOT=/workspace/experiment_outputs/stage1_pair
ANNO=/dev/shm/endoscapes/test/annotation_ds_coco.json
TEMPLATE=config/SwinCVS_softstage1_sd1.yaml
BB_LR=0.00001; EPOCHS=8
ts(){ date -u +%FT%TZ; }; log(){ echo "[$(ts)] $*"; }
gen_config(){ local tmp; tmp=$(mktemp); sed -e "s/^SEED:.*/SEED: $2/" -e "s/^  EPOCHS:.*/  EPOCHS: $4/" \
    -e "s/^    ENCODER_LR:.*/    ENCODER_LR: $3/" -e "s/^  SOFT_TRAIN_LABELS:.*/  SOFT_TRAIN_LABELS: $5/" \
    -e "s|^DATASET_DIR:.*|DATASET_DIR: '/dev/shm'|" "$TEMPLATE" > "$tmp"; mv "$tmp" "$1"; }
bestval(){ grep -oE "val mAP=[0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$"; }

run(){ # <tag> <seed> <soft> <exp>
  local tag="$1" sd="$2" soft="$3" exp="$4" pdir="$ROOT/$1" cfg="config/_ms_$1.yaml"
  mkdir -p "$pdir"; gen_config "$cfg" "$sd" "$BB_LR" "$EPOCHS" "$soft"; cp "$cfg" "$pdir/config_used.yaml"
  log "TRAIN $tag (seed=$sd soft=$soft) bb_lr=$BB_LR head_lr=1e-3 (RAM)"
  python -u SwinCVS.py --config_path "$cfg" > "$pdir/train.log" 2>&1; local rc=$?
  log "  $tag done (exit $rc) best-val=$(bestval "$pdir/train.log")"
  local ckpt="weights/${exp}_bestMAP.pt"
  if [ -f "$ckpt" ] && python -c "import torch; torch.load('$ckpt', map_location='cpu')" 2>/dev/null; then
    python -u scoring/infer_test.py --config "$cfg" --weights "${exp}_bestMAP.pt" --split test \
        --out "$pdir/preds_test.csv" > "$pdir/infer.log" 2>&1
    python -u scoring/score_model.py --predictions "$pdir/preds_test.csv" --annotation "$ANNO" \
        --name "$tag" --out_dir "$pdir" --provenance "Stage-1 $tag single-frame test (RAM)" > "$pdir/score.log" 2>&1
    cp "$ckpt" "$pdir/" 2>/dev/null || true
    log "  $tag scored: $(grep -oE 'Mean mAP: [0-9.]+' "$pdir/score.log" | tail -1)"
  else log "  WARN $tag: checkpoint missing/corrupt — see $pdir/train.log"; fi
}

echo "============================================================"
echo "MULTI-SEED CONTROLLED PAIR (seeds 2,3) — start $(ts)"
echo "============================================================"
pip install -q -r requirements.txt 2>&1 | tail -1
for sd in 2 3; do
  run hard_sd$sd $sd False SwinV2_backbone_IMNP_sd$sd
  run soft_sd$sd $sd True  SwinV2_backbone_IMNP_F1a_sd$sd
done
log "all multi-seed runs done; consolidating 3-seed panel"
python3 -u /workspace/consolidate.py > "$ROOT/CONSOLIDATED_REPORT.md" 2>&1 || true
echo "============================================================"; cat "$ROOT/CONSOLIDATED_REPORT.md"
echo "============================================================"; echo "MULTISEED_DONE $(ts)"
