#!/bin/bash
# SOFT-LABEL STAGE 1 — trains a per-frame SwinV2 backbone with soft annotator-
# agreement labels, to be FROZEN as the init for a Stage-2 frozen SwinCVS.
# Run from /workspace/SwinCVS:  bash launch_softstage1_seed.sh <SEED> [hard]
#   - no 2nd arg  -> SOFT labels (the intervention)
#   - "hard"      -> HARD-label control (SOFT_TRAIN_LABELS forced False)
#
# Produces weights/<EXP>_bestMAP.pt. Feed that filename into the Stage-2 frozen
# config's BACKBONE.PRETRAINED to complete: soft Stage-1 -> freeze -> Stage-2.
#
# NOTE: this exercises the bare-backbone (LSTM=False) single-frame path. Run the
# 1-epoch smoke test first (see scoring/ / session notes) before multi-seed.
set -euo pipefail

SEED="${1:?usage: bash launch_softstage1_seed.sh <SEED> [hard]}"
MODE="${2:-soft}"
cd /workspace/SwinCVS

export DATASET_DIR=/workspace
export NUM_WORKERS=4
export SWINCVS_AUTO=1
export PYTHONUNBUFFERED=1

TEMPLATE=config/SwinCVS_softstage1_sd1.yaml
CONFIG=config/SwinCVS_softstage1_${MODE}_sd${SEED}.yaml
sed -e "s/^SEED:.*/SEED: ${SEED}/" \
    -e "s/^EXPERIMENT_NAME:.*/EXPERIMENT_NAME: 'SwinCVS_softstage1_${MODE}_sd${SEED}'/" \
    "$TEMPLATE" > "$CONFIG"
if [ "$MODE" = "hard" ]; then
    sed -i "s/^  SOFT_TRAIN_LABELS:.*/  SOFT_TRAIN_LABELS: False/" "$CONFIG"
    EXP_NAME=SwinV2_backbone_IMNP_sd${SEED}          # no _F1a suffix for hard control
else
    EXP_NAME=SwinV2_backbone_IMNP_F1a_sd${SEED}      # auto-name with _F1a (soft)
fi

PERSIST=/workspace/experiment_outputs/softstage1_${MODE}_sd${SEED}
mkdir -p "$PERSIST" weights results
LAUNCH_LOG=$PERSIST/launch.log
exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "============================================================"
echo "SwinCVS SOFT-STAGE-1 (${MODE}) sd${SEED} — $(date -u +%FT%TZ)"
echo "config=$CONFIG  expected_weight=weights/${EXP_NAME}_bestMAP.pt"
echo "persist_dir=$PERSIST"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""; echo "[1/4] pip install requirements"
pip install -q -r requirements.txt
echo "  done"

echo ""; echo "[2/4] training: 8 epochs, bs=4, bare SwinV2 single-frame (${MODE} labels)"
TRAIN_START=$(date +%s)
python -u SwinCVS.py --config_path "$CONFIG" 2>&1 | tee "$PERSIST/train.log"
TRAIN_END=$(date +%s)
echo "  training wall-clock: $((TRAIN_END - TRAIN_START))s"

if ! grep -q "SwinV2 model selected" "$PERSIST/train.log"; then
    echo "FATAL: train.log did not print 'SwinV2 model selected' — LSTM=False did not take effect."; exit 2
fi
CKPT=weights/${EXP_NAME}_bestMAP.pt
[ -f "$CKPT" ] || { echo "FATAL: checkpoint $CKPT not found."; ls -la weights/; exit 3; }
echo "  Stage-1 backbone checkpoint: $CKPT ($(du -h "$CKPT" | awk '{print $1}'))"

echo ""; echo "[3/4] persistence"
cp "$CKPT" "$PERSIST/"
cp "results/${EXP_NAME}_results.json" "$PERSIST/training_journal.json"
cp "$CONFIG" "$PERSIST/config_used.yaml"
( cd /workspace/SwinCVS && \
  tar --sort=name --mtime='2000-01-01' --owner=0 --group=0 --numeric-owner -cf - \
      --exclude='weights' --exclude='results' --exclude='experiment_outputs' \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' . \
  | sha256sum | awk '{print $1}' ) > "$PERSIST/source_tree_sha256.txt"
echo "  pod source tree sha256: $(cat "$PERSIST/source_tree_sha256.txt")"

# Per-file sha256 manifest + bundle (Model-B lesson: verifiable before teardown).
{
  echo "# Manifest — SwinCVS_softstage1_${MODE}_sd${SEED}"
  echo "Generated: $(date -u +%FT%TZ)"
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "Source tree sha256: $(cat "$PERSIST/source_tree_sha256.txt")"
  echo "Training wall-clock: $((TRAIN_END - TRAIN_START))s"
  echo ""
  echo "## Files (size | sha256 | name)"
  for f in "$PERSIST"/*; do
    case "$f" in *artifacts.tar.gz|*manifest.txt) continue ;; esac
    [ -f "$f" ] || continue
    size=$(stat --printf='%s' "$f")
    sha=$(sha256sum "$f" | awk '{print $1}')
    printf "%12d %s  %s\n" "$size" "$sha" "$(basename "$f")"
  done
} > "$PERSIST/manifest.txt"
cat "$PERSIST/manifest.txt"

TAR=$PERSIST/artifacts.tar.gz
( cd "$PERSIST" && tar -czf artifacts.tar.gz --exclude=artifacts.tar.gz $(ls | grep -v artifacts.tar.gz) )
echo "wrote $TAR ($(du -h "$TAR" | awk '{print $1}'))"; sha256sum "$TAR"

echo ""; echo "[4/4] next step"
echo "  To complete the two-stage path, edit a frozen config's BACKBONE.PRETRAINED:"
echo "    PRETRAINED: '${EXP_NAME}_bestMAP.pt'"
echo "  then: bash launch_frozen_seed.sh ${SEED}   (after copying ${EXP_NAME}_bestMAP.pt into weights/)"
echo "============================================================"
