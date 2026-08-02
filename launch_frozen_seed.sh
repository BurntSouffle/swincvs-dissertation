#!/bin/bash
# FROZEN SwinCVS — the decision-critical run. Reproduces the paper's frozen
# recipe (Stage-1 endoscapes backbone frozen, LSTM+head trained) in OUR harness.
# Run from /workspace/SwinCVS on the A100 80GB pod:  bash launch_frozen_seed.sh <SEED>
#
# Gates the SOTA plan: does our frozen base land ~67 (plan live), ~64 (soft
# labels must carry more), or ~62 (freezing premise fails)?
#
# RUN ONE SEED FIRST and report before launching the multi-seed plan.
#
# Mirrors launch_seed1.sh: probe -> train -> infer -> score -> persist.
set -euo pipefail

SEED="${1:?usage: bash launch_frozen_seed.sh <SEED>}"
cd /workspace/SwinCVS

export DATASET_DIR=/workspace
export NUM_WORKERS=4
export SWINCVS_AUTO=1
export PYTHONUNBUFFERED=1

# Per-seed config from the sd1 template (only SEED + EXPERIMENT_NAME change).
TEMPLATE=config/SwinCVS_frozen_sd1.yaml
CONFIG=config/SwinCVS_frozen_sd${SEED}.yaml
# Write via a temp file then move — for SEED=1, CONFIG==TEMPLATE and a direct
# `sed TEMPLATE > CONFIG` would truncate the template to empty before sed reads it.
TMPCFG=$(mktemp)
sed -e "s/^SEED:.*/SEED: ${SEED}/" \
    -e "s/^EXPERIMENT_NAME:.*/EXPERIMENT_NAME: 'SwinCVS_frozen_sd${SEED}'/" \
    "$TEMPLATE" > "$TMPCFG"
mv "$TMPCFG" "$CONFIG"

EXP_NAME=SwinCVS_frozen_ENDP_sd${SEED}   # auto-name from f_environment.py (LSTM, !E2E, ENDP)
PERSIST=/workspace/experiment_outputs/frozen_sd${SEED}
mkdir -p "$PERSIST" weights results

LAUNCH_LOG=$PERSIST/launch.log
exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "============================================================"
echo "SwinCVS FROZEN sd${SEED} launch — $(date -u +%FT%TZ)"
echo "config=$CONFIG  expected_weight=weights/${EXP_NAME}_bestMAP.pt"
echo "backbone(frozen)=Swin_backbone_no_augm_sd4_bestMAP.pt"
echo "persist_dir=$PERSIST"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo ""; echo "[1/6] pip install requirements"
pip install -q -r requirements.txt
echo "  done"

echo ""; echo "[2/6] memory probe at bs=4"
python -u scoring/memory_probe.py --config "$CONFIG" --steps 5 \
    2>&1 | tee "$PERSIST/memory_probe.log"
if ! grep -q "PROBE PASS" "$PERSIST/memory_probe.log"; then
    echo "FATAL: memory probe did not print 'PROBE PASS' — aborting."; exit 1
fi

echo ""; echo "[3/6] training: 8 epochs, bs=4, frozen backbone, best-val reload"
TRAIN_START=$(date +%s)
python -u SwinCVS.py --config_path "$CONFIG" 2>&1 | tee "$PERSIST/train.log"
TRAIN_END=$(date +%s)
echo "  training wall-clock: $((TRAIN_END - TRAIN_START))s"

if ! grep -q "Frozen SwinCVS" "$PERSIST/train.log"; then
    echo "FATAL: train.log did not print 'Frozen SwinCVS' — freeze flag did not take effect."; exit 2
fi
if ! grep -q "Loading best-val checkpoint for test eval" "$PERSIST/train.log"; then
    echo "FATAL: best-val reload line missing."; exit 2
fi
CKPT=weights/${EXP_NAME}_bestMAP.pt
[ -f "$CKPT" ] || { echo "FATAL: checkpoint $CKPT not found."; ls -la weights/; exit 3; }
echo "  checkpoint at: $CKPT ($(du -h "$CKPT" | awk '{print $1}'))"

echo ""; echo "[4/6] test inference"
python -u scoring/infer_test.py \
    --config "$CONFIG" --weights "${EXP_NAME}_bestMAP.pt" --split test \
    --out "$PERSIST/predictions_frozen_sd${SEED}.csv" \
    2>&1 | tee "$PERSIST/inference.log"

echo ""; echo "[5/6] harness scoring"
python -u scoring/score_model.py \
    --predictions "$PERSIST/predictions_frozen_sd${SEED}.csv" \
    --annotation /workspace/endoscapes/test/annotation_ds_coco.json \
    --name "SwinCVS_frozen_sd${SEED}" \
    --out_dir "$PERSIST" \
    --provenance "Frozen SwinCVS (paper recipe) seed ${SEED}: A100 80GB bs=4, 8 epochs, Stage-1 backbone Swin_backbone_no_augm_sd4 FROZEN, LSTM+fc_lstm trained, best-val reload. Reproduction of the 67.45 frozen number in our harness." \
    2>&1 | tee "$PERSIST/harness.log"

echo ""; echo "[6/6] persistence"
cp "$CKPT" "$PERSIST/"
cp "results/${EXP_NAME}_results.json" "$PERSIST/training_journal.json"
cp "$CONFIG" "$PERSIST/config_used.yaml"
( cd /workspace/SwinCVS && \
  tar --sort=name --mtime='2000-01-01' --owner=0 --group=0 --numeric-owner -cf - \
      --exclude='weights' --exclude='results' --exclude='experiment_outputs' \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' . \
  | sha256sum | awk '{print $1}' ) > "$PERSIST/source_tree_sha256.txt"
echo "  pod source tree sha256: $(cat "$PERSIST/source_tree_sha256.txt")"

# Per-file sha256 manifest (Model-B lesson: verifiable before teardown).
{
  echo "# Manifest — SwinCVS_frozen_sd${SEED} (frozen recipe, seed ${SEED})"
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
echo ""; echo "wrote $TAR ($(du -h "$TAR" | awk '{print $1}'))"; sha256sum "$TAR"
echo ""
echo "============================================================"
echo "DONE frozen sd${SEED}. Pull with:"
echo "  scp runpod:$TAR ./swincvs_analysis/runpod_artifacts/frozen_sd${SEED}/"
echo "============================================================"
