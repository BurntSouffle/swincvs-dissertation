#!/bin/bash
# Locked-recipe baseline, seed 1. First paid RunPod run of the intervention
# phase. Run from /workspace/SwinCVS on the A100 80GB pod.
#
# Stages:
#   1. pip install requirements
#   2. memory probe at bs=4 (fail fast on OOM)
#   3. full 8-epoch training (uses the fixed checkpoint-selection in SwinCVS.py)
#   4. test inference -> predictions CSV
#   5. harness scoring -> metrics JSON + md
#   6. persistence -> tar + sha256 manifest in /workspace/experiment_outputs/baseline_sd1/
#
# Recovery: if a stage fails, fix the cause and re-run from that stage. The
# script is idempotent across re-runs because outputs are addressed by name.
set -euo pipefail

cd /workspace/SwinCVS

export DATASET_DIR=/workspace
export NUM_WORKERS=4
export SWINCVS_AUTO=1
export PYTHONUNBUFFERED=1

CONFIG=config/SwinCVS_baseline_sd1.yaml
EXP_NAME=SwinCVS_E2E_MC_IMNP_sd1   # constructed by scripts/f_environment.py:validate_config
PERSIST=/workspace/experiment_outputs/baseline_sd1
mkdir -p $PERSIST weights results

LAUNCH_LOG=$PERSIST/launch.log
exec > >(tee -a $LAUNCH_LOG) 2>&1

echo "============================================================"
echo "SwinCVS baseline sd1 launch — $(date -u +%FT%TZ)"
echo "config=$CONFIG  expected_weight=weights/${EXP_NAME}_bestMAP.pt"
echo "persist_dir=$PERSIST"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# ── 1. pip install ────────────────────────────────────────────────────────
echo ""
echo "[1/6] pip install requirements"
pip install -q -r requirements.txt
echo "  done"

# ── 2. memory probe (bs=4, 5 fwd+bwd steps) ───────────────────────────────
echo ""
echo "[2/6] memory probe at bs=4"
python -u scoring/memory_probe.py --config $CONFIG --steps 5 \
    2>&1 | tee $PERSIST/memory_probe.log
if ! grep -q "PROBE PASS" $PERSIST/memory_probe.log; then
    echo "FATAL: memory probe did not print 'PROBE PASS' — aborting."
    exit 1
fi

# ── 3. full 8-epoch training ──────────────────────────────────────────────
echo ""
echo "[3/6] training: 8 epochs, bs=4, best-val reload from disk"
TRAIN_START=$(date +%s)
python -u SwinCVS.py --config_path $CONFIG 2>&1 | tee $PERSIST/train.log
TRAIN_END=$(date +%s)
echo "  training wall-clock: $((TRAIN_END - TRAIN_START))s"

# Sanity: best-val reload must be present in the log
if ! grep -q "Loading best-val checkpoint for test eval" $PERSIST/train.log; then
    echo "FATAL: SwinCVS.py did not print best-val reload line — fix did not take effect."
    exit 2
fi
echo "  best-val reload line found in train.log"

# Sanity: checkpoint exists where we expect it
CKPT=weights/${EXP_NAME}_bestMAP.pt
if [ ! -f $CKPT ]; then
    echo "FATAL: expected checkpoint $CKPT not found."
    ls -la weights/
    exit 3
fi
echo "  checkpoint at: $CKPT ($(du -h $CKPT | awk '{print $1}'))"

# ── 4. test inference ─────────────────────────────────────────────────────
echo ""
echo "[4/6] test inference"
python -u scoring/infer_test.py \
    --config $CONFIG \
    --weights ${EXP_NAME}_bestMAP.pt \
    --out $PERSIST/predictions_sd1.csv \
    2>&1 | tee $PERSIST/inference.log

# ── 5. harness scoring ────────────────────────────────────────────────────
echo ""
echo "[5/6] harness scoring"
python -u scoring/score_model.py \
    --predictions $PERSIST/predictions_sd1.csv \
    --annotation /workspace/endoscapes/test/annotation_ds_coco.json \
    --name SwinCVS_baseline_sd1 \
    --out_dir $PERSIST \
    --provenance "Locked-recipe baseline seed 1: A100 80GB bs=4, 8 epochs, best-val reload from disk; first run with the fixed SwinCVS.py checkpoint-selection. Single seed (n=1 of 3 planned)." \
    2>&1 | tee $PERSIST/harness.log

# ── 6. persistence ────────────────────────────────────────────────────────
echo ""
echo "[6/6] persistence"
cp $CKPT $PERSIST/
cp results/${EXP_NAME}_results.json $PERSIST/training_journal.json
cp $CONFIG $PERSIST/config_used.yaml

# Code provenance:
#   _code_state.txt was generated locally pre-upload and captures git HEAD,
#   branch, dirty status, and the full `git diff HEAD` of uncommitted changes.
#   That is the authoritative record of which code ran (the local repo had
#   uncommitted edits at upload time: SwinCVS.py checkpoint-selection fix
#   plus new config + scoring/ scripts).
# We also compute a deterministic sha256 of the on-pod source tree as a
# secondary check that what got uploaded matches what we expect.
cp _code_state.txt $PERSIST/code_state_local.txt
( cd /workspace/SwinCVS && \
  tar --sort=name --mtime='2000-01-01' --owner=0 --group=0 --numeric-owner \
      -cf - \
      --exclude='weights' --exclude='results' --exclude='experiment_outputs' \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' . \
  | sha256sum | awk '{print $1}' \
) > $PERSIST/source_tree_sha256.txt
SRC_SHA=$(cat $PERSIST/source_tree_sha256.txt)
LOCAL_GIT_HEAD=$(grep '^git_head:' _code_state.txt | awk '{print $2}')
LOCAL_GIT_DIRTY=$(grep '^git_dirty:' _code_state.txt | awk '{print $2}')
echo "  local git HEAD: $LOCAL_GIT_HEAD (dirty=$LOCAL_GIT_DIRTY)"
echo "  pod source tree sha256: $SRC_SHA"

# Manifest
{
  echo "# Manifest — SwinCVS_baseline_sd1 (locked recipe, seed 1)"
  echo "Generated: $(date -u +%FT%TZ)"
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  echo "Source tree sha256: $SRC_SHA"
  echo "Training wall-clock: $((TRAIN_END - TRAIN_START))s"
  echo ""
  echo "## Files (size | sha256 | name)"
  for f in $PERSIST/*; do
    case "$f" in
      *artifacts.tar.gz|*manifest.txt) continue ;;
    esac
    [ -f "$f" ] || continue
    size=$(stat --printf='%s' "$f")
    sha=$(sha256sum "$f" | awk '{print $1}')
    printf "%12d %s  %s\n" "$size" "$sha" "$(basename "$f")"
  done
} > $PERSIST/manifest.txt
cat $PERSIST/manifest.txt

# Bundle
TAR=$PERSIST/artifacts.tar.gz
( cd $PERSIST && \
  tar -czf artifacts.tar.gz \
    --exclude=artifacts.tar.gz \
    $(ls | grep -v artifacts.tar.gz) )
echo ""
echo "wrote $TAR ($(du -h $TAR | awk '{print $1}'))"
sha256sum $TAR

echo ""
echo "============================================================"
echo "DONE. Pull from local machine with:"
echo "  scp runpod:$TAR ./swincvs_analysis/runpod_artifacts/baseline_sd1/"
echo "============================================================"
