#!/bin/bash
# STEP 2 — differential-LR HARD-label Stage-1 re-sweep (after the build_optimizer fix).
# Head LR = CLASSIFIER_LR (1e-3, FIXED); backbone LR = ENCODER_LR, swept {1e-5,3e-5,1e-4}.
# HARD labels, single-frame, 10 ep, 3 runs CONCURRENT. TARGET best-val mAP ~0.55 (i.e.
# clearly out of the old 0.30 single-LR plateau). Picks the best backbone LR, scores it on
# single-frame TEST, reports. Does NOT run the soft pair — that is STEP 3 (gated on this).
set -uo pipefail
cd /workspace/SwinCVS
export DATASET_DIR=/workspace NUM_WORKERS=4 SWINCVS_AUTO=1 PYTHONUNBUFFERED=1
ROOT=/workspace/experiment_outputs/stage1_difflr
mkdir -p "$ROOT" weights results
TEMPLATE=config/SwinCVS_softstage1_sd1.yaml
ANNO=/workspace/endoscapes/test/annotation_ds_coco.json
EPOCHS=10
ts(){ date -u +%FT%TZ; }
log(){ echo "[$(ts)] $*"; }
fcmp(){ awk "BEGIN{exit !($1)}"; }
gen_config(){ local tmp; tmp=$(mktemp); sed -e "s/^SEED:.*/SEED: $2/" -e "s/^  EPOCHS:.*/  EPOCHS: $4/" \
    -e "s/^    ENCODER_LR:.*/    ENCODER_LR: $3/" -e "s/^  SOFT_TRAIN_LABELS:.*/  SOFT_TRAIN_LABELS: $5/" "$TEMPLATE" > "$tmp"; mv "$tmp" "$1"; }
bestval(){ grep -oE "val mAP=[0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$"; }
bestepoch(){ grep -oE "\(epoch [0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+$"; }
traj(){ grep -E "^Epoch:|^mAP " "$1" 2>/dev/null | paste - - | sed 's/\t/  /'; }

echo "=== STAGE-1 DIFF-LR SWEEP (head 1e-3 fixed; backbone swept) start $(ts) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q -r requirements.txt 2>&1 | tail -1

# --- probe the NEW optimizer (2 param groups) ---
gen_config config/_difflr_probe.yaml 1 0.00001 10 False
python -u scoring/memory_probe.py --config config/_difflr_probe.yaml --steps 5 > "$ROOT/probe.log" 2>&1
grep -q "PROBE PASS" "$ROOT/probe.log" || { log "FATAL probe"; tail -8 "$ROOT/probe.log"; exit 1; }
log "probe OK: $(grep 'PROBE PASS' "$ROOT/probe.log")"

TAGS="bb1e5 bb3e5 bb1e4"
declare -A LR=( [bb1e5]=0.00001 [bb3e5]=0.00003 [bb1e4]=0.0001 )
declare -A SD=( [bb1e5]=11 [bb3e5]=12 [bb1e4]=13 )

# --- SEQUENTIAL backbone-LR checks (hard, single-frame) — solo = full GPU, first
#     result lands fast (~75 min) so the diff-LR question is answered early. ---
for tag in $TAGS; do
  mkdir -p "$ROOT/$tag"; cfg=config/_difflr_$tag.yaml
  gen_config "$cfg" "${SD[$tag]}" "${LR[$tag]}" "$EPOCHS" False
  cp "$cfg" "$ROOT/$tag/config_used.yaml"
  log "TRAIN $tag (sequential)  backbone_lr=${LR[$tag]} head_lr=1e-3 seed=${SD[$tag]}"
  python -u SwinCVS.py --config_path "$cfg" > "$ROOT/$tag/train.log" 2>&1
  log "  $tag done: best-val=$(bestval "$ROOT/$tag/train.log") ep=$(bestepoch "$ROOT/$tag/train.log")"
done
log "all 3 sweeps done"

# --- parse + pick best ---
PICK=""; PICK_BV=0
for tag in $TAGS; do
  bv=$(bestval "$ROOT/$tag/train.log"); bv=${bv:-0}; be=$(bestepoch "$ROOT/$tag/train.log")
  log "  $tag (bb_lr=${LR[$tag]}): best-val=$bv ep=${be:-?}"
  if fcmp "$bv > $PICK_BV"; then PICK=$tag; PICK_BV=$bv; fi
done
log "WINNER: $PICK (backbone_lr=${LR[$PICK]}) best-val=$PICK_BV"

# --- score winner single-frame TEST ---
WIN_EXP=SwinV2_backbone_IMNP_sd${SD[$PICK]}
WIN_TEST=NA
if [ -f "weights/${WIN_EXP}_bestMAP.pt" ]; then
  python -u scoring/infer_test.py --config config/_difflr_$PICK.yaml --weights ${WIN_EXP}_bestMAP.pt \
      --split test --out "$ROOT/$PICK/preds_test.csv" > "$ROOT/$PICK/infer.log" 2>&1
  python -u scoring/score_model.py --predictions "$ROOT/$PICK/preds_test.csv" --annotation "$ANNO" \
      --name "$PICK" --out_dir "$ROOT/$PICK" --provenance "diff-LR hard Stage-1 winner $PICK ($WIN_EXP)" > "$ROOT/$PICK/score.log" 2>&1
  cp "weights/${WIN_EXP}_bestMAP.pt" "$ROOT/$PICK/" 2>/dev/null || true
  WIN_TEST=$(grep -oE "Mean mAP: [0-9.]+" "$ROOT/$PICK/score.log" 2>/dev/null | grep -oE "[0-9.]+$")
fi

VERD="UNKNOWN"
if   fcmp "$PICK_BV >= 0.50"; then VERD="SUCCESS — diff-LR reproduces the ~0.55 ballpark (best-val $PICK_BV), clearly out of the 0.30 plateau. STEP 3 (controlled soft-vs-hard pair) justified."
elif fcmp "$PICK_BV >= 0.40"; then VERD="PARTIAL — best-val $PICK_BV: above the 0.30 plateau but short of ~0.55. Consider more epochs / a 4th LR before the pair."
else VERD="FAIL — best-val $PICK_BV: still near the 0.30 plateau. Diff-LR did NOT restore trainability; soft comparison still blocked."; fi

for tag in $TAGS; do { echo "# manifest $tag $(ts)"; for f in "$ROOT/$tag"/*; do [ -f "$f" ] || continue; case "$f" in *manifest.txt) continue;; esac
  printf "%12d  %s  %s\n" "$(stat -c%s "$f")" "$(sha256sum "$f"|cut -d' ' -f1)" "$(basename "$f")"; done; } > "$ROOT/$tag/manifest.txt"; done

{
echo "# STAGE-1 DIFF-LR HARD-LABEL RE-SWEEP — REPORT $(ts)"; echo ""
echo "build_optimizer fix: differential LRs — head(model.head)=CLASSIFIER_LR 1e-3 (fixed),"
echo "backbone(rest of SwinV2)=ENCODER_LR (swept). HARD labels, single-frame, ${EPOCHS} ep."
echo ""
echo "| backbone LR | best-val mAP | best epoch |"
echo "|---|---|---|"
for tag in $TAGS; do echo "| ${LR[$tag]} | $(bestval "$ROOT/$tag/train.log") | $(bestepoch "$ROOT/$tag/train.log") |"; done
echo ""
echo "Trajectories (epoch -> val mAP):"
for tag in $TAGS; do echo "- $tag (bb_lr=${LR[$tag]}):"; traj "$ROOT/$tag/train.log" | sed 's/^/    /'; done
echo ""
echo "WINNER: backbone_lr=${LR[$PICK]}  best-val mAP=$PICK_BV (fraction)  single-frame TEST mAP=${WIN_TEST:-NA}%"
echo "Reference: no_augm_sd4 ~0.55 val / 65.64% test ; old broken single-LR plateau = 0.30 val"
echo ""
echo "VERDICT: $VERD"
} > "$ROOT/REPORT.md"
( cd "$ROOT" && for tag in $TAGS; do tar -czf "$tag.tar.gz" "$tag"; done; sha256sum REPORT.md *.tar.gz > SHA256SUMS.txt 2>/dev/null )
echo "============================================================"; cat "$ROOT/REPORT.md"
echo "============================================================"; echo "DIFFLR_SWEEP_DONE $(ts)"
