#!/bin/bash
# ============================================================================
# CONTROLLED soft-vs-hard Stage-1 backbone experiment (overnight, gated, n=1).
#
# QUESTION: does soft-labelling the Stage-1 backbone beat hard-labelling it
# UNDER A MATCHED RECIPE (isolating soft labels from the LR change)?
#
# STEP 1  LR sweep: hard-label Stage-1 at ENCODER_LR in {1e-4, 5e-4}, short
#         (5 ep), seeds 91/92. Pick the LR with the best val mAP that clears a
#         learning threshold. ABORT (no controlled pair) if neither learns —
#         the locked 1e-5 is dead, do NOT run the real pair at a dead LR.
# STEP 2  Controlled pair at the picked LR, full 8 ep, SEED 1:
#           soft  (SOFT_TRAIN_LABELS=True)   -> SwinV2_backbone_IMNP_F1a_sd1
#           hard  (SOFT_TRAIN_LABELS=False)  -> SwinV2_backbone_IMNP_sd1
#         Identical everything except the labels. Infer single-frame test +
#         harness score (mAP + ECE) for BOTH backbones.
# STEP 3  GATE (report only, no auto-proceed): soft vs hard backbone delta.
#         Reference bars: hard control, no_augm_sd4 65.64, frozen 5-seed
#         65.81+-1.25, paper FROZEN 67.45 (the eventual Stage-2 bar).
#
# Sequential (clean control; Stage-1 only ~7.4GB so co-tenancy unnecessary).
# Persist + sha256 everything. Does NOT run frozen-Stage-2 or extra seeds.
# ============================================================================
set -uo pipefail
cd /workspace/SwinCVS
export DATASET_DIR=/workspace NUM_WORKERS=4 SWINCVS_AUTO=1 PYTHONUNBUFFERED=1

ROOT=/workspace/experiment_outputs/stage1_soft_expt
mkdir -p "$ROOT" weights results
TEMPLATE=config/SwinCVS_softstage1_sd1.yaml
ANNO=/workspace/endoscapes/test/annotation_ds_coco.json
THRESH=0.30                  # working-LR gate on best-val mAP (dead 1e-5 ~ base rate 0.15)
CHECK_EPOCHS=5
FULL_EPOCHS=8

ts(){ date -u +%FT%TZ; }
log(){ echo "[$(ts)] $*"; }
fcmp(){ awk "BEGIN{exit !($1)}"; }   # fcmp "$a > $b" -> exit 0 if true

gen_config(){ # <out> <seed> <lr> <epochs> <soft True|False>
  local tmp; tmp=$(mktemp)
  sed -e "s/^SEED:.*/SEED: $2/" \
      -e "s/^  EPOCHS:.*/  EPOCHS: $4/" \
      -e "s/^    ENCODER_LR:.*/    ENCODER_LR: $3/" \
      -e "s/^  SOFT_TRAIN_LABELS:.*/  SOFT_TRAIN_LABELS: $5/" \
      "$TEMPLATE" > "$tmp"
  mv "$tmp" "$1"
}
bestval(){ grep -oE "val mAP=[0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$"; }

run_train(){ # <tag> <seed> <lr> <epochs> <soft> ; writes $ROOT/<tag>/{train.log,config_used.yaml}
  local tag="$1" pdir="$ROOT/$1" cfg="config/_stage1_$1.yaml"
  mkdir -p "$pdir"
  gen_config "$cfg" "$2" "$3" "$4" "$5"
  cp "$cfg" "$pdir/config_used.yaml"
  log "TRAIN $tag (seed=$2 lr=$3 ep=$4 soft=$5)"
  python -u SwinCVS.py --config_path "$cfg" > "$pdir/train.log" 2>&1; local rc=$?
  log "  $tag done (exit $rc); best-val mAP=$(bestval "$pdir/train.log")"
}

infer_score(){ # <tag> <exp_name>
  local tag="$1" exp="$2" pdir="$ROOT/$1"
  local ckpt="weights/${exp}_bestMAP.pt"
  if [ ! -f "$ckpt" ]; then log "  WARN $tag: $ckpt missing — skip infer"; return 1; fi
  log "INFER+SCORE $tag ($exp)"
  python -u scoring/infer_test.py --config "config/_stage1_$1.yaml" --weights "${exp}_bestMAP.pt" \
      --split test --out "$pdir/preds_test.csv" > "$pdir/infer.log" 2>&1
  python -u scoring/score_model.py --predictions "$pdir/preds_test.csv" --annotation "$ANNO" \
      --name "$tag" --out_dir "$pdir" \
      --provenance "Stage-1 $tag single-frame test; $exp" > "$pdir/score.log" 2>&1
  cp "$ckpt" "$pdir/" 2>/dev/null || true
}
mean_map(){ grep -oE "Mean mAP: [0-9.]+" "$ROOT/$1/score.log" 2>/dev/null | grep -oE "[0-9.]+$"; }
mean_ece(){ grep -E "\*\*Mean\*\*" "$ROOT/$1"/metrics_*.md 2>/dev/null | sed -n '3p' | grep -oE "[0-9.]+" | head -1; }

sha_dir(){ # write per-file sha manifest for a run dir
  local pdir="$1"
  { echo "# manifest $(basename "$pdir")  $(ts)"
    for f in "$pdir"/*; do [ -f "$f" ] || continue; case "$f" in *manifest.txt) continue;; esac
      printf "%12d  %s  %s\n" "$(stat -c%s "$f")" "$(sha256sum "$f"|cut -d' ' -f1)" "$(basename "$f")"; done
  } > "$pdir/manifest.txt"
}

echo "============================================================"
echo "STAGE-1 SOFT-vs-HARD CONTROLLED EXPERIMENT — start $(ts)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q -r requirements.txt 2>&1 | tail -1

# ---- gate: memory probe ----
log "[probe] Stage-1 memory probe bs=4"
gen_config config/_stage1_probe.yaml 1 0.0001 8 False
python -u scoring/memory_probe.py --config config/_stage1_probe.yaml --steps 5 > "$ROOT/probe.log" 2>&1
if ! grep -q "PROBE PASS" "$ROOT/probe.log"; then log "FATAL probe failed"; tail -5 "$ROOT/probe.log"; exit 1; fi
log "  $(grep 'PROBE PASS' "$ROOT/probe.log")"

# ============================ STEP 1 — LR sweep ============================
echo ""; echo "===== STEP 1: LR sweep (hard, ${CHECK_EPOCHS} ep) ====="
run_train lrcheck_1em4 91 0.0001 "$CHECK_EPOCHS" False
run_train lrcheck_5em4 92 0.0005 "$CHECK_EPOCHS" False
BV1=$(bestval "$ROOT/lrcheck_1em4/train.log"); BV1=${BV1:-0}
BV2=$(bestval "$ROOT/lrcheck_5em4/train.log"); BV2=${BV2:-0}
log "STEP1: 1e-4 best-val=$BV1 | 5e-4 best-val=$BV2 (threshold $THRESH)"

if fcmp "$BV1 >= $BV2"; then PICK_LR=0.0001; PICK_BV=$BV1; else PICK_LR=0.0005; PICK_BV=$BV2; fi
if ! fcmp "$PICK_BV >= $THRESH"; then
  log "STEP1 ABORT: neither LR cleared $THRESH (dead). NOT running controlled pair."
  { echo "# STAGE-1 EXPERIMENT — ABORTED AT STEP 1  $(ts)"
    echo "LR sweep: 1e-4 best-val mAP=$BV1 ; 5e-4 best-val mAP=$BV2 ; threshold=$THRESH"
    echo "VERDICT: no working LR found in {1e-4,5e-4} within ${CHECK_EPOCHS} epochs."
    echo "Locked 1e-5 known dead. Next: widen LR grid / more epochs before the controlled pair."
  } > "$ROOT/REPORT.md"
  cat "$ROOT/REPORT.md"; echo "STAGE1_DONE_ABORTED"; exit 0
fi
log "STEP1 PICK: ENCODER_LR=$PICK_LR (best-val mAP=$PICK_BV)"

# ====================== STEP 2 — controlled pair (seed 1) ======================
echo ""; echo "===== STEP 2: controlled pair at LR=$PICK_LR, seed 1, ${FULL_EPOCHS} ep ====="
run_train soft_sd1 1 "$PICK_LR" "$FULL_EPOCHS" True
infer_score soft_sd1 SwinV2_backbone_IMNP_F1a_sd1
run_train hard_sd1 1 "$PICK_LR" "$FULL_EPOCHS" False
infer_score hard_sd1 SwinV2_backbone_IMNP_sd1

SOFT_MAP=$(mean_map soft_sd1); SOFT_ECE=$(mean_ece soft_sd1)
HARD_MAP=$(mean_map hard_sd1); HARD_ECE=$(mean_ece hard_sd1)
SOFT_MAP=${SOFT_MAP:-NA}; HARD_MAP=${HARD_MAP:-NA}; SOFT_ECE=${SOFT_ECE:-NA}; HARD_ECE=${HARD_ECE:-NA}

# ============================ STEP 3 — gate ============================
DELTA=NA; VERDICT="INCONCLUSIVE (a run failed — see logs)"
if [ "$SOFT_MAP" != "NA" ] && [ "$HARD_MAP" != "NA" ]; then
  DELTA=$(awk "BEGIN{printf \"%.2f\", $SOFT_MAP-$HARD_MAP}")
  if fcmp "$DELTA > 0.5"; then VERDICT="SOFT > HARD by ${DELTA}pp — soft labels help the REPRESENTATION (lever works). Frozen-Stage-2 + multi-seed justified NEXT session. (n=1; confirm vs seed variance.)"
  elif fcmp "$DELTA < -0.5"; then VERDICT="SOFT < HARD by ${DELTA}pp — soft labels HURT the representation at this recipe. Stop."
  else VERDICT="SOFT ~= HARD (delta ${DELTA}pp, |.|<=0.5) — soft labels do NOT help the representation. Stop."; fi
fi

for t in lrcheck_1em4 lrcheck_5em4 soft_sd1 hard_sd1; do sha_dir "$ROOT/$t" 2>/dev/null || true; done

{
echo "# STAGE-1 SOFT-vs-HARD CONTROLLED EXPERIMENT — REPORT  $(ts)"
echo ""
echo "## STEP 1 — working LR"
echo "- ENCODER_LR 1e-4: best-val mAP = $BV1"
echo "- ENCODER_LR 5e-4: best-val mAP = $BV2  (locked 1e-5 = dead, known)"
echo "- PICKED: ENCODER_LR = $PICK_LR (best-val mAP $PICK_BV), threshold $THRESH"
echo ""
echo "## STEP 2 — controlled pair (single-frame TEST, seed 1, matched recipe)"
echo "| Backbone | mean mAP | mean ECE |"
echo "|---|---|---|"
echo "| soft (SOFT_TRAIN_LABELS=True)  | ${SOFT_MAP} | ${SOFT_ECE} |"
echo "| hard (control)                 | ${HARD_MAP} | ${HARD_ECE} |"
echo "| delta (soft - hard) mAP        | ${DELTA} | |"
echo ""
echo "## Reference bars"
echo "- hard control (above) = the matched-recipe baseline"
echo "- no_augm_sd4 (prior Stage-1 backbone) = 65.64 single-frame"
echo "- frozen 5-seed (Stage-2, our harness) = 65.81 +- 1.25"
echo "- paper FROZEN (eventual Stage-2 bar) = 67.45 (5-seed)"
echo "  NOTE: these are single-frame BACKBONE numbers; the eventual frozen-Stage-2"
echo "  adds the LSTM (which added ~0 mAP on the frozen base). Beating 67.45 needs"
echo "  a strong backbone here AND that to survive Stage-2."
echo ""
echo "## STEP 3 — GATE VERDICT"
echo "$VERDICT"
} > "$ROOT/REPORT.md"

# top-level sha + bundle
( cd "$ROOT" && for t in lrcheck_1em4 lrcheck_5em4 soft_sd1 hard_sd1; do [ -d "$t" ] && tar -czf "$t.tar.gz" "$t"; done; sha256sum REPORT.md *.tar.gz > SHA256SUMS.txt 2>/dev/null )

echo ""; echo "============================================================"
cat "$ROOT/REPORT.md"
echo "============================================================"
echo "STAGE1_DONE_OK $(ts)"
