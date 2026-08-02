#!/bin/bash
# ============================================================================
# CONTROLLED soft-vs-hard Stage-1 experiment — v2 (refined LR sweep).
# v1 found {1e-4 -> 0.25 too-slow, 5e-4 -> 0.30 THRASHING (best@epoch1 fluke)}.
# v2 sweeps the intermediate band {2e-4, 3e-4} and uses a STABILITY-AWARE pick:
# an LR qualifies only if it GENUINELY IMPROVED (best epoch >= 2) and best-val
# mAP >= MINBV. Picks the highest-best-val qualifier. If NONE qualify -> STOP
# with a diagnostic (do NOT run the controlled pair at a non-working LR — the
# single-LR head+backbone tension / dead LR-scheduler likely needs a code fix).
# STEP 2 (only if a clean LR): soft vs hard, seed 1, 8 ep, infer+score both.
# STEP 3: gate verdict. Persist + sha. Full LR landscape (1e-4..5e-4) in report.
# ============================================================================
set -uo pipefail
cd /workspace/SwinCVS
export DATASET_DIR=/workspace NUM_WORKERS=4 SWINCVS_AUTO=1 PYTHONUNBUFFERED=1

ROOT=/workspace/experiment_outputs/stage1_soft_expt
mkdir -p "$ROOT" weights results
TEMPLATE=config/SwinCVS_softstage1_sd1.yaml
ANNO=/workspace/endoscapes/test/annotation_ds_coco.json
MINBV=0.30                   # min best-val mAP to call an LR "working"
CHECK_EPOCHS=5
FULL_EPOCHS=8

ts(){ date -u +%FT%TZ; }
log(){ echo "[$(ts)] $*"; }
fcmp(){ awk "BEGIN{exit !($1)}"; }

gen_config(){ # <out> <seed> <lr> <epochs> <soft>
  local tmp; tmp=$(mktemp)
  sed -e "s/^SEED:.*/SEED: $2/" -e "s/^  EPOCHS:.*/  EPOCHS: $4/" \
      -e "s/^    ENCODER_LR:.*/    ENCODER_LR: $3/" \
      -e "s/^  SOFT_TRAIN_LABELS:.*/  SOFT_TRAIN_LABELS: $5/" "$TEMPLATE" > "$tmp"
  mv "$tmp" "$1"
}
bestval(){   grep -oE "val mAP=[0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$"; }
bestepoch(){ grep -oE "\(epoch [0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+$"; }
traj(){      grep -E "^Epoch:|^mAP " "$1" 2>/dev/null | paste - - | sed 's/\t/  /'; }  # per-epoch lines

run_train(){ # <tag> <seed> <lr> <epochs> <soft>
  local tag="$1" pdir="$ROOT/$1" cfg="config/_stage1_$1.yaml"
  mkdir -p "$pdir"; gen_config "$cfg" "$2" "$3" "$4" "$5"; cp "$cfg" "$pdir/config_used.yaml"
  log "TRAIN $tag (seed=$2 lr=$3 ep=$4 soft=$5)"
  python -u SwinCVS.py --config_path "$cfg" > "$pdir/train.log" 2>&1; local rc=$?
  log "  $tag done (exit $rc) best-val=$(bestval "$pdir/train.log") best-epoch=$(bestepoch "$pdir/train.log")"
}
infer_score(){ # <tag> <exp>
  local tag="$1" exp="$2" pdir="$ROOT/$1" ckpt="weights/$2_bestMAP.pt"
  [ -f "$ckpt" ] || { log "  WARN $tag: $ckpt missing"; return 1; }
  log "INFER+SCORE $tag ($exp)"
  python -u scoring/infer_test.py --config "config/_stage1_$1.yaml" --weights "$2_bestMAP.pt" \
      --split test --out "$pdir/preds_test.csv" > "$pdir/infer.log" 2>&1
  python -u scoring/score_model.py --predictions "$pdir/preds_test.csv" --annotation "$ANNO" \
      --name "$tag" --out_dir "$pdir" --provenance "Stage-1 $tag single-frame test; $exp" > "$pdir/score.log" 2>&1
  cp "$ckpt" "$pdir/" 2>/dev/null || true
}
mean_map(){ grep -oE "Mean mAP: [0-9.]+" "$ROOT/$1/score.log" 2>/dev/null | grep -oE "[0-9.]+$"; }
mean_ece(){ grep -E "\*\*Mean\*\*" "$ROOT/$1"/metrics_*.md 2>/dev/null | sed -n '3p' | grep -oE "[0-9.]+" | head -1; }

echo "============================================================"
echo "STAGE-1 SOFT-vs-HARD v2 (refined LR sweep) — start $(ts)"
echo "============================================================"
pip install -q -r requirements.txt 2>&1 | tail -1

# ---------- STEP 1: refined LR sweep ----------
echo ""; echo "===== STEP 1b: refined LR sweep {2e-4, 3e-4}, ${CHECK_EPOCHS} ep hard ====="
run_train lrcheck_2em4 94 0.0002 "$CHECK_EPOCHS" False
run_train lrcheck_3em4 95 0.0003 "$CHECK_EPOCHS" False

# prior v1 results (kept on disk)
BV_1em4=$(bestval "$ROOT/lrcheck_1em4/train.log"); BE_1em4=$(bestepoch "$ROOT/lrcheck_1em4/train.log")
BV_5em4=$(bestval "$ROOT/lrcheck_5em4/train.log"); BE_5em4=$(bestepoch "$ROOT/lrcheck_5em4/train.log")
BV_2em4=$(bestval "$ROOT/lrcheck_2em4/train.log"); BE_2em4=$(bestepoch "$ROOT/lrcheck_2em4/train.log")
BV_3em4=$(bestval "$ROOT/lrcheck_3em4/train.log"); BE_3em4=$(bestepoch "$ROOT/lrcheck_3em4/train.log")
for v in BV_1em4 BV_5em4 BV_2em4 BV_3em4 BE_1em4 BE_5em4 BE_2em4 BE_3em4; do
  [ -z "${!v:-}" ] && eval "$v=0"; done

log "LR landscape: 1e-4=$BV_1em4(ep$BE_1em4) 2e-4=$BV_2em4(ep$BE_2em4) 3e-4=$BV_3em4(ep$BE_3em4) 5e-4=$BV_5em4(ep$BE_5em4)"

# stability-aware pick among the new {2e-4,3e-4}: qualify if best-epoch>=2 AND best-val>=MINBV
PICK_LR=""; PICK_TAG=""; PICK_BV=0
for cand in "0.0002:$BV_2em4:$BE_2em4:lrcheck_2em4" "0.0003:$BV_3em4:$BE_3em4:lrcheck_3em4"; do
  IFS=: read -r lr bv be tag <<< "$cand"
  if fcmp "$be >= 2" && fcmp "$bv >= $MINBV" && fcmp "$bv > $PICK_BV"; then
    PICK_LR=$lr; PICK_BV=$bv; PICK_TAG=$tag
  fi
done

write_landscape(){
  echo "## STEP 1 — LR landscape (single-frame Stage-1, hard labels, best-val mAP / best epoch)"
  echo "| ENCODER_LR | best-val mAP | best epoch | note |"
  echo "|---|---|---|---|"
  echo "| 1e-4 | $BV_1em4 | $BE_1em4 | too slow |"
  echo "| 2e-4 | $BV_2em4 | $BE_2em4 | |"
  echo "| 3e-4 | $BV_3em4 | $BE_3em4 | |"
  echo "| 5e-4 | $BV_5em4 | $BE_5em4 | THRASHING (best@ep1 fluke) |"
  echo ""
  echo "Trajectories (epoch -> val mAP):"
  for t in lrcheck_1em4 lrcheck_2em4 lrcheck_3em4 lrcheck_5em4; do
    echo "- $t:"; traj "$ROOT/$t/train.log" | sed 's/^/    /'
  done
}

if [ -z "$PICK_LR" ]; then
  log "STEP1 STOP: no LR in {2e-4,3e-4} genuinely learned (best-epoch>=2 & best-val>=$MINBV)."
  { echo "# STAGE-1 SOFT-vs-HARD — STOPPED AT STEP 1 (no clean working LR)  $(ts)"; echo ""
    write_landscape; echo ""
    echo "## VERDICT"
    echo "No fixed ENCODER_LR in {1e-4,2e-4,3e-4,5e-4} cleanly trains the single-frame"
    echo "Stage-1 backbone (all peak ~0.25-0.30 val, far below no_augm_sd4 ~0.55+)."
    echo "ROOT CAUSE (likely): build_optimizer puts the WHOLE bare backbone on one LR"
    echo "(ENCODER_LR), so the fresh head wants ~1e-3 while the pretrained backbone wants"
    echo "~1e-5 — no single fixed LR serves both, AND the LR-scheduler/warmup is DEAD in"
    echo "this codebase (build_scheduler never imported). no_augm_sd4 was likely trained"
    echo "with a working schedule or differential LRs. The matched soft-vs-hard pair was"
    echo "NOT run (would be noise at this regime). NEXT: differential LR (head>backbone)"
    echo "or wire up warmup+cosine, then re-sweep — needs a code change + human sign-off."
  } > "$ROOT/REPORT.md"
  cat "$ROOT/REPORT.md"; echo "STAGE1_DONE_STOPPED $(ts)"; exit 0
fi
log "STEP1 PICK: ENCODER_LR=$PICK_LR ($PICK_TAG, best-val=$PICK_BV, best-epoch ok)"

# ---------- STEP 2: controlled pair at picked LR ----------
echo ""; echo "===== STEP 2: controlled pair LR=$PICK_LR seed 1 ${FULL_EPOCHS} ep ====="
run_train soft_sd1 1 "$PICK_LR" "$FULL_EPOCHS" True
infer_score soft_sd1 SwinV2_backbone_IMNP_F1a_sd1
run_train hard_sd1 1 "$PICK_LR" "$FULL_EPOCHS" False
infer_score hard_sd1 SwinV2_backbone_IMNP_sd1

SOFT_MAP=$(mean_map soft_sd1); SOFT_ECE=$(mean_ece soft_sd1)
HARD_MAP=$(mean_map hard_sd1); HARD_ECE=$(mean_ece hard_sd1)
SOFT_MAP=${SOFT_MAP:-NA}; HARD_MAP=${HARD_MAP:-NA}; SOFT_ECE=${SOFT_ECE:-NA}; HARD_ECE=${HARD_ECE:-NA}

DELTA=NA; VERDICT="INCONCLUSIVE (a run failed — see logs)"
if [ "$SOFT_MAP" != NA ] && [ "$HARD_MAP" != NA ]; then
  DELTA=$(awk "BEGIN{printf \"%.2f\", $SOFT_MAP-$HARD_MAP}")
  if fcmp "$DELTA > 0.5";   then VERDICT="SOFT > HARD by ${DELTA}pp — soft labels help the REPRESENTATION (lever works). Frozen-Stage-2 + multi-seed justified NEXT session. (n=1; confirm vs seed variance.)"
  elif fcmp "$DELTA < -0.5"; then VERDICT="SOFT < HARD by ${DELTA}pp — soft HURTS the representation at this recipe. Stop."
  else VERDICT="SOFT ~= HARD (delta ${DELTA}pp) — soft labels do NOT help the representation. Stop."; fi
fi

for t in lrcheck_1em4 lrcheck_2em4 lrcheck_3em4 lrcheck_5em4 soft_sd1 hard_sd1; do
  [ -d "$ROOT/$t" ] || continue
  { echo "# manifest $t $(ts)"; for f in "$ROOT/$t"/*; do [ -f "$f" ] || continue; case "$f" in *manifest.txt) continue;; esac
    printf "%12d  %s  %s\n" "$(stat -c%s "$f")" "$(sha256sum "$f"|cut -d' ' -f1)" "$(basename "$f")"; done; } > "$ROOT/$t/manifest.txt"
done

{
echo "# STAGE-1 SOFT-vs-HARD CONTROLLED EXPERIMENT — REPORT  $(ts)"; echo ""
write_landscape; echo ""
echo "## STEP 1 PICK: ENCODER_LR = $PICK_LR (best-val $PICK_BV, genuine improvement)"
echo ""
echo "## STEP 2 — controlled pair (single-frame TEST, seed 1, matched recipe, ${FULL_EPOCHS} ep)"
echo "| Backbone | mean mAP | mean ECE |"
echo "|---|---|---|"
echo "| soft (SOFT_TRAIN_LABELS=True) | ${SOFT_MAP} | ${SOFT_ECE} |"
echo "| hard (control)               | ${HARD_MAP} | ${HARD_ECE} |"
echo "| delta (soft - hard) mAP      | ${DELTA} | |"
echo ""
echo "## Reference bars (context; these are single-frame BACKBONE numbers)"
echo "- no_augm_sd4 prior Stage-1 backbone = 65.64 single-frame"
echo "- frozen 5-seed Stage-2 (our harness) = 65.81 +- 1.25"
echo "- paper FROZEN (eventual Stage-2 bar) = 67.45 (5-seed)"
echo "  NB: eventual frozen-Stage-2 adds the LSTM (~0 mAP on the frozen base). If the"
echo "  backbone here is far below 65.64, beating 67.45 via this path is unlikely without"
echo "  recovering a stronger Stage-1 recipe."
echo ""
echo "## STEP 3 — GATE VERDICT"
echo "$VERDICT"
} > "$ROOT/REPORT.md"

( cd "$ROOT" && for t in lrcheck_1em4 lrcheck_2em4 lrcheck_3em4 lrcheck_5em4 soft_sd1 hard_sd1; do [ -d "$t" ] && tar -czf "$t.tar.gz" "$t"; done; sha256sum REPORT.md *.tar.gz > SHA256SUMS.txt 2>/dev/null )

echo ""; echo "============================================================"; cat "$ROOT/REPORT.md"
echo "============================================================"; echo "STAGE1_DONE_OK $(ts)"
