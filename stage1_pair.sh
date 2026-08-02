#!/bin/bash
# ============================================================================
# STEP 3+4 — CONTROLLED soft-vs-hard Stage-1 pair at the LOCKED recipe.
# Recipe frozen from STEP 2: backbone(ENCODER_LR)=1e-5, head(CLASSIFIER_LR)=1e-3,
# single-frame, 8 ep, seed 1. hard and soft differ ONLY in SOFT_TRAIN_LABELS
# (verified by an explicit config diff). Trains from /dev/shm (RAM) -> reliable.
# Scores BOTH backbones single-frame TEST (mAP + ECE). GATE: soft vs hard.
# ============================================================================
set -uo pipefail
cd /workspace/SwinCVS
export DATASET_DIR=/dev/shm NUM_WORKERS=4 SWINCVS_AUTO=1 PYTHONUNBUFFERED=1
ROOT=/workspace/experiment_outputs/stage1_pair
mkdir -p "$ROOT" weights results
TEMPLATE=config/SwinCVS_softstage1_sd1.yaml
ANNO=/dev/shm/endoscapes/test/annotation_ds_coco.json
BB_LR=0.00001        # locked backbone LR; head stays CLASSIFIER_LR=1e-3
EPOCHS=8
ts(){ date -u +%FT%TZ; }; log(){ echo "[$(ts)] $*"; }; fcmp(){ awk "BEGIN{exit !($1)}"; }
gen_config(){ # <out> <seed> <bb_lr> <epochs> <soft>
  local tmp; tmp=$(mktemp)
  sed -e "s/^SEED:.*/SEED: $2/" -e "s/^  EPOCHS:.*/  EPOCHS: $4/" \
      -e "s/^    ENCODER_LR:.*/    ENCODER_LR: $3/" \
      -e "s/^  SOFT_TRAIN_LABELS:.*/  SOFT_TRAIN_LABELS: $5/" \
      -e "s|^DATASET_DIR:.*|DATASET_DIR: '/dev/shm'|" "$TEMPLATE" > "$tmp"; mv "$tmp" "$1"; }
bestval(){ grep -oE "val mAP=[0-9.]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$"; }
mean_map(){ grep -oE "Mean mAP: [0-9.]+" "$ROOT/$1/score.log" 2>/dev/null | grep -oE "[0-9.]+$"; }
mean_ece(){ grep -E "\*\*Mean\*\*" "$ROOT/$1"/metrics_*.md 2>/dev/null | sed -n '3p' | grep -oE "[0-9.]+" | head -1; }
per_crit_map(){ grep -E "Mean mAP|C[123] mAP" "$ROOT/$1/score.log" 2>/dev/null; }

run(){ # <tag> <soft> <exp>
  local tag="$1" pdir="$ROOT/$1" cfg="config/_pair_$1.yaml"
  mkdir -p "$pdir"; gen_config "$cfg" 1 "$BB_LR" "$EPOCHS" "$2"; cp "$cfg" "$pdir/config_used.yaml"
  log "TRAIN $tag (soft=$2) bb_lr=$BB_LR head_lr=1e-3 seed=1 (RAM)"
  python -u SwinCVS.py --config_path "$cfg" > "$pdir/train.log" 2>&1; local rc=$?
  log "  $tag done (exit $rc) best-val=$(bestval "$pdir/train.log")"
  local ckpt="weights/$3_bestMAP.pt"
  if [ -f "$ckpt" ]; then
    log "  INFER+SCORE $tag ($3)"
    python -u scoring/infer_test.py --config "$cfg" --weights "$3_bestMAP.pt" --split test \
        --out "$pdir/preds_test.csv" > "$pdir/infer.log" 2>&1
    python -u scoring/score_model.py --predictions "$pdir/preds_test.csv" --annotation "$ANNO" \
        --name "$tag" --out_dir "$pdir" --provenance "Stage-1 $tag single-frame test (RAM); $3" > "$pdir/score.log" 2>&1
    cp "$ckpt" "$pdir/" 2>/dev/null || true
  else log "  WARN $tag: $ckpt missing — train may have crashed"; fi
}

echo "============================================================"
echo "STAGE-1 CONTROLLED PAIR (RAM, locked recipe) — start $(ts)"
echo "============================================================"
pip install -q -r requirements.txt 2>&1 | tail -1

# explicit confirmation: the two configs differ ONLY in SOFT_TRAIN_LABELS
gen_config config/_pair_hard.yaml 1 $BB_LR $EPOCHS False
gen_config config/_pair_soft.yaml 1 $BB_LR $EPOCHS True
echo "=== CONFIG DIFF (hard vs soft) — MUST be ONLY SOFT_TRAIN_LABELS ==="
diff config/_pair_hard.yaml config/_pair_soft.yaml | tee "$ROOT/config_diff.txt"
echo "==="

run hard False SwinV2_backbone_IMNP_sd1
run soft True  SwinV2_backbone_IMNP_F1a_sd1

HM=$(mean_map hard); HE=$(mean_ece hard); SM=$(mean_map soft); SE=$(mean_ece soft)
HM=${HM:-NA}; SM=${SM:-NA}; HE=${HE:-NA}; SE=${SE:-NA}
DELTA=NA; VERD="INCONCLUSIVE (a run failed — see logs)"
if [ "$SM" != NA ] && [ "$HM" != NA ]; then
  DELTA=$(awk "BEGIN{printf \"%.2f\", $SM-$HM}")
  if   fcmp "$DELTA > 0.5";  then VERD="SOFT > HARD by ${DELTA}pp — soft labels help the REPRESENTATION (lever works). Frozen-Stage-2 + multi-seed + SOTA-vs-67.45 justified NEXT (report for sign-off). n=1, confirm vs seed variance."
  elif fcmp "$DELTA < -0.5"; then VERD="SOFT < HARD by ${DELTA}pp — soft HURTS the representation at matched recipe. Stop."
  else VERD="SOFT ~= HARD (delta ${DELTA}pp, within noise) — soft labels do NOT help the representation. STOP: no downstream run clears 67.45 via this lever."; fi
fi

for t in hard soft; do [ -d "$ROOT/$t" ] || continue
  { echo "# manifest $t $(ts)"; for f in "$ROOT/$t"/*; do [ -f "$f" ] || continue; case "$f" in *manifest.txt) continue;; esac
    printf "%12d  %s  %s\n" "$(stat -c%s "$f")" "$(sha256sum "$f"|cut -d' ' -f1)" "$(basename "$f")"; done; } > "$ROOT/$t/manifest.txt"; done

{
echo "# STAGE-1 CONTROLLED soft-vs-hard PAIR — REPORT $(ts)"; echo ""
echo "Locked recipe (frozen from STEP 2, IDENTICAL both arms except labels):"
echo "  backbone(ENCODER_LR)=1e-5, head(CLASSIFIER_LR)=1e-3, single-frame, ${EPOCHS} ep, seed 1, /dev/shm."
echo ""
echo "Config diff (hard vs soft) — the ONLY difference:"; sed 's/^/    /' "$ROOT/config_diff.txt"; echo ""
echo "## STEP 3 — backbone comparison (single-frame TEST)"
echo "| arm | best-val mAP | mean test mAP | mean ECE |"
echo "|---|---|---|---|"
echo "| hard (control) | $(bestval "$ROOT/hard/train.log") | ${HM} | ${HE} |"
echo "| soft           | $(bestval "$ROOT/soft/train.log") | ${SM} | ${SE} |"
echo "| delta (soft-hard) mAP | | ${DELTA} | |"
echo ""
echo "Per-criterion test mAP:"
echo "- hard:"; per_crit_map hard | sed 's/^/    /'
echo "- soft:"; per_crit_map soft | sed 's/^/    /'
echo ""
echo "## Reference bars (single-frame BACKBONE numbers)"
echo "- no_augm_sd4 = 65.64 test ; frozen 5-seed Stage-2 = 65.81+-1.25 ; paper FROZEN = 67.45 (eventual Stage-2 bar)"
echo ""
echo "## STEP 4 — GATE VERDICT"
echo "$VERD"
} > "$ROOT/REPORT.md"
( cd "$ROOT" && for t in hard soft; do [ -d "$t" ] && tar -czf "$t.tar.gz" "$t"; done; sha256sum REPORT.md config_diff.txt *.tar.gz > SHA256SUMS.txt 2>/dev/null )
echo "============================================================"; cat "$ROOT/REPORT.md"
echo "============================================================"; echo "PAIR_DONE $(ts)"
