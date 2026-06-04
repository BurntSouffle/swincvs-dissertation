"""Shared scoring harness for SwinCVS variants.

Inputs:
  --predictions   path to a predictions CSV with columns
                    video_id, frame, c1_pred, c2_pred, c3_pred,
                    c1_label, c2_label, c3_label
                  (MV labels = round(ds) per criterion; matches
                  scripts/f_dataset.py:get_dataframe).
  --name          model identifier used in output filenames.
  --annotation    test-split annotation JSON containing per-frame `ds`
                  (defaults to endoscapes/test/annotation_ds_coco.json).
  --provenance    free-text note recorded under "provenance" in the JSON
                  output (use this to flag e.g. "historical reference,
                  n=1; not the locked-recipe baseline").

The harness intentionally does NOT run inference. To score a new
checkpoint, generate a predictions CSV first (run_inference.py is the
reference implementation) and then point this harness at it. That keeps
the harness free of model/CUDA/config dependencies and identical across
machines.

Metrics (per criterion + mean):
  - mAP
  - Balanced accuracy @ threshold 0.5
  - ECE with 10 equal-width bins over [0, 1]
  - High-agreement-subset mAP (ds == 0 or 1 for that criterion;
    subset size is criterion-specific and reported)
  - Positive emission rate (% pred >= 0.5) vs ground-truth MV positive
    rate

CIs: percentile bootstrap, 1000 resamples, fixed seed 42, 95% (2.5/97.5).
Resamples that yield an undefined metric (e.g. AP with one class) are
skipped; the JSON records how many were dropped per metric.

Output:
  swincvs_analysis/harness/metrics_{name}.json
  swincvs_analysis/harness/metrics_{name}.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = ROOT / "swincvs_analysis" / "predictions.csv"
DEFAULT_ANNOTATION = ROOT / "endoscapes" / "test" / "annotation_ds_coco.json"
OUT_DIR = ROOT / "swincvs_analysis" / "harness"

CRITERIA = ("C1", "C2", "C3")
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
ECE_BINS = 10
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"video_id", "frame",
                "c1_pred", "c2_pred", "c3_pred",
                "c1_label", "c2_label", "c3_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["video_id"] = df["video_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    return df


def load_ds_table(annotation_path: Path) -> pd.DataFrame:
    """Per-frame `ds` (per-criterion mean over 3 annotators)."""
    with open(annotation_path) as f:
        ann = json.load(f)
    rows = []
    for img in ann["images"]:
        stem = img["file_name"].rsplit(".", 1)[0]
        vid, frame = stem.split("_", 1)
        rows.append({
            "video_id": int(vid),
            "frame": int(frame),
            "ds_c1": float(img["ds"][0]),
            "ds_c2": float(img["ds"][1]),
            "ds_c3": float(img["ds"][2]),
        })
    return pd.DataFrame(rows)


def join_predictions_with_ds(preds: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    merged = preds.merge(ds, on=["video_id", "frame"], how="left",
                         validate="many_to_one")
    nan_rows = merged[["ds_c1", "ds_c2", "ds_c3"]].isna().any(axis=1).sum()
    if nan_rows:
        raise ValueError(
            f"{nan_rows}/{len(merged)} prediction rows did not match a "
            f"frame in the annotation JSON. Check the annotation path."
        )
    return merged

# ---------------------------------------------------------------------------
# Metric primitives (vectorised, called many times under bootstrap)
# ---------------------------------------------------------------------------

def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(average_precision_score(y_true, y_score))


def safe_balanced_accuracy(y_true: np.ndarray, y_pred01: np.ndarray) -> float:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan
    return float(balanced_accuracy_score(y_true, y_pred01))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Equal-width binning on [0, 1]; bin edges [i/n, (i+1)/n] for i=0..n-1.
    A probability of exactly 1.0 is assigned to the top bin (right-closed
    handling). Empty bins contribute 0.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins[1:-1], right=False), 0, n_bins - 1)
    N = len(probs)
    if N == 0:
        return float("nan")
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc_b = float(labels[mask].mean())
        conf_b = float(probs[mask].mean())
        ece += (n_b / N) * abs(acc_b - conf_b)
    return float(ece)


def positive_emission_rate(probs: np.ndarray) -> float:
    if len(probs) == 0:
        return float("nan")
    return float((probs >= THRESHOLD).mean())


def positive_truth_rate(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    return float(labels.mean())

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(fn: Callable[[np.ndarray], float],
                 indices: np.ndarray,
                 n: int = BOOTSTRAP_N,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    """Generic resampling: `fn(resampled_indices)` returns the metric on a
    bootstrap sample. Returns point estimate + 95% percentile CI.
    """
    point = fn(indices)
    rng = np.random.default_rng(seed)
    N = len(indices)
    samples = []
    dropped = 0
    for _ in range(n):
        boot_idx = rng.integers(0, N, size=N)
        resampled = indices[boot_idx]
        val = fn(resampled)
        if not np.isfinite(val):
            dropped += 1
            continue
        samples.append(val)
    if len(samples) < 10:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_bootstrap_valid": len(samples), "n_bootstrap_dropped": dropped}
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi,
            "n_bootstrap_valid": len(samples),
            "n_bootstrap_dropped": dropped}

# ---------------------------------------------------------------------------
# Per-criterion metric panel
# ---------------------------------------------------------------------------

def per_criterion_panel(probs: np.ndarray,
                        labels: np.ndarray,
                        high_agree_mask: np.ndarray) -> dict:
    """All metrics for a single criterion. probs/labels are 1-D arrays of
    length N; high_agree_mask is a bool array of the same length.
    """
    N = len(probs)
    all_idx = np.arange(N)
    high_idx = np.where(high_agree_mask)[0]

    preds01 = (probs >= THRESHOLD).astype(int)

    # mAP / balanced acc / ECE / emission on full set
    def _ap(idx):
        return safe_average_precision(labels[idx], probs[idx])

    def _bacc(idx):
        return safe_balanced_accuracy(labels[idx], preds01[idx])

    def _ece(idx):
        return compute_ece(probs[idx], labels[idx])

    def _emit(idx):
        return positive_emission_rate(probs[idx])

    def _truth(idx):
        return positive_truth_rate(labels[idx])

    def _ap_high(idx):
        # restrict to high-agreement subset within the resample
        sel = np.intersect1d(idx, high_idx, assume_unique=False)
        return safe_average_precision(labels[sel], probs[sel])

    panel = {
        "n_total": int(N),
        "n_high_agreement": int(len(high_idx)),
        "positive_truth_rate": bootstrap_ci(_truth, all_idx),
        "positive_emission_rate": bootstrap_ci(_emit, all_idx),
        "mAP": bootstrap_ci(_ap, all_idx),
        "balanced_accuracy_at_0p5": bootstrap_ci(_bacc, all_idx),
        "ECE_10bins": bootstrap_ci(_ece, all_idx),
        "mAP_high_agreement": bootstrap_ci(_ap_high, all_idx),
    }
    return panel


def mean_panel(per_crit: dict[str, dict]) -> dict:
    """Mean across C1/C2/C3 for the headline metrics. CI on the mean is
    a sanity bootstrap: redraw and re-average. Implemented by averaging
    the per-criterion point estimates and bootstrapping the row indices
    once (shared across criteria) using the joined DataFrame, but to
    keep this function lightweight we re-aggregate from the per-crit
    bootstraps deterministically.
    """
    out = {}
    for metric in ["mAP", "balanced_accuracy_at_0p5", "ECE_10bins",
                   "mAP_high_agreement", "positive_emission_rate",
                   "positive_truth_rate"]:
        points = [per_crit[c][metric]["point"] for c in CRITERIA]
        out[metric] = {
            "point": float(np.nanmean(points)),
            "ci_note": "mean of per-criterion point estimates; per-criterion CIs reported above",
        }
    return out

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(v: float, suffix: str = "%") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v*100:.2f}{suffix}"


def render_markdown(panel: dict, model_name: str, provenance: str | None) -> str:
    L = []
    L.append(f"# Harness metric panel — {model_name}")
    L.append("")
    L.append(f"**Generated:** {panel['generated_utc']}")
    L.append(f"**Predictions:** `{panel['inputs']['predictions_path']}`")
    L.append(f"**Annotation JSON:** `{panel['inputs']['annotation_path']}`")
    L.append(f"**Test rows scored:** {panel['inputs']['n_rows']}")
    L.append("")
    if provenance:
        L.append("> **Provenance:** " + provenance.replace("\n", " "))
        L.append("")
    L.append(f"Bootstrap: {BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}, "
             f"95% percentile CI. ECE: {ECE_BINS} equal-width bins on [0, 1] "
             f"(left-closed, top bin includes 1.0). Threshold for "
             f"balanced accuracy and positive emission: {THRESHOLD}.")
    L.append("")

    # Per-criterion table
    for metric_label, key in [
        ("mAP", "mAP"),
        ("Balanced accuracy @ 0.5", "balanced_accuracy_at_0p5"),
        ("ECE (10 bins)", "ECE_10bins"),
        ("mAP on high-agreement subset", "mAP_high_agreement"),
        ("Positive emission rate (pred ≥ 0.5)", "positive_emission_rate"),
        ("Ground-truth MV positive rate", "positive_truth_rate"),
    ]:
        L.append(f"### {metric_label}")
        L.append("")
        L.append("| Criterion | Point | 95% CI | n |")
        L.append("|---|---|---|---|")
        for c in CRITERIA:
            r = panel["per_criterion"][c][key]
            n = panel["per_criterion"][c]["n_high_agreement"] if "high_agreement" in key \
                else panel["per_criterion"][c]["n_total"]
            L.append(f"| {c} | {fmt(r['point'])} | "
                     f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}] | {n} |")
        mean_pt = panel["mean"][key]["point"]
        L.append(f"| **Mean** | **{fmt(mean_pt)}** | "
                 f"— | — |")
        L.append("")

    L.append("---")
    L.append("")
    L.append("Mean rows show the unweighted average of the three per-criterion "
             "point estimates. No CI is computed on the mean here; bootstrap a "
             "joint resample if a mean CI is needed.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score(predictions_path: Path, annotation_path: Path,
          model_name: str, provenance: str | None) -> dict:
    preds = load_predictions(predictions_path)
    ds = load_ds_table(annotation_path)
    merged = join_predictions_with_ds(preds, ds)

    per_crit = {}
    for c, ds_col, pred_col, lbl_col in [
        ("C1", "ds_c1", "c1_pred", "c1_label"),
        ("C2", "ds_c2", "c2_pred", "c2_label"),
        ("C3", "ds_c3", "c3_pred", "c3_label"),
    ]:
        probs = merged[pred_col].to_numpy(dtype=float)
        labels = merged[lbl_col].to_numpy(dtype=int)
        ds_vals = merged[ds_col].to_numpy(dtype=float)
        high_mask = (ds_vals == 0.0) | (ds_vals == 1.0)
        per_crit[c] = per_criterion_panel(probs, labels, high_mask)

    panel = {
        "model_name": model_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "predictions_path": str(predictions_path),
            "annotation_path": str(annotation_path),
            "n_rows": int(len(merged)),
        },
        "bootstrap": {
            "n_resamples": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
            "ci_percentile": [2.5, 97.5],
        },
        "ece_scheme": {
            "n_bins": ECE_BINS,
            "binning": "equal-width on [0,1]; bin i = [i/n, (i+1)/n), top bin includes 1.0",
        },
        "threshold": THRESHOLD,
        "provenance": provenance or "",
        "per_criterion": per_crit,
        "mean": mean_panel(per_crit),
    }
    return panel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    p.add_argument("--name", required=True,
                   help="model identifier used in output filenames")
    p.add_argument("--provenance", default="",
                   help="free-text note recorded in the JSON output")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = score(args.predictions, args.annotation, args.name, args.provenance)

    json_path = args.out_dir / f"metrics_{args.name}.json"
    md_path = args.out_dir / f"metrics_{args.name}.md"
    json_path.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(panel, args.name, args.provenance),
                       encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")

    # also dump headline to stdout for quick inspection
    print()
    print("Headline (point estimates):")
    for c in CRITERIA:
        ap = panel["per_criterion"][c]["mAP"]["point"]
        print(f"  {c} mAP: {ap*100:.2f}%")
    mean_ap = panel["mean"]["mAP"]["point"]
    print(f"  Mean mAP: {mean_ap*100:.2f}%")


if __name__ == "__main__":
    main()
