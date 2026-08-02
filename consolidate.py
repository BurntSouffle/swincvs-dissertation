#!/usr/bin/env python3
# Consolidate the 3-seed controlled soft-vs-hard Stage-1 panel from the per-run
# score.log (mean + per-criterion mAP) and metrics_*.md (ECE). Seed 1 reuses the
# existing hard/soft dirs; seeds 2,3 from the multi-seed run.
import os, re, statistics, glob
ROOT = "/workspace/experiment_outputs/stage1_pair"
DIRS = {1: ("hard", "soft"), 2: ("hard_sd2", "soft_sd2"), 3: ("hard_sd3", "soft_sd3")}

def parse(rd):
    out = {"mean": None, "C1": None, "C2": None, "C3": None, "ECE": None}
    sc = f"{ROOT}/{rd}/score.log"
    if os.path.exists(sc):
        t = open(sc).read()
        for k, p in [("mean", "Mean mAP"), ("C1", "C1 mAP"), ("C2", "C2 mAP"), ("C3", "C3 mAP")]:
            m = re.search(rf"{p}:\s*([0-9.]+)", t); out[k] = float(m.group(1)) if m else None
    md = glob.glob(f"{ROOT}/{rd}/metrics_*.md")
    if md:
        rows = [l for l in open(md[0]) if "**Mean**" in l]
        if len(rows) >= 3:
            m = re.search(r"([0-9.]+)", rows[2]); out["ECE"] = float(m.group(1)) if m else None
    return out

def ms(vs):
    vs = [v for v in vs if v is not None]
    if not vs: return None, None
    return statistics.mean(vs), (statistics.stdev(vs) if len(vs) > 1 else 0.0)

seeds = sorted(DIRS)
H = {sd: parse(DIRS[sd][0]) for sd in seeds}
S = {sd: parse(DIRS[sd][1]) for sd in seeds}

print("# 3-seed CONTROLLED soft-vs-hard Stage-1 panel (single-frame test)\n")
print("Recipe: backbone 1e-5, head 1e-3, 8 ep, /dev/shm. Arms differ ONLY in SOFT_TRAIN_LABELS.\n")
print("## Per-seed mean mAP")
print("| seed | hard | soft | delta |\n|---|---|---|---|")
deltas = []
for sd in seeds:
    h, s = H[sd]["mean"], S[sd]["mean"]
    d = (s - h) if (h is not None and s is not None) else None
    if d is not None: deltas.append(d)
    print(f"| {sd} | {h} | {s} | {('%+.2f' % d) if d is not None else 'NA'} |")

print("\n## 3-seed mean +/- std")
print("| metric | hard | soft | delta(soft-hard) |\n|---|---|---|---|")
for k, label in [("mean", "mean mAP"), ("C1", "C1"), ("C2", "C2"), ("C3", "C3"), ("ECE", "ECE")]:
    hm, hs = ms([H[sd][k] for sd in seeds]); sm, ss = ms([S[sd][k] for sd in seeds])
    dl = [S[sd][k] - H[sd][k] for sd in seeds if S[sd][k] is not None and H[sd][k] is not None]
    dm, ds = ms(dl)
    if hm is None or sm is None or dm is None:
        print(f"| {label} | NA | NA | NA |")
    else:
        print(f"| {label} | {hm:.2f}+/-{hs:.2f} | {sm:.2f}+/-{ss:.2f} | {dm:+.2f}+/-{ds:.2f} |")

dm, ds = ms(deltas)
allpos = bool(deltas) and all(d > 0 for d in deltas)
print("\n## VERDICT")
if dm is None:
    print("INCONCLUSIVE — missing runs.")
elif dm > 0.5 and (dm - ds) > 0 and allpos:
    print(f"SOFT CONSISTENTLY BEATS HARD: mean delta {dm:+.2f}+/-{ds:.2f}pp, positive on all "
          f"{len(deltas)} seeds, lower bound > 0. The representational lever HOLDS -> frozen-Stage-2 "
          f"+ SOTA-vs-67.45 justified (report for sign-off).")
elif allpos and dm > 0.3:
    print(f"SOFT WEAKLY BEATS HARD: mean delta {dm:+.2f}+/-{ds:.2f}pp, positive on all seeds but the "
          f"spread reaches ~0. Real but modest; weigh against the cost + the sub-65 absolute level.")
else:
    print(f"SOFT ~= HARD: mean delta {dm:+.2f}+/-{ds:.2f}pp (crosses 0 / inconsistent across seeds). "
          f"The representational lever does NOT reliably help -> stop / reconsider this path.")
print(f"\nPer-seed mean deltas: {['%+.2f' % d for d in deltas]}")
print("Also weigh: C2 regressed at seed 1 (-4.2); ECE better for soft at seed 1 (-2.43).")
print("Absolute level caveat: both arms ~60 test mAP, BELOW no_augm_sd4 (65.64) -> this 8-ep")
print("recipe overfits; not competitive for 67.45 without recipe tuning (separate from this gate).")
