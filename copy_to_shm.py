#!/usr/bin/env python3
# Copy the LABELLED keyframes (single-frame Stage-1 needs only these) + per-split
# annotation JSONs from the flaky MooseFS /workspace volume into /dev/shm (RAM, 117G
# free). Training then reads from RAM -> no MooseFS read errors/hangs. Per-file OSError
# retry; verifies non-zero size. Idempotent (skips already-copied).
import json, os, shutil, time, sys
SRC = "/workspace/endoscapes"
DST = "/dev/shm/endoscapes"
total_fail = 0
for split in ["train", "val", "test"]:
    os.makedirs(f"{DST}/{split}", exist_ok=True)
    # copy all annotation JSONs for the split (small)
    for j in os.listdir(f"{SRC}/{split}"):
        if j.endswith(".json"):
            try: shutil.copy(f"{SRC}/{split}/{j}", f"{DST}/{split}/{j}")
            except OSError: pass
    d = json.load(open(f"{SRC}/{split}/annotation_ds_coco.json"))
    names = [os.path.basename(img["file_name"]) for img in d["images"]]
    fail = 0
    for i, n in enumerate(names):
        s, t = f"{SRC}/{split}/{n}", f"{DST}/{split}/{n}"
        if os.path.exists(t) and os.path.getsize(t) > 0:
            continue
        ok = False
        for attempt in range(6):
            try:
                shutil.copy(s, t)
                if os.path.getsize(t) > 0:
                    ok = True; break
            except OSError:
                time.sleep(0.5)
        if not ok:
            fail += 1; print(f"  FAIL {split}/{n}", flush=True)
        if (i + 1) % 2000 == 0:
            print(f"  {split}: {i+1}/{len(names)}", flush=True)
    print(f"{split}: {len(names)} frames, {fail} failed", flush=True)
    total_fail += fail
print(f"COPY_DONE total_fail={total_fail}", flush=True)
