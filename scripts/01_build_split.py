#!/usr/bin/env python3
"""
01_build_split.py - parse file names, subsample frames, and build a leakage-safe train/val/test split.

Your file names:  <subject>_<glasses|noglasses>_<scenario>_<frame>_<drowsy|notdrowsy>.jpg
Only 4 subjects exist (001, 002, 005, 006), so a 70/15/15 split BY SUBJECT is impossible. Strategy:
  * TEST  = one whole held-out subject (default 005)  -> measures real cross-person generalisation
  * VAL   = blocks of consecutive frames cut from the remaining subjects' videos (with a gap around
            each block so neighbouring near-identical frames never end up in both train and val)
  * TRAIN = everything else (including the tiny subject 006)

Usage:
  python 01_build_split.py --src /content/raw/train_data --dst /content/dataset
  python 01_build_split.py --src ... --dst ... --stride 5 --test-subject 005 --val-frac 0.15 --block 150 --gap 15

Outputs:
  <dst>/images/{train,val,test}/<original name>.jpg   (copies; use --symlink to link instead)
  <dst>/manifest.csv                                   (split, path, subject, glasses, scenario, frame, label)
  Prints a summary table. Does not modify --src.
"""
import argparse
import csv
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict

PAT = re.compile(r"^(\d+)_(glasses|noglasses)_([A-Za-z]+)_(\d+)_(drowsy|notdrowsy)\.(jpg|jpeg|png)$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder that contains drowsy/ and notdrowsy/")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--stride", type=int, default=5, help="keep every Nth frame of each video (default 5)")
    ap.add_argument("--test-subject", default="005", help="subject held out completely for testing")
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of blocks (non-test) used for validation")
    ap.add_argument("--block", type=int, default=150, help="frames per block before striding (150 = 5 s at 30 fps)")
    ap.add_argument("--gap", type=int, default=15, help="frames dropped at both edges of each val block")
    ap.add_argument("--noglasses-only", action="store_true", help="drop all 'glasses' frames")
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="only print the summary, copy nothing")
    a = ap.parse_args()

    rows, bad, mismatch = [], 0, 0
    for label_dir in ("drowsy", "notdrowsy"):
        d = os.path.join(a.src, label_dir)
        if not os.path.isdir(d):
            sys.exit(f"Missing folder: {d}")
        for fn in os.listdir(d):
            m = PAT.match(fn)
            if not m:
                bad += 1
                continue
            subj, gl, scen, frame, lab, _ = m.groups()
            if lab.lower() != label_dir:
                mismatch += 1
            rows.append(dict(path=os.path.join(d, fn), name=fn, subject=subj, glasses=gl.lower(),
                             scenario=scen, frame=int(frame), label=label_dir))
    print(f"Parsed {len(rows)} files; unparsable names: {bad}; folder/suffix mismatches: {mismatch}")
    if not rows:
        sys.exit("No files parsed - check --src and the filename pattern.")

    if a.noglasses_only:
        rows = [r for r in rows if r["glasses"] == "noglasses"]

    # one "video" = subject + glasses + scenario
    videos = defaultdict(list)
    for r in rows:
        videos[(r["subject"], r["glasses"], r["scenario"])].append(r)

    rng = random.Random(a.seed)
    kept = []
    for key, rs in sorted(videos.items()):
        subj = key[0]
        # pick val blocks for this video (only for non-test subjects)
        blocks = sorted({r["frame"] // a.block for r in rs})
        n_val = 0 if subj == a.test_subject else max(1, round(len(blocks) * a.val_frac)) if len(blocks) > 1 else 0
        val_blocks = set(rng.sample(blocks, n_val)) if n_val else set()
        for r in rs:
            if r["frame"] % a.stride != 0:
                continue
            b = r["frame"] // a.block
            if subj == a.test_subject:
                split = "test"
            elif b in val_blocks:
                pos = r["frame"] - b * a.block
                if pos < a.gap or pos >= a.block - a.gap:
                    continue  # buffer zone, discard
                split = "val"
            else:
                # discard train frames that sit right next to a val block (leak guard)
                pos = r["frame"] - b * a.block
                if (b + 1) in val_blocks and pos >= a.block - a.gap:
                    continue
                if (b - 1) in val_blocks and pos < a.gap:
                    continue
                split = "train"
            r["split"] = split
            kept.append(r)

    print(f"\nKept {len(kept)} of {len(rows)} frames after stride={a.stride}\n")
    for title, keyf in (("split x label", lambda r: (r["split"], r["label"])),
                        ("split x glasses", lambda r: (r["split"], r["glasses"])),
                        ("split x scenario", lambda r: (r["split"], r["scenario"])),
                        ("split x subject", lambda r: (r["split"], r["subject"]))):
        c = Counter(keyf(r) for r in kept)
        print(f"--- {title} ---")
        for k in sorted(c):
            print(f"  {k[0]:6s} {k[1]:22s} {c[k]:7d}")
    tot = Counter(r["split"] for r in kept)
    n = sum(tot.values())
    print("\nShare: " + ", ".join(f"{k}={v} ({100*v/n:.0f}%)" for k, v in sorted(tot.items())))

    if a.dry_run:
        print("\nDry run: nothing copied.")
        return

    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(a.dst, "images", sp), exist_ok=True)
    for r in kept:
        dst = os.path.join(a.dst, "images", r["split"], r["name"])
        if a.symlink:
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(r["path"]), dst)
        else:
            shutil.copy2(r["path"], dst)
    with open(os.path.join(a.dst, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "name", "subject", "glasses", "scenario", "frame", "label"])
        for r in kept:
            w.writerow([r["split"], r["name"], r["subject"], r["glasses"], r["scenario"], r["frame"], r["label"]])
    print(f"\nDone. Images in {a.dst}/images/<split>/ and {a.dst}/manifest.csv")


if __name__ == "__main__":
    main()
