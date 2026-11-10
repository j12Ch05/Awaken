#!/usr/bin/env python3
"""
audit_dataset.py - inspect a dataset folder and print everything needed to plan Phase 2.

Usage (Colab or local):
    python audit_dataset.py /path/to/dataset_folder
    python audit_dataset.py /path/to/dataset_folder --out audit_report.txt --samples 300

Works with: loose images, YOLO/VOC/COCO/CSV/JSON labels, zip/tar archives, parquet files
(Hugging Face format). Reads only a sample of images, never modifies anything.
Needs only the standard library; Pillow, numpy and pyarrow are used if installed
(pip install pillow numpy pyarrow).

Afterwards: send the content of audit_report.txt (and sample_sheet.jpg if created).
"""
import argparse
import collections
import io
import json
import os
import random
import re
import sys
import tarfile
import zipfile

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
LABEL_EXT = {".txt", ".json", ".xml", ".csv", ".tsv", ".yaml", ".yml"}

try:
    from PIL import Image
except Exception:
    Image = None
try:
    import numpy as np
except Exception:
    np = None
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

LINES = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


def hr(title):
    out("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def walk(root):
    files = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".git")]
        for f in fn:
            p = os.path.join(dp, f)
            try:
                files.append((p, os.path.getsize(p)))
            except OSError:
                pass
    return files


def section_overview(root, files):
    hr("1. OVERVIEW")
    total = sum(s for _, s in files)
    out(f"Root: {root}")
    out(f"Files: {len(files)}   Total size: {human(total)}")
    ext = collections.Counter(os.path.splitext(p)[1].lower() or "(none)" for p, _ in files)
    size_by_ext = collections.Counter()
    for p, s in files:
        size_by_ext[os.path.splitext(p)[1].lower() or "(none)"] += s
    out("\nFiles by extension (count, size):")
    for e, c in ext.most_common(20):
        out(f"  {e:10s} {c:8d}   {human(size_by_ext[e])}")


def section_tree(root, files, depth=3):
    hr("2. FOLDER STRUCTURE (depth %d, counts of files per folder)" % depth)
    counts = collections.Counter()
    for p, _ in files:
        rel = os.path.relpath(os.path.dirname(p), root)
        parts = [] if rel == "." else rel.split(os.sep)
        for d in range(0, min(len(parts), depth) + 1):
            counts[os.sep.join(parts[:d])] += 1
    shown = 0
    for k in sorted(counts):
        lvl = 0 if k == "" else k.count(os.sep) + 1
        if lvl > depth:
            continue
        name = "." if k == "" else k
        out(f"  {'  ' * lvl}{name}  [{counts[k]} files incl. subfolders]")
        shown += 1
        if shown > 150:
            out("  ... (truncated)")
            break
    # leaf folder class-like counts
    leaf = collections.Counter(os.path.relpath(os.path.dirname(p), root) for p, _ in files
                               if os.path.splitext(p)[1].lower() in IMG_EXT)
    if leaf:
        out("\nImages per folder (top 40) - folder names are often the class labels:")
        for k, c in leaf.most_common(40):
            out(f"  {c:8d}  {k}")


def section_names(files):
    hr("3. FILE NAMES (look for subject / session / frame IDs)")
    imgs = sorted(p for p, _ in files if os.path.splitext(p)[1].lower() in IMG_EXT)
    if not imgs:
        out("No loose image files found.")
        return
    rnd = random.Random(0)
    sample = rnd.sample(imgs, min(25, len(imgs)))
    out("25 random image paths:")
    for p in sorted(sample):
        out("  " + p)
    # number-pattern guess
    patt = collections.Counter(re.sub(r"\d+", "#", os.path.basename(p)) for p in imgs)
    out("\nFilename patterns (digits replaced by #), top 10:")
    for k, c in patt.most_common(10):
        out(f"  {c:8d}  {k}")
    tok = collections.Counter()
    for p in imgs[:20000]:
        for t in re.split(r"[^A-Za-z0-9]+", os.path.splitext(os.path.basename(p))[0]):
            if t and not t.isdigit():
                tok[t] += 1
    out("\nMost common non-numeric tokens in names (may be class or subject words):")
    out("  " + ", ".join(f"{t}({c})" for t, c in tok.most_common(25)))
    # numeric tokens: how many distinct values in the first numeric field
    firsts = collections.Counter()
    for p in imgs[:50000]:
        m = re.findall(r"\d+", os.path.basename(p))
        if m:
            firsts[m[0]] += 1
    if firsts:
        out(f"\nDistinct values of the FIRST number in filenames: {len(firsts)} "
            f"(if 5-200 this is probably a subject ID). Top: "
            + ", ".join(f"{k}({c})" for k, c in firsts.most_common(10)))


def section_images(files, nsamples, sheet_path):
    hr("4. IMAGES (sampled)")
    imgs = [p for p, _ in files if os.path.splitext(p)[1].lower() in IMG_EXT]
    out(f"Loose images: {len(imgs)}")
    if not imgs:
        return
    if Image is None:
        out("Pillow not installed -> pip install pillow  (skipping pixel stats)")
        return
    rnd = random.Random(1)
    sample = rnd.sample(imgs, min(nsamples, len(imgs)))
    modes = collections.Counter()
    sizes = collections.Counter()
    gray_like = 0
    ok = 0
    thumbs = []
    for p in sample:
        try:
            with Image.open(p) as im:
                modes[im.mode] += 1
                sizes[im.size] += 1
                if np is not None:
                    a = np.asarray(im.convert("RGB"))
                    if (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all():
                        gray_like += 1
                if len(thumbs) < 24:
                    t = im.convert("RGB")
                    t.thumbnail((160, 160))
                    thumbs.append(t)
                ok += 1
        except Exception as e:
            out(f"  unreadable: {p} ({e})")
    out(f"Sampled {ok} images")
    out("Modes: " + ", ".join(f"{m}={c}" for m, c in modes.most_common()))
    out("Sizes (WxH): " + ", ".join(f"{s}={c}" for s, c in sizes.most_common(8)))
    if np is not None:
        out(f"Grayscale content (R=G=B everywhere): {gray_like}/{ok}")
    if thumbs:
        cols = 6
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 160, rows * 160), (30, 30, 30))
        for i, t in enumerate(thumbs):
            sheet.paste(t, ((i % cols) * 160, (i // cols) * 160))
        sheet.save(sheet_path, quality=85)
        out(f"Contact sheet of 24 random images saved to: {sheet_path}  (please send/look at it)")


def section_labels(root, files):
    hr("5. LABEL / ANNOTATION FILES")
    cand = [(p, s) for p, s in files if os.path.splitext(p)[1].lower() in LABEL_EXT]
    out(f"Candidate label/metadata files: {len(cand)}")
    if not cand:
        out("None found -> labels are probably folder names, file names, or inside parquet/archives.")
        return
    by_ext = collections.defaultdict(list)
    for p, s in cand:
        by_ext[os.path.splitext(p)[1].lower()].append(p)
    # YOLO txt
    txts = by_ext.get(".txt", [])
    if txts:
        yolo_like = 0
        classes = collections.Counter()
        samples = []
        for p in random.Random(2).sample(txts, min(300, len(txts))):
            try:
                with open(p, "r", errors="ignore") as f:
                    ls = [l.strip() for l in f if l.strip()]
            except OSError:
                continue
            good = 0
            for l in ls:
                parts = l.split()
                if len(parts) == 5 and parts[0].isdigit():
                    try:
                        vals = list(map(float, parts[1:]))
                        if all(0 <= v <= 1 for v in vals):
                            good += 1
                            classes[int(parts[0])] += 1
                    except ValueError:
                        pass
            if ls and good == len(ls):
                yolo_like += 1
            if len(samples) < 3 and ls:
                samples.append((p, ls[:3]))
        out(f".txt files: {len(txts)}; of 300 sampled, YOLO-format (class cx cy w h in 0-1): {yolo_like}")
        if classes:
            out("Class ids seen in sample boxes: " + ", ".join(f"{k}={v}" for k, v in sorted(classes.items())))
        for p, ls in samples:
            out(f"  example {p}: {ls}")
    for ext in (".xml", ".json", ".csv", ".tsv", ".yaml", ".yml"):
        lst = by_ext.get(ext, [])
        if not lst:
            continue
        out(f"\n{ext} files: {len(lst)}  e.g. {lst[0]}")
        for p in lst[:3]:
            try:
                size = os.path.getsize(p)
                with open(p, "r", errors="ignore") as f:
                    head = f.read(1200 if size > 1200 else size)
                out(f"--- head of {p} ({human(size)}) ---")
                out(head)
                if ext == ".json" and size < 200_000_000:
                    with open(p, "r", errors="ignore") as f:
                        d = json.load(f)
                    if isinstance(d, dict):
                        out(f"JSON top-level keys: {list(d.keys())[:20]}")
                        if "categories" in d:
                            out(f"COCO categories: {d['categories']}")
                            out(f"COCO images: {len(d.get('images', []))}  annotations: {len(d.get('annotations', []))}")
            except Exception as e:
                out(f"  (could not read {p}: {e})")
    # YOLO-style label folder pairing
    lbl_dirs = [p for p, _ in files if "/labels/" in p.replace("\\", "/")]
    if lbl_dirs:
        out(f"\nFiles under a 'labels' folder: {len(lbl_dirs)}")


def section_archives(files):
    arch = [p for p, _ in files if p.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz", ".gz"))]
    if not arch:
        return
    hr("6. ARCHIVES (contents listed without extracting)")
    for p in arch[:10]:
        out(f"{p}  ({human(os.path.getsize(p))})")
        try:
            if p.lower().endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    names = z.namelist()
            else:
                with tarfile.open(p) as t:
                    names = [m.name for m in t.getmembers()[:100000]]
            ext = collections.Counter(os.path.splitext(n)[1].lower() for n in names)
            out(f"  entries: {len(names)}  by extension: {dict(ext.most_common(8))}")
            for n in names[:15]:
                out("   " + n)
        except Exception as e:
            out(f"  (cannot list: {e})")


def section_parquet(files):
    pqs = [p for p, _ in files if p.lower().endswith(".parquet")]
    if not pqs:
        return
    hr("7. PARQUET FILES (Hugging Face format)")
    out(f"Parquet files: {len(pqs)}")
    if pq is None:
        out("pyarrow not installed -> pip install pyarrow  (then re-run)")
        return
    rows_total = 0
    per_split = collections.Counter()
    schema_printed = False
    label_counts = collections.Counter()
    label_col = None
    for p in sorted(pqs):
        try:
            pf = pq.ParquetFile(p)
        except Exception as e:
            out(f"  cannot open {p}: {e}")
            continue
        n = pf.metadata.num_rows
        rows_total += n
        split = os.path.basename(p).split("-")[0]
        per_split[split] += n
        out(f"  {p}: {n} rows, {pf.metadata.num_row_groups} row groups, {human(os.path.getsize(p))}")
        if not schema_printed:
            schema_printed = True
            out("\nSchema:")
            for f in pf.schema_arrow:
                out(f"   {f.name}: {f.type}")
            md = pf.schema_arrow.metadata or {}
            if b"huggingface" in md:
                try:
                    info = json.loads(md[b"huggingface"])
                    out("HF features: " + json.dumps(info.get("info", {}).get("features", info), indent=1)[:2500])
                except Exception:
                    pass
            # first rows (non-image columns)
            img_cols = [f.name for f in pf.schema_arrow
                        if "struct" in str(f.type) and "bytes" in str(f.type) or f.name.lower() in ("image", "img")]
            other = [f.name for f in pf.schema_arrow if f.name not in img_cols]
            if other:
                try:
                    t = pf.read_row_group(0, columns=other).slice(0, 8).to_pylist()
                    out("\nFirst rows (non-image columns):")
                    for r in t:
                        out("   " + str(r)[:300])
                except Exception as e:
                    out(f"  (could not read first rows: {e})")
            for c in other:
                ty = str(pf.schema_arrow.field(c).type)
                if ty.startswith(("int", "string", "dictionary")) and c.lower() in (
                        "label", "labels", "class", "target", "category", "state", "drowsy", "y"):
                    label_col = c
            if img_cols and Image is not None:
                try:
                    r = pf.read_row_group(0, columns=[img_cols[0]]).slice(0, 5).to_pylist()
                    for row in r:
                        v = row[img_cols[0]]
                        b = v.get("bytes") if isinstance(v, dict) else v
                        if b:
                            im = Image.open(io.BytesIO(b))
                            out(f"   embedded image: size={im.size} mode={im.mode} format={im.format} path={v.get('path') if isinstance(v, dict) else ''}")
                except Exception as e:
                    out(f"  (could not decode image: {e})")
        if label_col:
            try:
                col = pf.read(columns=[label_col]).column(0).to_pylist()
                label_counts.update(col)
            except Exception:
                pass
    out(f"\nTotal rows: {rows_total}   by split prefix: {dict(per_split)}")
    if label_col:
        out(f"Label column '{label_col}' value counts: {dict(label_counts.most_common(30))}")
    else:
        out("No obvious label column found - read the schema above and tell me which column is the label.")


def section_videos(files):
    v = [p for p, _ in files if os.path.splitext(p)[1].lower() in VID_EXT]
    if not v:
        return
    hr("8. VIDEOS")
    out(f"Video files: {len(v)}, e.g. {v[:5]}")


def section_summary(files):
    hr("9. QUICK ANSWERS (for the plan)")
    n_img = sum(1 for p, _ in files if os.path.splitext(p)[1].lower() in IMG_EXT)
    n_txt = sum(1 for p, _ in files if p.lower().endswith(".txt"))
    n_xml = sum(1 for p, _ in files if p.lower().endswith(".xml"))
    n_json = sum(1 for p, _ in files if p.lower().endswith(".json"))
    n_pq = sum(1 for p, _ in files if p.lower().endswith(".parquet"))
    out(f"Loose images: {n_img} | .txt: {n_txt} | .xml: {n_xml} | .json: {n_json} | parquet: {n_pq}")
    if n_pq:
        out("-> Data is in parquet: images are probably embedded; extract them to files first.")
    elif n_img and n_txt >= n_img * 0.8:
        out("-> Looks like image + .txt pairs (maybe already YOLO boxes). Check section 5.")
    elif n_img and n_xml:
        out("-> Looks like Pascal-VOC boxes (.xml).")
    elif n_img:
        out("-> Images without label files: labels are likely the folder names -> auto-label with MediaPipe.")
    else:
        out("-> No images found at this path. Check the folder, or archives (section 6).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="audit_report.txt")
    ap.add_argument("--samples", type=int, default=300, help="images to open for pixel stats")
    ap.add_argument("--sheet", default="sample_sheet.jpg")
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        sys.exit(f"Not a folder: {a.root}")
    files = walk(a.root)
    section_overview(a.root, files)
    section_tree(a.root, files)
    section_names(files)
    section_images(files, a.samples, a.sheet)
    section_labels(a.root, files)
    section_archives(files)
    section_parquet(files)
    section_videos(files)
    section_summary(files)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    print(f"\nReport saved to {a.out}")


if __name__ == "__main__":
    main()
