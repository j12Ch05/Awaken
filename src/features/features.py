"""
Awaken! - MediaPipe feature module (Steps 1-2, owner P)

Quick test (draws boxes + prints EAR / MAR / pitch):
    python src/features/features.py --model src/features/face_landmarker.task --image data/raw/pic.jpg --out tests/vis.jpg --csv tests/pic.csv
    python src/features/features.py --model src/features/face_landmarker.task --folder data/raw/ --n 50 --outdir tests/vis/ --csv tests/sample.csv
    python src/features/features.py --model src/features/face_landmarker.task --folder data/raw/ --all --csv tests/features.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Landmark indices (MediaPipe Face Mesh topology, 468 points + 10 iris)

# EAR points in order p1..p6: outer corner, upper1, upper2, inner corner, lower2, lower1
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # subject's right eye (image left)
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]   # subject's left eye (image right)

# Full eye contours (used for the bounding boxes)
RIGHT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Mouth: inner-lip vertical pairs + corners (MAR), outer lip (box)
MOUTH_CORNERS = (78, 308)
MOUTH_VERTICAL_PAIRS = [(13, 14), (82, 87), (312, 317)]
MOUTH_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Head pose: nose tip, chin, outer eye corners, mouth corners
POSE_IDX = [1, 152, 33, 263, 61, 291]
# Generic 3D face model (mm, y points DOWN like image coordinates so pitch ~0 when facing camera)
POSE_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, 330.0, -65.0),      # chin
    (-225.0, -170.0, -135.0), # eye outer corner, image-left  (idx 33)
    (225.0, -170.0, -135.0),  # eye outer corner, image-right (idx 263)
    (-150.0, 150.0, -125.0),  # mouth corner, image-left  (idx 61)
    (150.0, 150.0, -125.0),   # mouth corner, image-right (idx 291)
], dtype=np.float64)

# Data container
@dataclass
class FaceFeatures:
    ear_left: float
    ear_right: float
    ear: float                # mean of both eyes
    mar: float
    pitch: float              # degrees, 0 = facing camera; check sign on your own images
    yaw: float
    roll: float
    # boxes in pixels: (x1, y1, x2, y2), already padded and clipped
    face_box: tuple
    left_eye_box: tuple
    right_eye_box: tuple
    mouth_box: tuple
    landmarks_px: np.ndarray  # (N, 2) float array in pixels

# Geometry helpers
def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def eye_aspect_ratio(pts: np.ndarray, idx: list[int]) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)

def mouth_aspect_ratio(pts: np.ndarray) -> float:
    width = _dist(pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    vertical = np.mean([_dist(pts[a], pts[b]) for a, b in MOUTH_VERTICAL_PAIRS])
    return float(vertical / width)

def head_pose(pts: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees using solvePnP."""
    image_pts = np.array([pts[i] for i in POSE_IDX], dtype=np.float64)
    focal = float(w)  # rough approximation, fine for relative pitch
    cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(POSE_MODEL_3D, image_pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)  # degrees: (x=pitch, y=yaw, z=roll)
    pitch, yaw, roll = (float(a) for a in angles)

    def wrap(a: float) -> float:  # keep in [-90, 90]
        if a > 90:
            a -= 180
        elif a < -90:
            a += 180
        return a

    return wrap(pitch), wrap(yaw), wrap(roll)

def box_from_points(pts: np.ndarray, idx: Optional[list[int]], img_w: int, img_h: int,
                    margin: float = 0.15) -> tuple:
    """Box around the given landmark indices (all points if idx is None).
    Margin is a fraction of the box's LONGER side, added on every side
    (so thin eye boxes still get a useful height). Clipped to the image."""
    sel = pts if idx is None else pts[idx]
    x1, y1 = sel.min(axis=0)
    x2, y2 = sel.max(axis=0)
    pad = margin * max(x2 - x1, y2 - y1)
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(img_w - 1.0, x2 + pad)
    y2 = min(img_h - 1.0, y2 + pad)
    return (float(x1), float(y1), float(x2), float(y2))

def to_yolo_line(class_id: int, box: tuple, img_w: int, img_h: int) -> str:
    """Pixel xyxy box -> 'class xc yc w h' (normalised) for a YOLO txt label."""
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

# Extractor
class FeatureExtractor:
    def __init__(self, model_path: str, box_margin: float = 0.15):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.box_margin = box_margin

    def close(self) -> None:
        self.landmarker.close()

    def __call__(self, image_bgr: np.ndarray) -> Optional[FaceFeatures]:
        """image_bgr: HxW (grayscale) or HxWx3 BGR uint8. Returns None if no face."""
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

        ear_r = eye_aspect_ratio(pts, RIGHT_EYE_EAR)
        ear_l = eye_aspect_ratio(pts, LEFT_EYE_EAR)
        mar = mouth_aspect_ratio(pts)
        pitch, yaw, roll = head_pose(pts, w, h)

        m = self.box_margin
        return FaceFeatures(
            ear_left=ear_l, ear_right=ear_r, ear=(ear_l + ear_r) / 2, mar=mar,
            pitch=pitch, yaw=yaw, roll=roll,
            face_box=box_from_points(pts, None, w, h, m),
            left_eye_box=box_from_points(pts, LEFT_EYE_OUTLINE, w, h, m),
            right_eye_box=box_from_points(pts, RIGHT_EYE_OUTLINE, w, h, m),
            mouth_box=box_from_points(pts, MOUTH_OUTLINE, w, h, m),
            landmarks_px=pts,
        )

# Visual check (Step 1.5 and Step 2.3)
def draw_features(img: np.ndarray, f: FaceFeatures) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = img.copy()
    boxes = [(f.face_box, (255, 200, 0)), (f.left_eye_box, (0, 255, 0)),
             (f.right_eye_box, (0, 255, 0)), (f.mouth_box, (0, 0, 255))]
    for (x1, y1, x2, y2), color in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    text = f"EAR {f.ear:.2f}  MAR {f.mar:.2f}  pitch {f.pitch:+.0f}"
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out

CSV_FIELDS = [
    "image", "face_found", "img_w", "img_h",
    "ear_left", "ear_right", "ear", "mar", "pitch", "yaw", "roll",
    "face_x1", "face_y1", "face_x2", "face_y2",
    "leye_x1", "leye_y1", "leye_x2", "leye_y2",
    "reye_x1", "reye_y1", "reye_x2", "reye_y2",
    "mouth_x1", "mouth_y1", "mouth_x2", "mouth_y2",
]

def features_to_row(name: str, img_w: int, img_h: int, f: Optional[FaceFeatures]) -> dict:
    """One CSV row. If no face was found, face_found=0 and the other fields are empty."""
    row = {k: "" for k in CSV_FIELDS}
    row.update(image=name, face_found=int(f is not None), img_w=img_w, img_h=img_h)
    if f is None:
        return row
    row.update(
        ear_left=round(f.ear_left, 4), ear_right=round(f.ear_right, 4), ear=round(f.ear, 4),
        mar=round(f.mar, 4), pitch=round(f.pitch, 2), yaw=round(f.yaw, 2), roll=round(f.roll, 2),
    )
    for prefix, box in (("face", f.face_box), ("leye", f.left_eye_box),
                        ("reye", f.right_eye_box), ("mouth", f.mouth_box)):
        for k, v in zip(("x1", "y1", "x2", "y2"), box):
            row[f"{prefix}_{k}"] = round(v, 1)
    return row

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to face_landmarker.task")
    ap.add_argument("--image", help="single image")
    ap.add_argument("--out", default="vis.jpg", help="output path for --image")
    ap.add_argument("--folder", help="folder of images (recursive)")
    ap.add_argument("--n", type=int, default=50, help="random images to visualise from --folder")
    ap.add_argument("--outdir", default="vis", help="output dir for --folder")
    ap.add_argument("--csv", help="write one row per image to this CSV file")
    ap.add_argument("--all", action="store_true",
                    help="process EVERY image in --folder (ignores --n) and skip drawing; use with --csv")
    args = ap.parse_args()

    ext = FeatureExtractor(args.model)
    rows: list[dict] = []

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Cannot read {args.image}")
        f = ext(img)
        rows.append(features_to_row(Path(args.image).name, img.shape[1], img.shape[0], f))
        if f is None:
            print("No face found")
        else:
            print(f"EAR L/R {f.ear_left:.3f}/{f.ear_right:.3f}  MAR {f.mar:.3f}  "
                  f"pitch/yaw/roll {f.pitch:+.1f}/{f.yaw:+.1f}/{f.roll:+.1f}")
            cv2.imwrite(args.out, draw_features(img, f))
            print("saved", args.out)

    if args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        root = Path(args.folder)
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
        if not args.all:
            random.shuffle(files)
            files = files[: args.n]
            Path(args.outdir).mkdir(parents=True, exist_ok=True)
        found = 0
        for i, p in enumerate(files, 1):
            img = cv2.imread(str(p))
            if img is None:
                continue
            f = ext(img)
            # relative path keeps subject/class folders, which you may need for splitting
            rows.append(features_to_row(str(p.relative_to(root)), img.shape[1], img.shape[0], f))
            if f is None:
                if not args.all:
                    print("no face:", p.name)
            else:
                found += 1
                if not args.all:
                    print(f"{p.name:40s} EAR {f.ear:.3f}  MAR {f.mar:.3f}  pitch {f.pitch:+.1f}")
                    cv2.imwrite(str(Path(args.outdir) / p.name), draw_features(img, f))
            if args.all and i % 500 == 0:
                print(f"{i}/{len(files)} processed")
        print(f"face found in {found}/{len(files)} images")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")

    ext.close()


if __name__ == "__main__":
    main()