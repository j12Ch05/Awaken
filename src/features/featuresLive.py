"""
Awaken! - live webcam version of the MediaPipe feature module.

Put next to features.py (src/features/) and run:
    python src/features/featuresLive.py --model src/features/face_landmarker.task
    python src/features/featuresLive.py --model src/features/face_landmarker.task --gray
    python src/features/featuresLive.py --model src/features/face_landmarker.task --csv tests/live.csv

Keys:  q = quit   c = calibrate (look at the camera, eyes open, mouth closed)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from features import FaceFeatures, FeatureExtractor, draw_features  # same folder

class EMA:
    """Exponential moving average to smooth jittery per-frame values."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to face_landmarker.task")
    ap.add_argument("--cam", type=int, default=0, help="camera index (0 = integrated)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--gray", action="store_true",
                    help="convert frames to grayscale before MediaPipe (default: original colour frames)")
    ap.add_argument("--alpha", type=float, default=0.4, help="EMA smoothing (1 = none)")
    ap.add_argument("--calib-sec", type=float, default=20.0, help="calibration length")
    ap.add_argument("--csv", help="log one row per frame to this CSV file")
    args = ap.parse_args()

    # CAP_DSHOW is faster to open on Windows; ignored elsewhere
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(args.cam)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.cam}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ext = FeatureExtractor(args.model)
    s_ear, s_mar, s_pitch = EMA(args.alpha), EMA(args.alpha), EMA(args.alpha)

    csv_file = writer = None
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["t", "face_found", "ear_left", "ear_right", "ear", "mar", "pitch",
                         "ear_smooth", "mar_smooth", "pitch_smooth", "ear_ratio", "infer_ms"])

    baseline: Optional[dict] = None     # {"ear":..., "mar":..., "pitch":...}
    calib_start: Optional[float] = None
    calib_samples: list[tuple[float, float, float]] = []

    t0 = time.time()
    fps, last = 0.0, time.time()
    print("Running. Press 'c' to calibrate, 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break

            # Model input: grayscale (as 3-channel) to match the training data
            model_in = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if args.gray else frame

            ti = time.time()
            f: Optional[FaceFeatures] = ext(model_in)
            infer_ms = (time.time() - ti) * 1000

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            display = frame.copy()
            ear_ratio = ""
            if f is not None:
                ear = s_ear.update(f.ear)
                mar = s_mar.update(f.mar)
                pitch = s_pitch.update(f.pitch)
                display = draw_features(display, f)

                if calib_start is not None:
                    calib_samples.append((f.ear, f.mar, f.pitch))
                if baseline:
                    ear_ratio = ear / baseline["ear"]
                    cv2.putText(display, f"EAR/base {ear_ratio:.2f}  dMAR {mar - baseline['mar']:+.2f}  "
                                f"dPitch {pitch - baseline['pitch']:+.0f}",
                                (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(display, f"smoothed EAR {ear:.2f} MAR {mar:.2f} pitch {pitch:+.0f}",
                            (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                # Do not carry stale values across a lost face
                s_ear.reset(); s_mar.reset(); s_pitch.reset()
                cv2.putText(display, "FACE NOT FOUND", (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Calibration progress / result
            if calib_start is not None:
                remaining = args.calib_sec - (now - calib_start)
                if remaining > 0:
                    cv2.putText(display, f"CALIBRATING {remaining:4.1f}s - eyes open, mouth closed",
                                (5, display.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 200, 255), 2, cv2.LINE_AA)
                else:
                    if len(calib_samples) >= 20:
                        arr = np.array(calib_samples)
                        baseline = {"ear": float(np.median(arr[:, 0])),
                                    "mar": float(np.median(arr[:, 1])),
                                    "pitch": float(np.median(arr[:, 2]))}
                        print(f"Baseline: {baseline}")
                    else:
                        print("Calibration failed: too few face frames")
                    calib_start = None
                    calib_samples = []
            elif baseline:
                cv2.putText(display, f"baseline EAR {baseline['ear']:.2f}", (5, display.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.putText(display, f"FPS {fps:4.1f}  infer {infer_ms:4.1f} ms", (5, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            if writer is not None:
                if f is not None:
                    writer.writerow([f"{now - t0:.3f}", 1, f"{f.ear_left:.4f}", f"{f.ear_right:.4f}",
                                     f"{f.ear:.4f}", f"{f.mar:.4f}", f"{f.pitch:.2f}",
                                     f"{s_ear.value:.4f}", f"{s_mar.value:.4f}", f"{s_pitch.value:.2f}",
                                     f"{ear_ratio:.4f}" if ear_ratio != "" else "", f"{infer_ms:.1f}"])
                else:
                    writer.writerow([f"{now - t0:.3f}", 0] + [""] * 9 + [f"{infer_ms:.1f}"])

            cv2.imshow("Awaken! features (q quit, c calibrate)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calib_start, calib_samples = time.time(), []
                print(f"Calibrating for {args.calib_sec:.0f}s ...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ext.close()
        if csv_file:
            csv_file.close()
            print(f"Saved log -> {args.csv}")

if __name__ == "__main__":
    main()