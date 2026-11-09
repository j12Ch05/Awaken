# Awaken! – Real-Time Driver Drowsiness Detection (YOLO11s + MediaPipe)

**Team:** P, J  |  **Type:** academic project, open source (AGPL-3.0 compatible)  |  **Goal:** real-time demo on a laptop webcam
**Schedule:** 8 weeks, Mon 9 Nov 2026 → Sun 3 Jan 2027 (hard deadline ≈ 7 Jan 2027, so 4 days of buffer)
**Exams:** start in ~4 weeks (≈ week 4–5). Weeks 5–6 are planned at **reduced capacity** (assumed; see open question Q1).

> Everything below applies the answers in `teammate_questions_template.md`. Items marked **[DEFAULT]** are values I proposed because the template left them blank or unclear. Please confirm or change them (see `open_questions.md`).

---

## 0. Decisions Applied (from the template)

| Topic | Decision |
|---|---|
| Model | **YOLO11s** (Ultralytics, AGPL-3.0 accepted because the project is open source) |
| Design | **Option C – hybrid:** YOLO11s detects states/regions; MediaPipe gives EAR / MAR / head-pitch; rules fuse both |
| Comparison with RF/SVM/NN/CNN | **Dropped** (YOLO only). Only ablation kept: *YOLO only* vs *YOLO + MediaPipe fusion* |
| Decision layer | **Threshold rules** (no second ML model) |
| Classes (7) | `face`, `eye_open`, `eye_closed`, `yawn`, `mouth_closed`, `head_down`, `posture_alert` |
| Cues | eye closure, yawning, head nodding, shoulder posture |
| Out of scope | distraction (looking away), phone use, seatbelt, night / infrared, glasses |
| Data | HuggingFace `chbh7051/driver-drowsiness-dataset` (≈ 3.1 GB, **grayscale** images), stored on Drive, split 70/15/15 |
| Training | Google Colab |
| Deployment | Laptop with integrated webcam, **offline** (assume no internet at demo) |
| Alarm | Beep, **2 levels**: Level 1 = drowsiness detected, Level 2 = asleep |
| UI | Overlay on the live camera feed |
| Logging | Log when an alert level triggers + the values that crossed the threshold. **No video stored** |
| Repo / comms | GitHub; WhatsApp + Discord |
| Python | Python 3.14.7 as stated, PyTorch, OpenCV. **[Flag]** see Q8 |

---

## 1. Architecture

```
Webcam frame
  → convert to grayscale (match training data)
  → YOLO11s (7 classes)  ──────────────┐
  → face crop → MediaPipe FaceLandmarker → EAR, MAR, head pitch
                                        │
                  Fusion + temporal rules (sliding windows, hysteresis)
                                        │
        Level 0 normal / Level 1 drowsy (beep A) / Level 2 asleep (beep B, louder, repeating)
                                        │
        Overlay on feed (boxes, state text, PERCLOS bar, FPS) + CSV event log
```

**Fusion rule [DEFAULT]:** an eye counts as *closed* when YOLO `eye_closed` conf ≥ 0.5 **OR** EAR < 0.75 × the driver's calibrated open-eye EAR (OR-logic favours recall, as you requested). Level 2 additionally requires the closure to persist (duration rule below), which cuts false alarms.

**Grayscale note:** the dataset is grayscale, the webcam is colour. Convert webcam frames to grayscale before inference (simplest), and/or add grayscale + brightness augmentation in training. Do not mix up the two or accuracy will drop in the live demo.

---

## 2. Calendar

| Wk | Dates | Capacity | Focus |
|---|---|---|---|
| 1 | 9–15 Nov | Full | P1 scoping/setup · P2 dataset audit starts |
| 2 | 16–22 Nov | Full | P2 labelling · P3 MediaPipe features |
| 3 | 23–29 Nov | Full | P3 done · P4 baseline training · P7 skeleton app |
| 4 | 30 Nov–6 Dec | Full (last full week) | **Milestone M1: first trained model + live skeleton running** |
| 5 | 7–13 Dec | Reduced (exams, assumed) | P4 tuning runs (unattended on Colab) · P5 rules |
| 6 | 14–20 Dec | Reduced (exams, assumed) | P5 finish · start P6 scripts |
| 7 | 21–27 Dec | Full | P6 evaluation · P7 full integration · P8 live tests |
| 8 | 28 Dec–3 Jan | Full | P8 final tests, report, slides, demo video |
| – | 4–7 Jan | Buffer | Only for slippage |

**Milestones**
- **M1 (end wk 4, 6 Dec):** YOLO11s trained once, live webcam loop with overlay + beep works end-to-end (even if inaccurate).
- **M2 (end wk 6, 20 Dec):** rules calibrated, ablation script ready.
- **M3 (end wk 7, 27 Dec):** feature-complete demo, evaluation numbers final.
- **M4 (3 Jan):** report, slides, demo video, tagged release.

Rule: **no new features after M3.**

---

## Phase 1 – Scoping, Setup & Decisions (Week 1) – Owner P / Backup J

**Tasks**
1. Freeze the decision table above in `docs/decisions.md` (use the template's decision log).
2. Create the GitHub repo with `LICENSE` (AGPL-3.0), `README.md`, `.gitignore` (exclude data, weights, runs), branch rule **[DEFAULT]:** `main` protected, work on feature branches, PR reviewed by the other person.
3. Environment: pin versions in `requirements.txt`. Check that PyTorch, Ultralytics, MediaPipe and OpenCV install on the chosen Python **before** anything else (Q8). Use the same Python minor version locally and on Colab where possible.
4. Short literature note (**not** 8 papers; 3–5 pages total): PERCLOS, EAR, MAR, yawning detection, YOLO drowsiness papers. Goal: justify thresholds.
5. Set up experiment tracking: Ultralytics built-in logging + TensorBoard or W&B (free).
6. Agree weekly sync **[DEFAULT]:** two 30-min calls per week on Discord (template said "NAN").

**Repo layout**
```
awaken/
├── data/            # NOT in git (raw/, processed/, splits/)
├── configs/         # dataset.yaml, thresholds.yaml
├── src/
│   ├── detect/      # YOLO inference wrapper
│   ├── features/    # MediaPipe EAR/MAR/pitch
│   ├── rules/       # sliding windows, PERCLOS, levels
│   ├── alarm/       # beep + overlay
│   └── app.py       # real-time loop
├── notebooks/       # Colab: data prep, training, evaluation
├── tests/
├── docs/
└── README.md
```

**Deliverables:** repo + environment working on both machines, decisions doc, short literature note.
**Exit criteria:** both teammates can run `python -c "import ultralytics, mediapipe, cv2, torch"` locally and on Colab.

---

## Phase 2 – Dataset Audit & Labelling (Week 1–2) – Owner J / Backup P

The dataset is already downloaded, but **its format has not been verified** (I could not open the page from this session). The whole plan depends on what is inside, so the first task is an audit.

**Day-1/2 audit checklist**
- Number of images, and per class if labelled.
- Label type: **bounding boxes (YOLO/COCO/VOC)** vs **image-level class labels** vs none.
- Which classes exist (does it contain closed eyes, yawns, head-down, shoulders?).
- Are subject/person IDs recoverable (filenames, folders, metadata)?
- Resolution, grayscale confirmation, is the **whole upper body/shoulders visible** or face-only (needed for `posture_alert`).
- Licence text (template says checked; record it in `DATA_SOURCES.md`).

**Decision gate (end of week 1)**

| Audit result | Path |
|---|---|
| Already has boxes for most of the 7 classes | Convert to YOLO format, fix gaps. Skip auto-labelling |
| Image-level labels only (most likely) | **Auto-label with MediaPipe** (see below) |
| No usable labels | Auto-label + larger manual review; consider dropping `posture_alert` and `head_down` |

**Auto-labelling pipeline [DEFAULT] (keeps manual work small, exams are coming):**
1. Run MediaPipe FaceLandmarker on every image → face box, left/right eye boxes, mouth box from landmark extremes (+ ~15 % margin).
2. Assign classes with rules + the dataset's image-level label: `eye_open`/`eye_closed` from EAR threshold, `yawn`/`mouth_closed` from MAR threshold, `head_down` from head pitch.
3. `posture_alert`: needs a definition first (Q5). If shoulders are visible, a MediaPipe Pose-based rule (shoulder line / slouch) can generate it.
4. **Manually review a sample** (≈ 300–500 images, stratified by class) with **Label Studio (local, free)** **[DEFAULT, template left annotator/tool blank]**. Fix errors, measure auto-label accuracy, and fix rules where they fail.
5. Convert to YOLO txt (`class x_center y_center w h`, normalised) and write `dataset.yaml`.

**Annotation rules (from the template, kept verbatim in `docs/annotation_guide.md`)**
- **Yawn:** maximal downward jaw extension with a distinct widening of the mouth cavity, often showing tongue/throat.
- **Talking (→ `mouth_closed` or ignore):** lip movement mostly horizontal/mild vertical, teeth relatively close, jaw does not lock at maximum extension.
- **Laughing (→ not yawn):** mouth corners pull backward and upward, not strictly downward.
- **To add (open Qs):** one box per eye or both eyes together; what counts as `eye_closed` (≥ 80 % lid closure?); definition of `head_down` and `posture_alert`.

**Split (70/15/15):** split **by subject** if IDs exist, otherwise by sequence/session, otherwise random **and state this as a limitation** (random splitting of frames from the same person inflates test scores).
**Class balance:** count boxes per class; if `eye_closed`/`yawn` are rare, oversample and use class-aware augmentation.
**Augmentation:** brightness/contrast, blur, small rotation/scale, mosaic. **Horizontal flip is safe** because classes are not left/right-specific.
**Data handling on Colab:** zip the processed dataset on Drive, copy it to Colab's local disk at session start (reading thousands of small files directly from Drive is very slow). Version data as `dataset_v1`, `v2` ... in Drive with a `CHANGELOG.txt`.

**Deliverables:** `dataset.yaml`, labelled/split data on Drive, class statistics, annotation guide, `DATA_SOURCES.md`.

---

## Phase 3 – MediaPipe Feature Module (Week 2–3) – Owner P / Backup J

**Tasks**
1. Landmark extractor: **MediaPipe FaceLandmarker** (dlib dropped **[DEFAULT]**: harder to install, slower, no benefit for this project). Run on the YOLO face crop when available, else on the full frame.
2. **EAR** = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖) per eye, averaged.
3. **MAR** = mouth opening / mouth width.
4. **Head pose:** `solvePnP` → pitch / yaw / roll; use pitch for nodding.
5. Smoothing (EMA or median over 3–5 frames), missing-face handling.
6. **Per-driver calibration:** first 20–30 s with eyes open and neutral mouth → store baseline EAR/MAR/pitch.
7. Unit tests with a few saved images/frames.

**Deliverables:** `src/features/` with tests; a notebook plotting EAR/MAR/pitch on a short test clip.
**Reuse:** the same module generates the auto-labels in Phase 2.

---

## Phase 4 – YOLO11s Training (Week 3–5) – Owner J / Backup P

**Runs (track every run):**
1. **Baseline:** `yolo11s.pt` pretrained, `imgsz=640`, 100 epochs, defaults, `patience=20`.
2. **Experiments (only as compute/time allows):** `imgsz` 416 vs 640; augmentation strength; freeze first N layers vs full fine-tune; class-imbalance handling.
3. Optional `yolo tune` only if Colab time is spare.
4. **Select the checkpoint by recall on `eye_closed` and `yawn` and overall mAP@0.5**, not mAP alone.
5. Export best weights to **ONNX** (and PyTorch `.pt`) and measure ms/frame on the demo laptop.

**Colab survival rules**
- Save checkpoints to Drive every epoch; use `resume=True` after disconnects.
- Long runs go in weeks 5–6 while exams limit active time; start them in the evening and check results later.
- Keep a `train.ipynb` that runs top-to-bottom from a fresh runtime.

**Deliverables:** `best.pt`, `best.onnx`, training curves, run table (settings → metrics).
**Milestone M1 (6 Dec):** at least one trained model integrated into the live loop.

---

## Phase 5 – Fusion & Temporal Rules (Week 4–6) – Owner J / Backup P

Per-frame outputs → time series → rule engine → 2 alert levels.

**Starting thresholds (all in `configs/thresholds.yaml`, tune in Phase 6) [DEFAULT]:**

| Cue | Level 1 – Drowsy | Level 2 – Asleep |
|---|---|---|
| Continuous eye closure | ≥ 1.0 s | ≥ 2.0 s |
| PERCLOS (60 s window, eyes ≥ 80 % closed) | ≥ 0.20 | ≥ 0.40 |
| Yawns | ≥ 3 in 5 min (each yawn lasting ≥ 1.5 s) | – |
| `head_down` / pitch beyond baseline | ≥ 3 nods in 60 s, or held ≥ 2 s | held ≥ 3 s **with** eyes closed |
| `posture_alert` lost (slumped) | held ≥ 3 s adds to Level 1 score | – |

Blink reference: a normal blink lasts ~0.1–0.4 s, so 1.0 s is a clear "not a blink".

**Logic**
- Rolling windows: short (≈ 3–5 s) for closure/nod, long (60 s) for PERCLOS, 5 min for yawns.
- **Hysteresis:** raise a level quickly, drop it only after ~3 s of normal state (avoids flicker).
- **Cooldown:** Level 1 beep at most once per 10 s **[DEFAULT]**; Level 2 repeats until eyes open.
- **No face for N seconds [DEFAULT, see Q10]:** keep searching; after 5 s show "Face not found" on the overlay (no drowsiness alarm); never silently report "alert" when no face is visible.
- **Event log (CSV):** timestamp, level, triggering cue, measured values (e.g. closure 2.3 s, PERCLOS 0.31), thresholds. No frames/video saved.

**Deliverables:** `src/rules/` with unit tests on synthetic time series; `thresholds.yaml`.

---

## Phase 6 – Evaluation (Week 6–7) – Owner P / Backup J

**Metrics and targets**

| Metric | Target | Note |
|---|---|---|
| mAP@0.5 | **≥ 0.80** **[DEFAULT]**, template said "sure" | on held-out test split |
| Recall `eye_closed`, `yawn` | **≥ 0.90** **[DEFAULT]**, template left "?" | at deployed conf threshold |
| Event-level recall (simulated drowsy events detected) | ≥ 0.90 | from live tests |
| False alarms / hour | ≤ 40 (template) | **I suggest tightening to ≤ 10/h**, see Q3 |
| FPS | ≥ 30 | end-to-end, laptop webcam |
| Latency | ≤ 10 ms (template) | see Q3: realistic only for YOLO inference on a GPU |
| Model size | ≤ 3000 MB (template) | YOLO11s is only ~20 MB; likely a typo, see Q3 |

**Experiments**
1. Per-class precision/recall, confusion matrix, PR curves on the test split.
2. **Ablation:** YOLO only vs YOLO + MediaPipe fusion (event recall and false alarms/hour).
3. Threshold sweep on a validation set → choose thresholds that favour recall with ≤ target false alarms.
4. Speed: ms per stage (capture, YOLO, MediaPipe, rules, drawing), CPU/GPU load.
5. Error analysis with example images (what the model gets wrong: partial lid closure, talking vs yawn, head angles).
6. Fairness check **[DEFAULT, Q9]:** test with as many different people as available (skin tone, age, gender, headscarf, facial hair) and report results per person in the limitations section.

**Deliverables:** results tables/plots in `docs/results.md`, final threshold file, final model.
**Loop:** if recall or false-alarm targets fail → retrain (P4) or retune rules (P5) once; no further loops after M3.

---

## Phase 7 – Real-Time App (skeleton Week 3–4, full integration Week 6–7) – Owner P / Backup J

**Why split:** building the skeleton early (with the pretrained COCO model and fake detections) finds FPS and integration problems before exams.

**Skeleton (week 3–4)**
- OpenCV capture thread + inference thread with a small queue (drop old frames, never lag).
- Overlay: boxes, state text, PERCLOS bar, FPS, alert level banner.
- Beep: two distinct sounds (e.g. `winsound` on Windows or `simpleaudio`/`pygame`/`sounddevice` cross-platform) played on a separate thread so it never blocks video.
- Config file loading, CSV logger, keyboard shortcuts (`q` quit, `c` recalibrate).

**Full integration (week 6–7)**
- Plug in `best.onnx`/`best.pt`, MediaPipe features, and the rule engine.
- Calibration screen on start (20–30 s).
- Measure FPS; if < 30: smaller `imgsz`, run MediaPipe only on the face crop, run MediaPipe every 2nd frame, ONNX Runtime/OpenVINO, FP16 if a GPU is present.
- Offline check: run with Wi-Fi off (assume no internet at demo).
- Package: `python app.py --config configs/thresholds.yaml`, plus a one-page "how to run" in README.

**Deliverables:** working demo script, performance table.

---

## Phase 8 – Testing, Documentation & Delivery (Week 7–8) – Owner P / J

**Tasks**
1. **Live tests (safe, seated, never while driving):** sit in front of the laptop; simulate long eye closure, microsleep, yawns (fake and real), head nods, slumping. Record each attempt and whether the system reacted. Test different lighting levels within the "normal indoor" range, and at least 3–4 different people.
2. Collect false alarms during ~30 minutes of normal use (reading, talking, drinking, looking at a screen) and compute false alarms/hour.
3. Fix bugs, final threshold tuning (freeze at M3).
4. **Report** (method, data and its limits, results, limitations, ethics/privacy: on-device, no video stored, volunteers informed verbally).
5. Slides + **demo video** (record a backup in case the live demo fails).
6. README, model card (data, intended use, limits: grayscale training data, no night/IR, no glasses, not a safety-certified product), tagged GitHub release `v1.0`.

**Deliverables:** report, slides, demo video, code release, trained model.

---

## Responsibilities (from the template)

| Area | Owner | Backup |
|---|---|---|
| Literature, annotation/QA, MediaPipe features, real-time app/alarm, evaluation/plots | **P** | J |
| Dataset/licences, YOLO training, temporal/decision logic, report/slides, repo/env | **J** | P |

**Integration contract (agree in week 1):** J's detector returns `{class, conf, box}`; P's features return `{ear, mar, pitch}`; J's rule engine consumes both per frame with a timestamp. Define these two function signatures first so you can work in parallel.

---

## Risks & Mitigations

| Risk | L / I | Mitigation |
|---|---|---|
| Dataset has no bounding boxes → manual annotation of thousands of images | M / H | MediaPipe auto-labelling + small manual review; decide at week-1 gate; drop `posture_alert`/`head_down` first if needed |
| Exams in weeks 5–6 (length unknown) | H / H | Front-load to M1 (6 Dec); unattended Colab runs; reduced scope in weeks 5–6 |
| Too many false alarms | H / H | Duration rules, hysteresis, OR-fusion only at Level 1, calibration, threshold sweep |
| 30 FPS / 10 ms on laptop not met | M / M | Skeleton early, ONNX, smaller imgsz, MediaPipe on crop, frame skipping |
| Grayscale training data vs colour webcam | M / M | Convert webcam to grayscale; grayscale augmentation; test early (M1) |
| Colab disconnects / GPU quota | L–M / H | Per-epoch checkpoints to Drive, `resume=True`, local-disk data copy |
| Python 3.14 package compatibility | M / M | Verify in week 1; fall back to 3.11/3.12 |
| Frame-level split leaks subjects → inflated scores | M / M | Split by subject/session if possible, else disclose |
| Fairness (skin tone, headscarf, age) not covered by data | M / M | Test on varied volunteers; report as limitation |
| Real-world demo fails on the day | M / H | Backup demo video; test on the exact laptop and room |

---

## Definition of Done

- [ ] Dataset audited, labelled (auto + reviewed), split, documented
- [ ] YOLO11s trained; per-class recall for `eye_closed`/`yawn` meets target
- [ ] MediaPipe EAR/MAR/pitch module with calibration
- [ ] 2-level rule engine with logging, tuned thresholds in config
- [ ] Ablation (YOLO only vs fused) done
- [ ] Live demo ≥ 30 FPS (or documented measured FPS) offline on the target laptop
- [ ] Live tests with several people, false alarms/hour measured
- [ ] Report, slides, demo video, README, model card, `v1.0` release (AGPL-3.0 licence file present)
