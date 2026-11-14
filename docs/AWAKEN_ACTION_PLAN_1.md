# Awaken! – Action Plan for J and P (revision 2: Roboflow dataset)

Status: **working plan, update it at each weekly sync.**
Written Sat 14 Nov 2026 (end of week 1). Deadline about 3 Jan 2027 (buffer 4–7 Jan).

**Built from:** `yolo_drowsiness_project_plan.md`, the answered `teammate_questions_template.md`, P's `features.py` and `featuresLive.py`, and the new Roboflow dataset. **Tags:** `[P]` / `[J]` = owner, the other person reviews. **F-n** = finding about P's code. **Q-n** = open question.

## What changed in revision 2

- **The dataset is now the Roboflow Universe project** `yolov5-7gacb/drowsiness-detection-cntmz-izq7w`, downloaded straight into Colab with the Roboflow package. It is already **split and in YOLO format**, so the whole "build the split" and "auto-label everything" work is replaced by an **audit, a class decision and a leakage check** (steps 6–14).
- **I could not open that dataset page** (Roboflow blocks automated readers). The project page of the original this seems to be forked from says it is **images of one person acting "drowsy" and "awake" in a car**. Your fork may differ. So **nothing about classes, counts or split quality is assumed**: step 7 measures it, and gate G1/G2 decides the path.
- **Biggest new risk: a one-person dataset** (if the audit confirms it). Validation and test numbers will look excellent and say little about real people. The plan now makes an **independent test set from your own webcam clips** a required step (step 11), not an option.
- **Grayscale rule removed.** The earlier "convert webcam frames to grayscale" came from the NTHU dataset. Roboflow images are probably colour; the audit tells you. The app must use **the same preprocessing as the training images** (colour vs gray, and stretch vs letterbox resize).
- P's two scripts are still used: for the live app, for the feature cross-check, and as **Plan C** (relabel with MediaPipe if the Roboflow classes are too coarse).
- The earlier NTHU / Hugging Face work (`splitting_dataset_plan.md`, `01_build_split.py`, `audit_dataset.py`) is **kept as Plan B** (extra test data), not deleted.

---

## 1. Where we are

| Done | Unknown until the audit |
|---|---|
| Project plan, diagram, question templates | Classes of the Roboflow dataset (2 state classes, or eye/mouth classes?) |
| P: `features.py` (EAR, MAR, pitch, boxes, CSV, YOLO-line helper) and `featuresLive.py` (webcam, smoothing, calibration, log) | Number of images, per-class counts, version number to download |
| NTHU dataset audited (4 subjects, folder labels only) and split script written (Plan B) | How many different people appear; whether valid/test share images with train |
| `audit_yolo_dataset.py` written for this dataset (tested on fake data) | Image size, colour vs gray, resize method, augmentation copies |
| | Licence of the Roboflow dataset |
| | Python 3.14.7 compatibility (Q-8) |

---

## 2. Decisions locked and decisions pending

| Area | Decision |
|---|---|
| Model | YOLO11s (Ultralytics, AGPL-3.0 accepted, repo is open source) |
| Design | Option C: YOLO detects states, MediaPipe gives EAR / MAR / pitch, threshold rules fuse both |
| Out of scope | distraction, phone, seatbelt, night/IR |
| Alerts | 2 levels (drowsy, asleep), beep, overlay on the camera feed |
| Logging | trigger events and the values that crossed the threshold; **no video stored** |
| Data | Roboflow `yolov5-7gacb/drowsiness-detection-cntmz-izq7w`, YOLO export, stored on Drive after download |
| Training / demo | Google Colab / laptop webcam, offline |
| **Pending (G1)** | **Class set.** The 7 planned classes only if the dataset has them. **[DEFAULT]** use the dataset's own classes (likely `awake`/`drowsy`) and let MediaPipe supply eye / yawn / nod cues |
| **Pending (G2)** | **Split.** Use Roboflow's train/valid/test only if the leakage check passes. **[DEFAULT]** accept it for model selection, but report real performance on **your own clips** |
| **Pending** | Input preprocessing = whatever the training images are (colour/gray, resize) |
| Targets | recall of the drowsy state first, false alarms second; 30 FPS end to end (see `open_questions.md` Q3) |

---

## 3. Review of P's two scripts (still valid)

**Good base:** standard EAR/MAR landmark sets, per-eye and mouth boxes, a YOLO-line converter, a CSV export that keeps relative paths, EMA smoothing that resets on a lost face, median-based calibration, per-frame CSV log with timing.

| ID | Finding | Action |
|---|---|---|
| F-1 | **Head-pose sign.** I tested the pose model on synthetic data: magnitudes are right, but P's 3D model (y down, negative z) is a mirror image, so pitch/yaw signs come out opposite to a right-handed convention. | Step 4: nod test on the webcam, write down whether pitch goes up or down when looking down. Define the nod rule by that sign |
| F-2 | `RunningMode.IMAGE` runs full detection every frame. | Fine for offline checks. In the app (step 22) try `RunningMode.VIDEO` (tracking between frames); check the API for your installed MediaPipe version |
| F-3 | `featuresLive.py` default is colour, `--gray` optional. | Step 4 measures both; after the audit, **match the dataset**. No grayscale unless the training images are gray |
| F-4 | `from features import ...` works only when both files sit in one folder. | Keep them together in `src/features/`, or use a package import in the app |
| F-5 | `face_landmarker.task` (3.6 MB) is a binary model file. | Decide: commit it, or git-ignore it and document the download (MediaPipe Face Landmarker model page) |
| F-6 | One 15 % margin is used for all boxes. | Only matters for Plan C relabelling; check 50 overlays then |
| F-7 | CSV has no class/split columns. | Join with the Roboflow file names (relative path is kept) |
| F-8 | `--all` is single-threaded. | Time 200 images first (step 10) |

---

## 4. Working guidelines

1. **Audit before you trust.** Never train on a downloaded dataset you have not looked at (report + contact sheet, step 7).
2. **API key = secret.** Store `ROBOFLOW_API_KEY` in Colab Secrets. Never paste it into a notebook cell output, the repo, or WhatsApp/Discord.
3. **Download once, then freeze.** Copy the raw download to Drive as `dataset_rf_raw.zip`. Roboflow versions can change or disappear; your training must use the frozen copy, with the **version number written down**.
4. **No test leakage.** Pick checkpoints and thresholds on **valid**, never on test. Your own clips are used once, at the end.
5. **Check what Roboflow baked in.** A downloaded version may contain augmented copies (file names with `.rf.`) and a fixed resize. Do not stack strong Ultralytics augmentation on top of baked augmentation without checking.
6. **Scale in steps:** 1 image, 50, 500, all.
7. **Colab habits:** unzip the dataset to `/content`, save checkpoints to Drive every epoch, notebooks must run top to bottom on a fresh runtime.
8. **Git:** feature branches, a pull request for every merge, reviewed by the other person. Never commit data, weights, runs, or keys.
9. **Decisions in `NOTES.md`:** one line each (date, decision, who, why).
10. **Exam weeks (5–6, assumed):** only unattended Colab runs and small tasks. Do everything risky before 6 Dec.
11. **Stay in scope:** no new classes or features after M3 (27 Dec).

---

## 5. Calendar

Labelling work shrinks (data is already annotated), so **training can start in week 2–3**.

| Week | Dates | Capacity | Focus |
|---|---|---|---|
| 1 | 9–15 Nov | Full | Setup, features scripts (done mostly) |
| 2 | 16–22 Nov | Full | Download, audit, gates G1/G2, own clips, first training |
| 3 | 23–29 Nov | Full | Freeze `dataset_v1`, baseline results, app skeleton |
| 4 | 30 Nov–6 Dec | Full (last) | **M1 (6 Dec):** trained model running in the live loop (can come earlier) |
| 5 | 7–13 Dec | Reduced | Training experiments (unattended), rules |
| 6 | 14–20 Dec | Reduced | Rules done, **M2 (20 Dec):** thresholds calibrated |
| 7 | 21–27 Dec | Full | Evaluation, full integration, live tests, **M3 (27 Dec):** feature freeze |
| 8 | 28 Dec–3 Jan | Full | Final tests, report, slides, video, release (**M4, 3 Jan**) |

---

## 6. Action steps

Do the steps in order inside each phase. **Done when** = acceptance check.

### A. Repository and environment (week 1–2)

1. **[J]** Create the repo `awaken` with `LICENSE` (AGPL-3.0), `README.md`, `NOTES.md`, `.gitignore` (data, weights, runs, `*.zip`, `.env`), folders:
   ```
   awaken/
   ├── data/            # not in git
   ├── configs/         # classes.yaml, thresholds.yaml, data.yaml (frozen copy)
   ├── scripts/         # audit_yolo_dataset.py, 02_remap_classes.py (if needed)
   ├── src/features/    # features.py, featuresLive.py, (face_landmarker.task: F-5)
   ├── src/detect/  src/rules/  src/alarm/  src/app.py
   ├── notebooks/       # colab: download+audit, train, evaluate
   ├── tests/  docs/
   ```
   *Done when:* both can clone and push; plan files and scripts are committed.
2. **[P]** Move `features.py` and `featuresLive.py` into `src/features/` with a short README and one fixed-image test (EAR/MAR within a range).
   *Done when:* `python src/features/features.py --model ... --image ...` runs from the repo root.
3. **[J]** `requirements.txt`; check that `torch`, `ultralytics`, `mediapipe`, `opencv-python`, `roboflow` install together on your Python (3.14.7 unverified) **and** in Colab. If anything fails use Python 3.11/3.12.
   *Done when:* `python -c "import torch, ultralytics, mediapipe, cv2"` works locally and in Colab.
4. **[P]** Webcam sanity (30 min): run `featuresLive.py` with and without `--gray`; note FPS and `infer_ms`; do the **nod test** (pitch direction); run a calibration (`c`) and look at the baseline; save one CSV.
   *Done when:* `NOTES.md` has the colour vs gray result, FPS, the pitch sign, and your own baseline EAR/MAR/pitch.
5. **[J]** `docs/contracts.md`: detector returns `{class, conf, box}`, features return `{ear, mar, pitch, boxes}`, rule engine consumes both per frame with a timestamp.
   *Done when:* P agrees in writing.

### B. Dataset: download, audit, decide (week 2)

6. **[J]** Get the download code from the dataset page (**Download Dataset**, format **YOLO11**; if the list only shows YOLOv8, take that, the text format is the same; **show download code**). Check the **version number** on the Versions tab. Put the API key in Colab Secrets. Then:
   ```python
   !pip install -q roboflow
   from google.colab import userdata
   from roboflow import Roboflow
   rf = Roboflow(api_key=userdata.get("ROBOFLOW_API_KEY"))
   project = rf.workspace("yolov5-7gacb").project("drowsiness-detection-cntmz-izq7w")
   dataset = project.version(1).download("yolov11", location="/content/rf_data")  # 1 = replace with the real version
   ```
   (Workspace and project names come from your URL. I could not test this call, so prefer the exact snippet shown on the page.) Zip the result to Drive as `dataset_rf_raw_v<N>.zip`.
   *Done when:* `/content/rf_data` contains `data.yaml` and `train`, `valid`, `test` folders; the zip is on Drive.
7. **[J]** Run the audit and send P the report and the picture:
   ```
   python scripts/audit_yolo_dataset.py /content/rf_data --out audit_yolo_report.txt --sheet boxes_sheet.jpg
   ```
   *Done when:* the report and `boxes_sheet.jpg` are shared. They show classes, boxes per class and split, image size, colour/gray, augmentation copies, **overlap and near-duplicates between splits**.
8. **[P + J]** **Gate G1: class set.** Look at `boxes_sheet.jpg` and the class table:

   | The dataset has | Decision |
   |---|---|
   | eye / mouth / head classes close to the plan | Map names to the project classes (step 12) and train on them |
   | only state classes (`awake` / `drowsy`) **[likely]** | Train YOLO on those; eye closure, yawn and nod come from **MediaPipe** (EAR / MAR / pitch) in the rule engine. Plan classes become `awake`, `drowsy` |
   | only one class or boxes that don't match faces | Stop. Choose another dataset or use **Plan C** (relabel with MediaPipe) |

   Also check: are the boxes on the whole face, or on the eyes/mouth? How many different people are visible?
   *Done when:* the class set is written in `NOTES.md` and `configs/classes.yaml`.
9. **[J]** **Gate G2: split quality.** Read sections 4 and 5 of the audit.
   - Shared source names between splits **> 0**, or near-duplicates of valid/test in train **> ~30 %**: the splits are leaking. Valid/test are only usable for rough monitoring; the **real test is your own clips** (step 11). Optionally re-split by source name (if frame sequences are visible in the names).
   - Otherwise accept Roboflow's split.
   - If the dataset shows **one person only**, write that as the first limitation in the report.
   *Done when:* the decision and the numbers are in `NOTES.md`.
10. **[P]** MediaPipe cross-check (also tests F-6/F-8): copy about 200 random images from train and 200 from valid into a folder and run
    ```
    python src/features/features.py --model src/features/face_landmarker.task --folder <folder> --all --csv features_check.csv
    ```
    Report: face-found rate per class, EAR / MAR / pitch distributions per class, and whether the dataset's "drowsy" really differs on those cues.
    *Done when:* a short table is in `docs/`; if the cues don't separate the classes, label noise is likely and G1 is revisited.
11. **[P + J]** **Record your own clips** (the independent test set; start now, not in week 7). P and J, laptop webcam, same room as the demo, 3–5 min each: normal, eyes closed 3 s and 5 s, slow blinks, yawn, nod, talking, laughing, looking at the screen; later also with a second person and with/without glasses if possible. Read a **script with timestamps** aloud ("eyes closed now ... open") so you get event labels without drawing boxes: save a CSV `time_start, time_end, event`.
    *Done when:* clips (kept on Drive, not in git) and their event CSVs exist.
12. **[J]** Class mapping `configs/classes.yaml` (dataset name to project class, or merged/dropped classes). If any label must change, write `scripts/02_remap_classes.py` (reads the YOLO txt files, rewrites class ids, writes a new folder; never edits the raw copy).
    *Done when:* new `data.yaml` loads in Ultralytics; counts per class printed.
13. **[J]** *(Optional, if time)* **Plan B cross-dataset test:** run the trained model on NTHU frames of the held-out subject and read "drowsy detected" against its drowsy/notdrowsy folder label. Domain gap is large (gray/IR vs colour), so treat it as a stress test only.
14. **[J]** **Freeze:** copy `data.yaml`, `classes.yaml`, the audit report and the commands to `docs/dataset_v1/`; zip the final folder to Drive as `dataset_v1.zip`; tag the repo `dataset_v1`.
    *Done when:* both can unzip `dataset_v1.zip` in a fresh Colab and see the same counts.

### C. YOLO training (week 2–5)

15. **[J]** Colab `train.ipynb`: copy and unzip `dataset_v1.zip` to `/content`; baseline `yolo11s.pt`, `imgsz` equal to the dataset's resize (audit section 3), 100 epochs, `patience=20`, project folder on Drive, resume instructions.
    *Done when:* training finishes on a fresh runtime; `best.pt` on Drive.
16. **[J]** Read validation results per class (precision, recall, confusion matrix, PR curve) into `docs/results.md`; look at the confusion matrix for `awake` vs `drowsy` mix-ups.
17. **[P]** Error analysis: 30 wrong predictions per class; decide whether the cause is labels, data (one person!) or model.
    *Done when:* three bullets with actions.
18. **[J]** Experiments, unattended in weeks 5–6, only as time allows: `imgsz`, augmentation strength (watch baked-in augmentation, guideline 5), class balance, freezing layers; pick the checkpoint **on valid**.
19. **[J]** Export `best.pt` to ONNX; **[P]** time it on the demo laptop.
    *Done when:* `best.onnx` and a timing table exist.
20. **[P + J] First reality check on own clips** (informal, not the final test): run the model on the clips from step 11 and look at mistakes. If it fails on your faces, the plan is: **fine-tune on your own frames** (needs a few hundred labelled images of P and J: use Roboflow's labelling tool or Label Studio) and keep a separate untouched test clip.
    *Done when:* a one-page note on how well it works on P and J.

### D. Real-time app (skeleton week 3–4, full week 6–7)

21. **[P]** `src/app.py`: capture thread + inference thread, overlay, two beep sounds on a separate thread, config file, `q` to quit. Start from `featuresLive.py` (calibration, EMA, log).
    *Done when:* overlay and both beeps work with fake inputs; FPS recorded.
22. **[P]** Calibration screen (20–30 s), baselines stored; try `RunningMode.VIDEO` (F-2).
    *Done when:* baselines are logged and survive until the app closes.
23. **[P]** Plug in the YOLO weights (`.pt` first, `.onnx` later) with **exactly the training preprocessing** (colour/gray; **stretch vs letterbox**: if the Roboflow version used "stretch to square", the live frame must be resized the same way or accuracy drops).
    *Done when:* live detection runs. **M1 reached.**
24. **[P]** Performance pass: smaller `imgsz`, MediaPipe on the face crop, every 2nd frame, ONNX Runtime; per-stage timing.
    *Done when:* a table of ms per stage and end-to-end FPS exists.

### E. Fusion and rules (week 4–6)

25. **[J]** `src/rules/`: per-frame inputs (YOLO state + EAR ratio + MAR + pitch delta) become time series; compute continuous eye-closure time, PERCLOS (60 s), yawn count (5 min), nod count; start from the Phase 5 thresholds in `yolo_drowsiness_project_plan.md`.
    *Done when:* unit tests pass on synthetic series (2 s closure triggers level 2; a 0.3 s blink does not).
26. **[J]** Fusion: **drowsy** if YOLO says `drowsy` OR the MediaPipe cues cross their thresholds (OR for level 1; level 2 additionally needs the duration). With only state classes, **MediaPipe provides the eye/yawn/nod evidence and YOLO is the second opinion**.
    *Done when:* unit tests cover the OR and duration logic.
27. **[J]** Hysteresis (raise fast, drop after about 3 s normal), beep cooldown (10 s at level 1; level 2 repeats until eyes open), "Face not found" after 5 s without alarm.
    *Done when:* tests cover flicker and no-face cases.
28. **[J]** Event log (CSV): timestamp, level, triggering cue, measured values, thresholds. No frames saved. Thresholds in `configs/thresholds.yaml`.
    *Done when:* a live run produces a log with no image data.
29. **[P]** Wire the rule engine into the app. **M2 (20 Dec):** thresholds calibrated (step 32).

### F. Evaluation (week 6–7)

30. **[P]** Evaluate the Roboflow **test split once**: per-class precision/recall, confusion matrix. State plainly if the audit showed leakage or a single person.
31. **[P]** Evaluate on **your own clips** with the event CSVs: event recall (did an alarm fire during each "eyes closed 3 s"?), false alarms per hour on the normal/talking parts. **This is the headline result.**
32. **[J]** Threshold sweep on valid (and on a *separate* part of your clips, never the final test clip) to favour recall with few false alarms.
    *Done when:* final `thresholds.yaml` committed with the reason.
33. **[P]** Ablation: YOLO only vs MediaPipe only vs fusion, measured as event recall and false alarms/hour on your clips.
34. **[J]** Fairness / limitation note: how many different people were in the training data and the test clips; results per person.

### G. Testing and delivery (week 7–8)

35. **[P + J]** Live tests seated in front of the laptop (never while driving): long closure, microsleep, yawns, nods, talking, lighting changes, 3–4 people; 30 minutes of normal use for false alarms/hour.
    *Done when:* a test log with pass/fail per case exists.
36. **[J]** Fix bugs; freeze thresholds at M3 (27 Dec).
37. **[J]** Report (method, data and its limits, results, limitations, ethics/privacy) and slides.
38. **[P]** Demo video plus a backup recording; run the demo once offline (Wi-Fi off) on the exact laptop.
39. **[J]** README, model card (dataset source and licence, number of people, no night, not a safety product), tag `v1.0`.
    *Done when:* a fresh clone follows the README and starts the app.

---

## 7. Plan C: relabel with MediaPipe (only if G1 says the classes are too coarse)

Use P's `features.py` to make the finer classes yourself:

| Class | Box from `features.csv` | Rule (tune on train/val only) |
|---|---|---|
| `face` | `face_x1..face_y2` | face found, head not down |
| `eye_open` / `eye_closed` | `leye_*`, `reye_*` | EAR ratio vs per-person baseline (**[DEFAULT]** ≥ 0.85 open, < 0.75 closed) |
| `yawn` / `mouth_closed` | `mouth_*` | MAR above/below threshold (**[DEFAULT]** about 0.5, tune from histograms) |
| `head_down` | `face_*` | pitch delta beyond **[DEFAULT]** 15–20° in the direction found in step 4 |

Rules: per-person baselines; **exclude uncertain frames** (between thresholds) instead of guessing; skip images without a face and count them; write `labels_report.csv`; draw 50 label files back on the images before using them; review 300–500 images by hand. Do this only if needed: it costs about a week.

---

## 8. Commands (Colab cheat sheet)

```bash
pip install -q ultralytics mediapipe opencv-python pillow numpy roboflow
# audit
python scripts/audit_yolo_dataset.py /content/rf_data --out audit_yolo_report.txt --sheet boxes_sheet.jpg
# features check (200 images)
python src/features/features.py --model src/features/face_landmarker.task --folder /content/check --all --csv /content/features_check.csv
# freeze
cd /content && zip -qr dataset_v1.zip dataset_v1 && cp dataset_v1.zip /content/drive/MyDrive/awaken/
# train
yolo detect train model=yolo11s.pt data=/content/dataset_v1/data.yaml imgsz=640 epochs=100 patience=20 \
     project=/content/drive/MyDrive/awaken/runs name=v1
# resume after a disconnect
yolo detect train resume model=/content/drive/MyDrive/awaken/runs/v1/weights/last.pt
```

---

## 9. Gates (definition of done)

- [ ] **G1 (step 8):** class set decided from the audit
- [ ] **G2 (step 9):** split quality judged (leakage numbers, number of people)
- [ ] **Step 11:** own test clips recorded with event CSVs
- [ ] **Step 14:** `dataset_v1` frozen on Drive, tag created
- [ ] **M1 (6 Dec):** model running in the live loop
- [ ] **M2 (20 Dec):** rules calibrated, `thresholds.yaml` final
- [ ] **M3 (27 Dec):** results on own clips final, feature freeze
- [ ] **M4 (3 Jan):** report, slides, video, `v1.0` release

---

## 10. Open questions

| # | Question | Default used |
|---|---|---|
| Q-1 | **What is the exact version number, class list and licence of the Roboflow dataset?** (I could not read the page.) Open the page, look at **Versions**, **Classes** and **License**, and paste them to me. | Audit step 7 decides |
| Q-2 | Is this project a fork of the "Drowsiness Detection" dataset by Augmented Startups (one person in a car)? If yes, are you comfortable that the model will have seen **one face** only? | Own clips + fine-tuning on P and J (steps 11, 20) |
| Q-3 | **Capacity.** `PROJECT_PLAN.md` (Scribbl Week) lists the same two people and the same 2 months. Do both projects run in parallel? If yes, agree the weekly hours for Awaken. | Full capacity weeks 1–4 and 7–8 |
| Q-4 | **Workload balance.** P owns features, QA, app, evaluation. I gave the class mapping, rules and thresholds to J. Agree? | As written |
| Q-5 | Do you want a **different dataset or a larger mix** (e.g. add NTHU frames for more people) if the Roboflow data turns out to be one person? | Keep Roboflow + own clips; NTHU only as a stress test |
| Q-6 | Exam dates (assumed weeks 5–6) and holidays between 21 Dec and 3 Jan. | Reduced capacity weeks 5–6 |
| Q-7 | Python 3.14.7 package compatibility. | Test in step 3, fall back to 3.11/3.12 |

---

## 11. Decision log

| Date | Decision | By | Reason |
|---|---|---|---|
| 14 Nov | Dataset changed from Hugging Face NTHU to Roboflow (already split, YOLO format) | P, J | Saves labelling time; NTHU kept as Plan B |
| | | | |
