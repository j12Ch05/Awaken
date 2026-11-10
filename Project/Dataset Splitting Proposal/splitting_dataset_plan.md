# Splitting Dataset Plan – Awaken! (Phase 2)

Based on the audit report of the dataset (`audit_report.txt`) and `01_build_split.py`.

---

## 1. What the audit found

| Item | Finding |
|---|---|
| Source | HuggingFace `chbh7051/driver-drowsiness-dataset` (looks like NTHU-DDD; confirm licence on the dataset page) |
| Size | 66,521 images, 2.8 GB, all `.jpg` |
| Image format | 640×480, saved as RGB but **grayscale content** (R=G=B in 300/300 sampled) |
| Labels | **Folder only:** `drowsy/` (36,030) and `notdrowsy/` (30,491). No boxes, no label files |
| File name pattern | `<subject>_<glasses\|noglasses>_<scenario>_<frame>_<label>.jpg` |
| Subjects | **Only 4:** 001 (19,016), 002 (15,134), 005 (13,087), 006 (2,763) |
| Scenarios | `sleepyCombination`, `slowBlinkWithNodding`, `yawning`, `nonsleepyCombination` |
| Glasses | about 50 % `glasses`, 50 % `noglasses` |
| Night / IR | not present |
| Not yet verified | whether shoulders are visible (needs `sample_sheet.jpg`), licence text |

### Consequences
1. A 70/15/15 split **by subject** is impossible with 4 subjects.
2. Consecutive frames are near-identical (frame numbers run into the thousands per video), so a random split leaks between train and test and gives falsely high scores.
3. The folder label means *drowsy / not drowsy per frame*. It does not say which cue is present, so classes such as `eye_closed`, `yawn`, `head_down` must be produced by auto-labelling (MediaPipe) and then reviewed.
4. Three training subjects is a small base for generalising to new faces, so own-recorded clips are needed (see section 6).

---

## 2. Definitions

- **Video (group):** one combination of `subject + glasses + scenario`, e.g. `002_glasses_yawning`. Frames of one video must never be spread across train and test without protection.
- **Block:** a run of consecutive frames (default 150 frames, ≈ 5 s at 30 fps).
- **Stride:** keep every N-th frame (default 5) to remove near-duplicates.
- **Gap:** frames discarded at block edges so neighbouring frames never sit in both train and val (default 15).

---

## 3. The split strategy

| Split | What goes in | Why |
|---|---|---|
| **Test** | **All frames of one held-out subject** (default: 005, ≈ 20 % of data) | Measures real cross-person generalisation. Never used for training, tuning or threshold selection |
| **Val** | ≈ 15 % of the 150-frame blocks of each video of the other subjects, with a 15-frame buffer at both edges | Used for early stopping, checkpoint choice, and rule thresholds |
| **Train** | All remaining frames (includes the small subject 006) | Learning |

**Resulting share (approx.):** train ≈ 55–60 %, val ≈ 10–15 %, test ≈ 20–25 %. The 70/15/15 target from the template cannot be met exactly; this is a deliberate, documented deviation.

**Alternatives considered**

| Option | Verdict |
|---|---|
| Random frame split 70/15/15 | **Rejected.** Near-identical neighbouring frames leak, scores become meaningless |
| Hold out subject 001 or 002 as test | Possible, but removes 23–29 % of all data from training |
| Hold out subject 006 as test | Only 4 % of data, too small and too little variety |
| Leave-one-subject-out cross-validation (4 folds) | Best statistically, but 4× the Colab training time. Optional if time remains |

---

## 4. Subsampling and size

- `--stride 5` reduces 66,521 frames to ≈ 13,000 images (≈ 8,000 train / 1,800 val / 2,800 test), enough for YOLO11s fine-tuning and quick to auto-label.
- If auto-labelling or training is too slow: `--stride 10`. If more data is needed: `--stride 3`.
- Balance check after the split: class counts (`drowsy` vs `notdrowsy`), glasses vs no glasses, and per scenario. The script prints this table.

---

## 5. Handling glasses and labels

- **Glasses:** the template placed glasses out of scope, but half of the data has them. Measure MediaPipe's face/eye success rate separately on `glasses` and `noglasses` frames.
  - ≥ ~80 % success → keep both and report results per condition.
  - Much lower → rerun the split with `--noglasses-only` and state the limitation.
- **Cue labels (step 2, next script):** box classes are derived from MediaPipe geometry plus the folder label:
  - `yawn` only if MAR is high **and** the frame is in `drowsy/` (keeps talking and laughing out of the yawn class).
  - `eye_closed` / `eye_open` from EAR relative to a threshold.
  - `head_down` from head pitch.
  - `posture_alert` only if shoulders are visible (undecided).
- Review 300–500 random auto-labelled images **from the val split** manually (Label Studio or CVAT) to measure label quality before training.

---

## 6. Extra data to reduce the generalisation risk

| Extra data | Use |
|---|---|
| Short clips of P and J on the real laptop webcam (eyes closed, yawn, nod, normal, talking), converted to grayscale | **Test only** (never mixed into train/val). Optional fine-tuning set if the live demo underperforms |
| Both with and without glasses if available | Checks robustness |

Own clips are stored locally/on Drive, not committed to git, and not shared (consistent with the "no video stored" privacy decision in the project plan).

---

## 7. Step-by-step procedure

1. **Download** the dataset to Colab local disk (`/content/raw`), not Drive.
2. **Dry run:**
   ```
   python 01_build_split.py --src /content/raw/train_data --dst /content/dataset --dry-run
   ```
   Check the summary table (split × label, glasses, scenario, subject).
3. **Run for real** (remove `--dry-run`). Optional flags: `--stride`, `--test-subject`, `--val-frac`, `--block`, `--gap`, `--noglasses-only`, `--symlink`.
4. **Auto-label** (step 2 script, to be written after the open questions are answered).
5. **Review** 300–500 images and correct.
6. **Convert** to the YOLO layout and write `dataset.yaml`:
   ```
   dataset/
   ├── images/{train,val,test}/
   ├── labels/{train,val,test}/
   ├── manifest.csv
   └── dataset.yaml
   ```
7. **Zip** `dataset/` and store the zip on Drive. At the start of each Colab session copy it to `/content` and unzip.
8. **Freeze** the split: save `manifest.csv` and the command used in the repo (`docs/split_command.txt`) so results are reproducible. **Never re-split after training has started**, or the test set is contaminated.

---

## 8. Rules to protect the evaluation

- The test subject is **never** used for training, validation, threshold tuning, or choosing the confidence level.
- Auto-labelling thresholds are tuned **only** on train/val images.
- Augmentation is applied **after** the split and only to train.
- Rule thresholds (PERCLOS, closure seconds) are chosen on val; the final numbers are reported once on test.
- Report limitations honestly: 4 subjects, one camera setup, grayscale, no night.

---

## 9. Checks before training (checklist)

- [ ] Summary table reviewed and accepted by both teammates
- [ ] No test subject frame appears in train or val (verify with `manifest.csv`)
- [ ] Minimum frame distance between val and train frames of the same video ≥ gap
- [ ] Class balance acceptable (drowsy vs notdrowsy within about 40/60)
- [ ] Glasses decision made and written in the decision log
- [ ] Sample sheet reviewed (shoulders visible? decision on `posture_alert`)
- [ ] Licence of the dataset recorded in `DATA_SOURCES.md`
- [ ] Split frozen and committed (`manifest.csv`, command)
- [ ] Zip uploaded to Drive with a version name, e.g. `dataset_v1.zip`

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Only 3 training subjects → poor generalisation to new faces | High | Own webcam clips for testing/fine-tuning, strong augmentation, honest reporting |
| Test subject unrepresentative (e.g. 005 is easy or hard) | Medium | Report per-scenario results; optionally leave-one-subject-out if time allows |
| Folder labels noisy (per-frame drowsy labels in sleepy videos) | Medium | Review sample, rely on geometry (EAR/MAR/pitch) for cue classes |
| MediaPipe fails on glasses / IR glare | Medium | Measure per condition, filter if poor |
| Re-splitting later contaminates the test set | High | Freeze split, commit manifest |
| Colab disk/time limits | Low | Stride, zip on Drive, copy to local disk |
