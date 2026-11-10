# Dataset Splitting – Questions for Teammate

> Fill this in together with your teammate. Answer what you can, write `TBD` for the rest.
> `Default` = what `splitting_dataset_plan.md` currently assumes. Status: ✅ Decided | 🟡 Discussing | ❌ Open

---

## A. Test set and split

| # | Question | Default | Answer | Status |
|---|---|---|---|---|
| S1 | Which subject is the **held-out test subject**? 005 (≈ 20 % of data), 001 (≈ 29 %), 002 (≈ 23 %) or 006 (≈ 4 %)? | 005 |default | ❌ |
| S2 | Do you accept that a 70/15/15 split by subject is impossible with 4 subjects and that the real split will be about 55–60 / 10–15 / 20–25? | Yes, documented deviation |default | ❌ |
| S3 | Should val come from **blocks of consecutive frames** of the training subjects (default), or do you prefer another val subject (e.g. 006)? | Blocks, 15 % of blocks per video |default  | ❌ |
| S4 | Block length (frames) and gap at block edges? | 150 frames, gap 15 |default | ❌ |
| S5 | Is leave-one-subject-out cross-validation (4 training runs) desired if time allows, or is a single split enough? | Single split |yes leave-one-subject-out cross-validation is desired | ❌ |
| S6 | Who runs the split and who verifies it (checks the summary table and leakage)? | J runs, P verifies | default  | ❌ |

## B. Size and sampling

| # | Question | Default | Answer | Status |
|---|---|---|---|---|
| S7 | Frame stride (keep every N-th frame)? 3 → ≈ 22k images, 5 → ≈ 13k, 10 → ≈ 6.6k | 5 | default  | ❌ |
| S8 | Do we balance classes (drowsy vs notdrowsy) or keep the natural ≈ 54 / 46 ratio? | Keep natural ratio | default  | ❌ |
| S9 | Do we cap images per video so no single video dominates? | No cap | default  | ❌ |
| S10 | How much time can we spend auto-labelling on Colab (CPU/GPU hours) before it blocks training? | ≤ 1–2 h |  <= 2h | ❌ |

## C. Glasses and scope

| # | Question | Default | Answer | Status |
|---|---|---|---|---|
| S11 | Keep glasses frames (≈ 50 % of the data) or use `--noglasses-only`? | Keep, decide after measuring MediaPipe success per condition | default  | ❌ |
| S12 | What success rate on glasses frames is the minimum to keep them? | ≥ 80 % | default  | ❌ |
| S13 | If glasses are kept, do we report results separately for glasses and no glasses? | Yes | default  | ❌ |
| S14 | Does the final demo need to work for people wearing glasses? (Template says glasses out of scope.) | No | no, but could if it happens | ❌ |

## D. Labels and classes (needed for the next script)

| # | Question | Default | Answer | Status |
|---|---|---|---|---|
| S15 | After viewing `sample_sheet.jpg`: are **shoulders visible**? If not, drop `posture_alert`? | Drop if not visible |  default  | ❌ |
| S16 | `eye_closed`: what EAR rule? (e.g. EAR < 0.75 × the person's open-eye EAR, or fixed 0.2) | 0.75 × per-subject median open-eye EAR | default  | ❌ |
| S17 | `yawn`: what MAR threshold and must the frame be in `drowsy/`? | MAR high (calibrate on the data) **and** in `drowsy/` | default  | ❌ |
| S18 | `head_down`: pitch angle below baseline? | ≥ 20° | default  | ❌ |
| S19 | Talking and laughing frames in `notdrowsy`: label the mouth `mouth_closed` or rename the class `mouth_normal`? | Rename to `mouth_normal` if talking is common | default  | ❌ |
| S20 | One box per eye, or one box covering both eyes? | One per eye | default  | ❌ |
| S21 | Which frames do we review manually, how many, and who does it? | 100–300 random val images, P reviews | default  | ❌ |
| S22 | What auto-label accuracy is good enough to start training? (e.g. ≥ 90 % correct on the review sample) | ≥ 90 % |>= 95% | ❌ |

## E. Extra data and generalisation

| # | Question | Default | Answer | Status |
|---|---|---|---|---|
| S23 | Do we record own webcam clips of P and J? How many minutes each, which actions (eyes closed, yawn, nod, normal, talking), glasses/no glasses? | 3–5 min each, test only | default  | ❌ |
| S24 | Are own clips used **only for testing**, or also for fine-tuning if the live demo underperforms? | Test first, fine-tune only if needed | default  | ❌ |
| S25 | Who records and when (before the exams start)? | P and J, this week or next |  default  | ❌ |
| S26 | Consent for volunteers other than us: needed? Only P and J are used if not. | Only P and J | default  | ❌ |

## F. Reproducibility and storage

| # | Question | Default | Answer | Status |
|---|---|---|---|---|
| S27 | Where is the frozen split stored? (`manifest.csv` + command in the GitHub repo, zip on Drive) | Both | default  | ❌ |
| S28 | Dataset zip naming/versioning (`dataset_v1.zip`, `v2` after relabelling) | v1, v2... with CHANGELOG | default  | ❌ |
| S29 | Who may re-run or change the split after training starts? (Suggestion: nobody, changes only through a new version number.) | Nobody | default  | ❌ |
| S30 | Licence of the dataset: has someone read the dataset page and recorded it in `DATA_SOURCES.md`? Does it allow our use and sharing of derived labels? | Not yet verified | regardless of attempt or effort there is nothing on the website that denies use or sharing of derived labels | ❌ |

---

## Parking lot (anything unclear)

1.
2.
3.

## Decision log

| Date | Decision | Decided by | Reason |
|---|---|---|---|
| | | | |
