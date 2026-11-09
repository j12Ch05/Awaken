# Project Plan Template & Open Questions – YOLO Driver Drowsiness Detection

> Fill this in together with your teammate(s). Answer what you can, mark the rest `TBD`, and revisit at each weekly meeting.
> Status legend: ✅ Decided | 🟡 Discussing | ❌ Open

---

## 1. Project Basics

| Item                                               | Answer             |
| -------------------------------------------------- | ------------------ |
| Project title                                      | Awaken!            |
| Team members & contacts                            | member P, member J |
| Supervisor / stakeholder                           | drivers            |
| Final deadline                                     | 2 months           |
| Weekly meeting day/time                            | NAN                |
| Communication channel (WhatsApp / Slack / Discord) | Whatsapp, Discord  |
| Code hosting (GitHub / GitLab) and branching rules | Github             |

---

## 2. Goals & Scope

| #   | Question                                                                                                               | Answer                                               | Status |
| --- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------ |
| 2.1 | Is this an academic project, graduation project, or product prototype?                                                 | academic project,                                    | ❌     |
| 2.2 | What is the primary goal (best accuracy, real-time demo, comparison study, deployable device)?                         | real-time demo                                       | ❌     |
| 2.3 | Must we also compare with the RF / SVM / Sequential NN / CNN models from the original plan, or is YOLO the only model? | YOLO only                                            | ❌     |
| 2.4 | What is **out of scope** (e.g. phone use detection, seatbelt, distraction)?                                      | distraction, phone use detection, seatbelt           | ❌     |
| 2.5 | Which drowsiness cues must be detected (eye closure, yawning, head nodding, all)?                                      | eye closure, yawning, head nodding, shoulder posture | ❌     |
| 2.6 | Is detecting *distraction* (looking away) also expected?                                                             | no                                                   | ❌     |

---

## 3. Success Metrics

| Metric                                  | Target | Agreed? |
| --------------------------------------- | ------ | ------- |
| Mean Average Precision @0.5 (detection) | sure   |         |
| Recall on drowsy classes                | ?      |         |
| Max false alarms per hour               | 40     |         |
| Minimum FPS                             | 30     |         |
| Max latency per frame (ms)              | 10     |         |
| Max model size (MB)                     | 3000   |         |

Questions:

- 3.1 Which matters more: catching every drowsy event (high recall) or avoiding false alarms (high precision)? catch every drowsy event (preferably with small false alarms)
- 3.2 Who defines the acceptance criteria – us or the supervisor? us

---

## 4. YOLO Approach & Technical Decisions

| #   | Question                                                                                                                     | Answer                                                                   | Status |
| --- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------ |
| 4.1 | Which YOLO version (v5 / v8 / v11 / other)?                                                                                  | v11                                                                      | ❌     |
| 4.2 | Which size (n / s / m)?                                                                                                      | s                                                                        | ❌     |
| 4.3 | Design option: A) state-detection YOLO, B) YOLO + CNN classifier, C) YOLO + landmarks hybrid, D) whole-frame classification? | seems C is the best for this project                                     | ❌     |
| 4.4 | Final class list (e.g. face, eye_open, eye_closed, yawn, mouth_closed, head_down)?                                           | face, eye_open, eye_closed, yawn, mouth_closed, head_down, posture_alert | ❌     |
| 4.5 | Is the Ultralytics AGPL-3.0 licence acceptable for our project/institution?                                                  | yes we're open source                                                    | ❌     |
| 4.6 | Do we still use MediaPipe/dlib for EAR/MAR/head pose, or rely only on YOLO?                                                  | use MediaPipe/dlib for EAR/MAR/head pose                                 | ❌     |
| 4.7 | Frameworks and versions (Python, PyTorch, OpenCV)?                                                                           | Python 3.14.7, PyTorch, OpenCV                                           | ❌     |
| 4.8 | Is drowsiness decided by rules (thresholds) or by a second ML model?                                                         | thresholds                                                               | ❌     |

---

## 5. Data

| #   | Question                                                                                | Answer                                                                                                                                                                                                                                                                                                                                                                                | Status |
| --- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 5.1 | Which datasets will we use (NTHU-DDD, UTA-RLDD, YawDD, DROZY, Roboflow/Kaggle, custom)? | [huggingface.co/datasets/chbh7051/driver-drowsiness-dataset/tree/main/data](https://huggingface.co/datasets/chbh7051/driver-drowsiness-dataset/tree/main/data)                                                                                                                                                                                                                         | ❌     |
| 5.2 | Have licences/access requests been checked and submitted? Who does it?                  | yes                                                                                                                                                                                                                                                                                                                                                                                   | ❌     |
| 5.3 | Do we record our own data? If yes: how many volunteers, what camera, consent forms?     | no we got a dataset of gray scale images                                                                                                                                                                                                                                                                                                                                              | ❌     |
| 5.4 | Who annotates, with which tool (CVAT, Label Studio, Roboflow)?                          |                                                                                                                                                                                                                                                                                                                                                                                       | ❌     |
| 5.5 | Annotation rules: what counts as "eye closed" and as "yawn" (vs. talking)?              | yawn= Maximal downward extension of the jaw accompanied by a distinct widening of the mouth cavity, often showing the tongue or throat.talking= Lip movement is primary horizontal/mild vertical, teeth remain relatively close, and the jaw does not lock at maximum extension.  **Laughing:** Mouth corners pull backward and upward rather than strictly stretching downward | ❌     |
| 5.6 | Target dataset size (images per class)?                                                 | its 3.1 GBs in total                                                                                                                                                                                                                                                                                                                                                                  | ❌     |
| 5.7 | Split ratio and rule (by subject)?                                                      | 70-15-15                                                                                                                                                                                                                                                                                                                                                                              | ❌     |
| 5.8 | Where is data stored (Drive, local, cloud) and how is it versioned?                     | drive                                                                                                                                                                                                                                                                                                                                                                                 | ❌     |
| 5.9 | How do we handle night / infrared images and glasses?                                   | out of scope                                                                                                                                                                                                                                                                                                                                                                          | ❌     |

---

## 6. Hardware & Environment

| #   | Question                                                                                      | Answer                    | Status |
| --- | --------------------------------------------------------------------------------------------- | ------------------------- | ------ |
| 6.1 | What hardware will training run on (GPU model, Colab, Kaggle, university cluster)?            | Colab                     | ❌     |
| 6.2 | What is the **deployment** target (laptop webcam, Raspberry Pi, Jetson, Android phone)? | laptop webcam             | ❌     |
| 6.3 | Which camera (resolution, FPS, IR support) and where is it mounted?                           | integrated camera         | ❌     |
| 6.4 | Is internet available during the demo? (affects cloud use)                                    | unknown, assumed no       | ❌     |
| 6.5 | Budget for hardware (camera, board, speaker)?                                                 | what we have so no budget | ❌     |

---

## 7. Real-Time System & Alarm

| #   | Question                                                              | Answer                                                                                                   | Status |
| --- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------ |
| 7.1 | What alarm type (beep, voice message, vibration, LED, all)?           | beep                                                                                                     | ❌     |
| 7.2 | How many alert levels, and what triggers each?                        | 2 levels, one for when its detected drowsiness, another for when its asleep                              | ❌     |
| 7.3 | Thresholds to start with (eyes closed seconds, PERCLOS %, yawns/min)? | make some value suggestions                                                                              | ❌     |
| 7.4 | Should the system log events? What data may be stored (privacy)?      | system logs when an alert level is triggered and stores the detected values that surpassed the thershold | ❌     |
| 7.5 | Is a UI/dashboard required, or only a console/overlay demo?           | a ui overlay on the camera feed                                                                          | ❌     |
| 7.6 | What should happen if no face is detected for N seconds?              | keeps trying to detect if a face is there                                                                | ❌     |

---

## 8. Testing, Ethics & Safety

| #   | Question                                                                          | Answer                                                     | Status |
| --- | --------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------ |
| 8.1 | How do we test drowsiness safely (simulation, never while driving)?               | recreationally as in sit infront of the camera and test it | ❌     |
| 8.2 | Do we need ethics approval or participant consent forms?                          | no                                                         | ❌     |
| 8.3 | How do we ensure fairness across skin tones, ages, genders, glasses, headscarves? | yes                                                        | ❌     |
| 8.4 | Will video be stored? If so, how is it protected/deleted?                         | no its not stored just parsed live                         | ❌     |

---

## 9. Roles & Responsibilities (RACI-style)

| Task area                                  | Owner | Backup | Notes |
| ------------------------------------------ | ----- | ------ | ----- |
| Literature review                          | P     | J      |       |
| Dataset collection & licences              | J     | P      |       |
| Annotation & QA                            | P     | J      |       |
| YOLO training & tuning                     | J     | P      |       |
| Landmark / feature module (EAR, MAR, pose) | P     | J      |       |
| Temporal / decision logic                  | J     | P      |       |
| Real-time app & alarm                      | P     | J      |       |
| Evaluation & plots                         | P     | J      |       |
| Report & slides                            | J     | P      |       |
| Repo / environment / CI                    | J     | P      |       |

---

## 10. Timeline Check

| Phase                     | Planned weeks | Owner | Start | End | Status |
| ------------------------- | ------------- | ----- | ----- | --- | ------ |
| 1 Scoping & literature    | 1             | P     |       |     |        |
| 2 Dataset & annotation    | 1–2          | J     |       |     |        |
| 3 Baseline & features     | 2–3          | P     |       |     |        |
| 4 YOLO training           | 4–7          | J     |       |     |        |
| 5 Temporal decision layer | 6–8          | J     |       |     |        |
| 6 Evaluation              | 8–9          | P     |       |     |        |
| 7 Real-time integration   | 9–11         | J     |       |     |        |
| 8 Testing & delivery      | 11–12        | P     |       |     |        |

Questions:

- 10.1 Is 12 weeks realistic with our availability (hours/week per person)? no we have about 2 months
- 10.2 Are there exams, holidays, or deadlines that block certain weeks? in 4 weeks exams
- 10.3 Which milestone dates are fixed by the supervisor? there isnt any

---

## 11. Risks (add your own)

| Risk                      | Likelihood (L/M/H) | Impact (L/M/H) | Mitigation                                          |
| ------------------------- | ------------------ | -------------- | --------------------------------------------------- |
| Annotation takes too long | M                  | H              | not sure                                            |
| Not enough GPU time       | L                  | H              | hopefully not too probamatic as we use google colab |
| Dataset access denied     | L                  | L              | impossibble as we downloaded the dataset            |
| Real-time FPS too low     | M                  | M              | optimise somehow                                    |
| Too many false alarms     | H                  | H              | tune model                                          |
|                           |                    |                |                                                     |
