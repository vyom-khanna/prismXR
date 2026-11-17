# Samsung PRISM Hackathon 2026
## Action-Aware Video Summarization (Student Package Guide)

Welcome to **Action-Aware Video Summarization**. Your objective is to build an egocentric video perception system that understands fine-grained actions and hand-object interactions in 2–10 minute first-person videos, selects key temporal segments (`<= 15%` of video duration, each `>= 4.0 s`), and outputs a standardized JSON file per video.

---

## 📦 1. Student Package Contents

This starter kit contains everything you need to download the evaluation datasets, format your predictions, and validate your outputs before submission:

```text
student_package/
├── manifest.csv             # Official list of 187 target videos (video_id, duration_sec, dataset)
├── supported_classes.json   # Supported vocabulary (16 labels, 98 verbs, 301 nouns)
├── validate.py              # Self-contained JSON schema, type, and format validator
├── TEAM_README.md           # Complete dataset, format, validation, and scoring guide
└── examples/                # Sample valid output JSON files (1 per dataset)
    ├── P02_109.json         # Sample output format for EPIC-KITCHENS-100
    └── skiing_03.json       # Sample output format for FPVSum
```

| File / Folder | Description |
|---|---|
| `manifest.csv` | **187 rows** (`video_id,duration_sec,dataset`): **138** `epic_kitchens_100` validation videos + **49** `fpvsum` annotated videos. Teams must process **every** video listed in this manifest. |
| `supported_classes.json` | Official class dictionary: **16 video labels** (14 lowercase FPVSum categories: `bike, bikepolo, boxing, horse, jumping, longboarding, motor, parkour, plane, rockclimbing, scuba, skate, ski, surfing` + `cooking` + `other`), **98 verbs**, and **301 nouns** (EPIC class keys + `other`). |
| `validate.py` | Standard-library Python script that verifies whether all required JSON fields are present and checks their data types, value ranges, and highlight segment rules. |
| `examples/` | Contains one reference JSON output for EPIC-KITCHENS-100 (`P02_109.json`) and one for FPVSum (`skiing_03.json`). |

---

## 🌐 2. Dataset Sourcing (Self-Download Instructions)

**Raw video files are NOT distributed by the organizers.** Teams must download the source videos directly from their official public repositories using the video IDs in `manifest.csv`.

### A. EPIC-KITCHENS-100 Validation Split (`N = 138` Videos)
* **Official Dataset Page:** https://epic-kitchens.github.io/2020
* **Official Downloader Repository:** https://github.com/epic-kitchens/epic-kitchens-download-scripts
* **How to Download:**
  1. Clone the official downloader repository:
     ```bash
     git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
     cd epic-kitchens-download-scripts
     ```
  2. Download the 138 manifest video IDs from the official `data.bris` endpoint:
     ```bash
     python epic_downloader.py --videos --specific-videos "<comma_separated_138_ids_from_manifest>" --output-path ../EPIC-KITCHENS
     ```
* **Important Note:** All 138 videos resolve on the official endpoint (total size **~184 GB**; downloads support resume and MD5 verification). **Start downloading early**, as full retrieval takes several hours.

### B. FPVSum Annotated Videos (`M = 49` Videos)
* **Official Repository:** https://github.com/azuxmioy/fpvsum (or https://github.com/v-iashin/FPVSum)
* **How to Download:**
  1. Clone the repository and install `yt-dlp` (substituting the retired `youtube-dl` utility in the crawler):
     ```bash
     pip install yt-dlp
     ```
  2. Download the **49 FPVSum video IDs** listed in `manifest.csv` (each ID uses the `<Category>_<YouTubeID>` format, e.g., `Ski_-Z5a4UQmZLs`).
* **Important Note:** While the original FPVSum dataset references 98 videos, several YouTube links are no longer active. `manifest.csv` lists **only the 49 videos** that are live and have ground-truth annotations.

---

## 📋 3. Output JSON Format & Field Reference

Your pipeline must produce **one JSON file per manifest video** named `<video_id>.json` inside your `outputs/` folder (**187 JSON files total**). Both datasets use the **exact same 6 top-level fields** (refer to `examples/P02_109.json` and `examples/skiing_03.json`).

### JSON Schema Example
```json
{
  "team_id": "team_alpha",
  "video_id": "P02_109",
  "duration_sec": 512.3,
  "label": {
    "name": "cooking",
    "score": 0.96
  },
  "actions": [
    {
      "segment": [12.4, 18.2],
      "verb": "open",
      "noun": "fridge",
      "score": 0.88
    }
  ],
  "highlights": [
    { "segment": [11.0, 19.5] },
    { "segment": [45.2, 53.0] }
  ]
}
```

### Field Reference & Constraints

| Field | Type | Constraints & Format Rules | Scored On |
|---|---|---|---|
| `team_id` | `string` | Non-empty string; must be identical across all submitted files. | — |
| `video_id` | `string` | Non-empty string; must match the filename (`<video_id>.json`) and `manifest.csv`. | — |
| `duration_sec` | `number` | Positive number (`> 0`); must be within `±0.5 s` of the official `manifest.csv` duration. | Budget check |
| `label` | `object` | `{"name": string, "score": number}` with `score` in `[0, 1]`. Use `"cooking"` for EPIC videos, or one of the 14 FPVSum categories (or `"other"`) for FPVSum videos. | **Label Accuracy (10%)** |
| `actions` | `array` | `0–2000` items of `{"segment": [start, end], "verb": string, "noun": string, "score": number}`. `0 <= start < end <= duration_sec + 0.5`; `score` in `[0, 1]`. May be `[]` for FPVSum videos. | **Action mAP (25%)** |
| `highlights` | `array` | At least 1 item of `{"segment": [start, end]}`. Must be chronologically sorted, strictly non-overlapping, with at least one segment `>= 4.0 s`. Target total length `<= 15%` of `duration_sec`. | **Action Coverage (40%) & Highlight F1 (25%)** |

> **Note:** Timestamps should be in seconds (up to 2 decimal places).

---

## ✅ 4. Validating Your Outputs (`validate.py`)

Before submitting, run `validate.py` to verify that all required keys are present and all data types, score bounds, and segment constraints are satisfied.

### Validate a Single File (with warnings)
```bash
python validate.py --file examples/P02_109.json -v
```

### Validate Your Full Output Directory
```bash
python validate.py --dir outputs/
```

* **What `validate.py` checks:**
  * Presence of all 6 required top-level fields (`team_id`, `video_id`, `duration_sec`, `label`, `actions`, `highlights`).
  * Data types (`string`, `number`, `object`, `array`) and finite numeric bounds (`score` in `[0, 1]`, `duration_sec > 0`).
  * Valid segment ordering (`0 <= start < end <= duration_sec + 0.5`), chronological sorting, and zero overlap between highlight segments.
  * Ensures at least one highlight segment is `>= 4.0 s` and warns if any segment is `< 4.0 s` (ignored during scoring) or if total highlight duration exceeds the `15%` budget.
* **Pass / Fail Rule:** Any missing or invalid JSON file scores **0** on all metrics for that video.

---

## 🏆 5. Evaluation Metrics & Final Scoring Formula

Submissions are evaluated across **four objective metrics**:

| Metric | Weight | Dataset & Target | 1-Line Definition |
|---|---|---|---|
| **Action mAP** | **25%** | EPIC-KITCHENS-100 (`actions`) | Evaluates how closely predicted verb-noun actions overlap with ground-truth boundaries across `temporal Intersection-over-Union (tIoU)` thresholds from `0.1 to 0.5`. |
| **Action Coverage** | **40%** | EPIC-KITCHENS-100 (`highlights`) | Measures the share of distinct ground-truth action pairs captured within the selected highlights (with a soft length penalty if highlights `exceed the 15% budget`). |
| **Highlight F1** | **25%** | FPVSum (`highlights`) | Calculates the frame-level F1 score comparing selected highlights directly against the `top 15%` of human-annotated consensus importance frames. |
| **Label Accuracy** | **10%** | FPVSum (`label.name`) | Evaluates the `classification accuracy` of predicting the overall video-level category on FPVSum videos. |

---

## 📤 6. Submission Package & Integrity Policy

### Required Upload Structure
Upload a single archive/folder named after your `TEAM_ID`:

```text
TEAM_ID/
├── run.sh             # Single-video inference entry point: ./run.sh <video.mp4> <out.json>
├── code/              # Complete inference source code
├── weights/           # Model weights / checkpoints
├── requirements.txt   # Exact Python dependencies
├── outputs/           # 187 validated <video_id>.json files (one per manifest video)
└── README.md          # Environment setup, model architecture, and reproduction steps
```

### Integrity & Verification Rules
* **Hidden Scored Subset:** Only a subset of the manifest videos is scored. The subset is hidden and will be revealed together with the results.
* **Originality & Reproducibility:** Outputs must be generated by your submitted system and must be reproducible when organizers re-run `./run.sh <video.mp4> <out.json>` on sample videos in an isolated environment.
* **Zero Tolerance for Hardcoding:** Automated code scans and spot re-runs are conducted prior to final ranking. Copied ground truth, hardcoded video IDs, or non-reproducible outputs lead to immediate disqualification.