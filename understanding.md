# Understanding — Samsung PRISM XR Hackathon 2026, Track 04
## Action-Aware Video Summarization (XR Perception Group)

> Context file. Sources: `Student package_track04/` → `TEAM_README.md`, `manifest.csv`,
> `supported_classes.json`, `P02_109.json`, `skiing_03.json`, and 3 WhatsApp screenshots of the
> hackathon slide deck (slides 18–26 of 27). Where the slides and the README disagree, the
> **README is treated as authoritative** for format and scoring (it's the detailed spec).

---

## 1. The problem in one paragraph

Build a system that watches a **first-person (egocentric) video** and works out **what actions the
person is doing**, especially fine-grained **hand–object interactions** (e.g. "open fridge",
"cut onion"), not just how the scene looks. It then **picks the most important, distinctive time
segments** and drops idle, repetitive or irrelevant footage, giving a short **highlight summary
made only from the original footage**. The summary should be about **15% of the video's length**,
and each segment should be **at least 4 s** long. The system writes **one standard JSON file per
video**.

**Why it matters for XR:** first-person frames become thousands of visual tokens. Pruning them to
the important parts makes downstream XR / multimodal models cheaper and faster.

Two goals named on the slides:
- **Action awareness:** understand meaningful activities and hand–object interactions, not just
  visual scene changes.
- **Importance & redundancy:** keep distinctive events; drop idle, repetitive or irrelevant footage.

---

## 2. Input → Output

- **Input:** one first-person `.mp4` (the slides say 2–10 min; the real durations vary a lot, see §4).
- **Entry point:** `./run.sh <video.mp4> <out.json>`, which runs inference on a single video.
- **Output:** `<video_id>.json`. The schema is the same for every dataset. **187 files** in total, one per manifest video.

---

## 3. Output JSON schema (from README; authoritative)

```json
{
  "team_id": "team_alpha",
  "video_id": "P02_109",
  "duration_sec": 512.3,
  "label": { "name": "cooking", "score": 0.96 },
  "actions": [
    { "segment": [12.4, 18.2], "verb": "open", "noun": "fridge", "score": 0.88 }
  ],
  "highlights": [
    { "segment": [11.0, 19.5] },
    { "segment": [45.2, 53.0] }
  ]
}
```

| Field | Type | Rules | Scored on |
|---|---|---|---|
| `team_id` | string | non-empty, **identical in every file** | — |
| `video_id` | string | non-empty, matches filename and `manifest.csv` | — |
| `duration_sec` | number | > 0, **within ±0.5 s of the manifest duration** | budget check |
| `label` | `{name, score}` | score ∈ [0,1]; `"cooking"` for EPIC; one of the 14 FPVSum categories or `"other"` for FPVSum | Label Accuracy 10% |
| `actions` | array (0–2000) | `{segment:[s,e], verb, noun, score}`; `0 ≤ s < e ≤ duration+0.5`; score ∈ [0,1]; may be `[]` for FPVSum | Action mAP 25% |
| `highlights` | array (≥1) | `{segment:[s,e]}`; **chronologically sorted, strictly non-overlapping, at least one segment ≥ 4.0 s**; target total ≤ 15% of duration | Coverage 40% + Highlight F1 25% |

- Timestamps are in seconds (up to 2 decimals).
- Highlight segments **< 4.0 s are ignored when scoring** (the validator only warns about them).
- Going **over the 15% budget** only triggers a validator warning, but Action Coverage gets a
  **soft length penalty**.
- Any **missing or invalid JSON scores 0 on all metrics** for that video.
- The verbs and nouns must come from `supported_classes.json` (see §5).

> ⚠ The slide deck shows an **older, different schema**: `"actions":[{"verb","noun","start","end"}]`,
> `"highlights":[{"start","end","importance"}]`, and `"label":"category"` as a plain string. **Ignore it.**
> Use the README schema with `segment: [s, e]` and `label: {name, score}`, which matches the example files.

---

## 4. Datasets (the manifest: 187 videos)

Raw videos are **not provided**. Teams download them from the official sources.

### A. EPIC-KITCHENS-100, validation split: 138 videos
- Kitchen activity filmed with a head-mounted camera (participants P01–P32). Label is always `cooking`.
- Downloaded with `epic-kitchens-download-scripts`: `python epic_downloader.py --videos --specific-videos "<ids>" --output-path ...`
- About **184 GB** in total; supports resume and MD5 checks. **Start downloading early** (it takes hours).
- From the manifest: durations range from **11.13 s to 1968.6 s (~33 min)**, mean ≈ 344 s, total ≈ **13.2 h**.
  - 46 videos are under 2 min, 17 are over 10 min. So "2–10 min" is only roughly true.
  - **3 videos are so short that 15% of their length is under 4 s** (P03_26 at 11.13 s, P26_30 at 15.28 s,
    P28_23 at 25.16 s). The rules conflict for these: we must output at least one ≥ 4 s segment, which goes
    over the 15% budget. We'll need a policy for this (e.g. a single 4 s segment and accept the small penalty).
- EPIC-100 has public **ground-truth action annotations** (verb, noun, start/end) for the validation split.
  That's useful for development and analysis, but **outputs must come from our model**. Copying GT means disqualification.

### B. FPVSum: 49 videos
- First-person sports videos with **human importance annotations** (frame-level importance scores from multiple annotators).
- Repo: `github.com/azuxmioy/fpvsum`. Download with `yt-dlp` (`youtube-dl` is retired). Only the 49 still-live videos are listed.
- IDs look like `<Category>_<YouTubeID>`, e.g. `Ski_-Z5a4UQmZLs`, `BikePolo_DHbYfmjX3WQ_1`.
- Durations: **37.34 s to 377.07 s**, mean ≈ 175 s, total ≈ **2.4 h**.
- Category counts: Bike 4, BikePolo 4, Horse 4, Jumping 4, Motor 4, Parkour 4, Ski 4, Surfing 4, Boxing 3,
  Longboarding 3, Plane 3, RockClimbing 3, Scuba 3, Skate 2.
- Labels must be **lowercase** class keys: `bike, bikepolo, boxing, horse, jumping, longboarding, motor,
  parkour, plane, rockclimbing, scuba, skate, ski, surfing` (+ `other`).
- ⚠ **Integrity trap:** the category is right there in the video ID / filename. Reading the label from
  the filename counts as "hardcoding" and risks disqualification. The label must come from the **video
  content** (e.g. zero-shot CLIP/VLM classification over frames).

---

## 5. Vocabulary (`supported_classes.json`, version `prism2026-theme01-v1`)

- **16 labels:** the 14 FPVSum categories + `cooking` + `other`.
- **98 verbs:** the EPIC-KITCHENS-100 canonical verb classes (97) + `other`. Examples: take, put, wash, open, close,
  insert, turn-on, cut, turn-off, pour, mix, …
- **301 nouns:** the EPIC-100 canonical noun classes (300) + `other`. Compound nouns use a colon, e.g.
  `board:chopping`, `liquid:washing`, `maker:coffee`, `peeler:potato`.
- For FPVSum actions, the example uses `verb: "other", noun: "other"` (actions are not scored on FPVSum).

---

## 6. Evaluation (100 points)

| Metric | Weight | Dataset | Field | What it measures |
|---|---|---|---|---|
| **Action Coverage** | **40%** | EPIC-100 | `highlights` | Share of **distinct GT (verb, noun) pairs** captured inside the chosen highlights. Soft length penalty if highlights go over 15% |
| **Action mAP** | 25% | EPIC-100 | `actions` | Temporal action detection mAP for verb–noun pairs, averaged over **tIoU 0.1–0.5** |
| **Highlight F1** | 25% | FPVSum | `highlights` | **Frame-level F1** between our highlights and the **top 15% of human consensus importance frames** |
| **Label Accuracy** | 10% | FPVSum | `label.name` | Video-level category classification accuracy |

**Takeaways for strategy:**
1. **EPIC highlights are the most important part (40%).** The goal there is not "pretty" highlights.
   It's to **cover as many distinct verb–noun pairs as possible** inside the 15% budget. Many short
   segments (each ≥ 4 s) spread over different actions beat a few long ones. Repeated actions add
   nothing, so pick for **diversity**.
2. **EPIC-only work = 65%** (Coverage + mAP), so action detection on EPIC-100 drives most of the score.
   A good temporal action detector (verb + noun + boundaries) feeds both metrics.
3. **FPVSum = 35%.** Highlight F1 needs a per-frame **importance** predictor (motion, saliency, events).
   Label accuracy is a 14-way sports classification (zero-shot CLIP/VLM should work well).
4. Spend the **full 15% budget** where possible (F1 rewards matching the top-15% frames, and
   coverage grows with more footage), but don't go over.
5. **Only a hidden subset is scored**, so every video must get a valid, sensible output.
6. mAP rewards many scored candidate proposals (up to 2000 per video), with calibrated scores.

---

## 7. Validation

- `validate.py` (stdlib-only) checks the 6 keys, types, score ranges, segment ordering, sorting,
  non-overlap, the "at least one segment ≥ 4 s" rule, and warns about segments under 4 s or going over the 15% budget.
- Run with `python validate.py --file x.json -v` or `python validate.py --dir outputs/`.
- ⚠ **`validate.py` is NOT in our package** (the `examples/` folder is also missing; the two
  example JSONs are loose in the folder). We need to get it from the organizers or write an
  equivalent checker that follows the rules above.

### Notes on the example files
- `P02_109.json`: `team_id "T07"`, label cooking, 3 actions, 4 highlights totalling 73 s / 512.3 s ≈ **14.2%**.
  `P02_109` is **not in the manifest**; it's just a format sample.
- `skiing_03.json`: label `"skiing"`, which is **not a valid class key** (it should be `ski`). Its
  video_id is also not a real manifest ID. 3 highlights totalling 37 s / 258 s ≈ 14.3%. Treat it as a format illustration only.
- Both examples use ~11–32 s highlight segments that sit around the action times.

---

## 8. Submission package

```
TEAM_ID/
├── run.sh            # ./run.sh <video.mp4> <out.json>  (single-video inference)
├── code/             # full inference source
├── weights/          # model weights / checkpoints
├── requirements.txt  # exact Python deps
├── outputs/          # 187 validated <video_id>.json
└── README.md         # env setup, model architecture, reproduction steps (+ data used)
```

**Integrity rules**
- Organizers re-run `run.sh` on sample videos **in an isolated environment**, so outputs must be reproducible.
- Automated code scans and spot re-runs. **Copied GT, hardcoded video IDs, or non-reproducible
  outputs → immediate disqualification.**
- (Implication: no reading labels from filenames, no lookup tables keyed by video ID, and weights must be shipped or downloadable.)

**Also needed (from slide 24, general submission rules)**
- Working prototype code in a public or shared GitHub repo.
- README with reproducible setup instructions, Docker files and other requirements.
- **Demo video**, max **3 min** (YouTube or Drive link). ⚠ Other slides say "5-min demo video".
  Aim for ≤ 3 min to be safe.
- Presentation (PPT or PDF) named `CollegeName_TeamName_Submission_ppt` (follow the naming scheme). It should cover:
  theme ID, project title, team details; the problem statement in our own words; the solution and an architecture diagram;
  tools and tech stack; innovation highlights, results and limitations.
- One submission per team, sent through the Google Form. **Not following the guidelines means disqualification.**
- Team: up to 4 members from a single college, team name format `CollegeName_TeamName`, one theme.

---

## 9. Timeline (from slides)

| Date (2026) | Event |
|---|---|
| 5 Nov | Launch, registration opens |
| 8 Nov | Registration closes, final-submission link opens |
| **22 Nov, 11:59 PM** | **Final submission deadline** (prototype code, demo video, deck) |
| 29 Nov | Top 12–15 teams announced |
| 9 Dec | Final demo to the Samsung R&D jury |
| 12 Dec | Final results (one slide says 2 Dec for awards) |

⚠ Other dates on the slides don't match: slide 21 says "Submitted Nov 18, 2026" and slide 26 says "demo video by
18th November". **Treat 18 Nov as the safe target and 22 Nov 11:59 PM as the hard deadline.**
Today is **16 Nov 2026**, so there are about **2–6 days** left.

Rewards: certificates, prizes worth ₹1.0 L, a PRISM worklet with a Samsung mentor, and a possible 2-month Samsung R&D internship.
Queries go to the PRISM Samsung email shown on the slides.

---

## 10. Open issues and risks to keep in mind

1. **Time:** about 184 GB of EPIC video plus about 16 h of footage to run inference on, with a deadline in about 2–6 days.
   Compute and download time are the main constraint.
2. `validate.py` is missing → get it, or write our own checker.
3. Schema conflict between the slides and the README → follow the README.
4. Very short EPIC videos (<26.7 s): the ≥ 4 s segment rule conflicts with the 15% budget.
5. Some EPIC videos are long (up to ~33 min), so the pipeline must handle long videos efficiently.
6. FPVSum labels must come from content, not from the filename.
7. Some FPVSum YouTube links may have died since the manifest was made → we still need a valid JSON
   for every manifest ID.
8. Demo-video length (3 vs 5 min) and submission date (18 vs 22 Nov) conflict → choose the safe option.
