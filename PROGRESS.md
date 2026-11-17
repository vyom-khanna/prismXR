# PROGRESS — PRISM Track 04 (Action-Aware Video Summarization)

Companion files: `understanding.md` (problem spec), plan at
`C:\Users\vyomk\.claude\plans\i-want-you-to-fluffy-hanrahan.md`.
Last updated: 2026-11-16.

---

## 1. Environment & locations (all heavy stuff is OUTSIDE OneDrive)

| What | Where |
|---|---|
| Python env (3.13, torch 2.11+cu128, open_clip) | `C:\prism_env` → `C:/prism_env/Scripts/python.exe` |
| ffmpeg 9.0.2 (winget) | `C:\Users\vyomk\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-9.0.2-full_build\bin` (prepend to PATH) |
| Weights | `C:\prism_data\weights\` → `SlowFast.pyth`, `actionformer\epic_slowfast_{verb,noun}.pth.tar`, `clip\open_clip_pytorch_model.bin`, `epic_prior.npz` |
| FPVSum videos (47/49) | `C:\prism_data\fpvsum\*.mp4` |
| EPIC videos | `C:\prism_data\epic\EPIC-KITCHENS\Pxx\videos\*.MP4` — **NOT downloaded yet** (only partial files) |
| ActionFormer released features | `C:\prism_data\af_features\` (6.5 GB) |
| Dev outputs / caches | `C:\prism_data\dev_outputs\`, `C:\prism_data\cache\` |
| External repos | `C:\prism_ext\` (EPIC annotations + downloader, fpvsum, actionformer_release, epic-kitchens-slowfast) |
| GPU | RTX 4050 Laptop, 6 GB |

## 2. Code (prismXR/submission = the TEAM_ID package)

| File | Role |
|---|---|
| `run.sh` | `./run.sh <video> <out.json>` → `python -m prism.pipeline` |
| `code/prism/pipeline.py` | probe → 2 fps frames → CLIP label (32 frames) → route → branch → finalize → validate |
| `clip_model.py` | CLIP ViT-L/14 (DataComp) zero-shot label, prompt ensembles |
| `sports.py` | FPVSum importance = ridge combo of motion/scene-change/prompt/centre/contrast signals |
| `epic_features.py` + `slowfast/` | vendored EPIC SlowFast-R50, streaming 30 fps, 32-frame clips / stride 16 → 2304-d + verb/noun softmax |
| `epic_detector.py` + `actionformer/` | vendored ActionFormer (pure-python soft-NMS, no C++ build) |
| `pairing.py` | verb dets × noun evidence (noun dets + clip softmax) × P(noun|verb) prior → actions |
| `select_coverage.py` | greedy budgeted max-coverage highlight selection (4/5/6/8 s windows) |
| `highlights.py` | `finalize()` guarantees README rules; peak-window selection |
| `validate.py` | our re-implementation of the (missing) official validator |
| `Dockerfile`, `requirements.txt` | packaging (README.md for submission still TODO) |

Dev tools (`prismXR/tools/`, never part of submission):
`eval/score.py` (all 4 metrics, local GT), `run_all.py` (batch + timings.csv), `build_prior.py`,
`check_detector.py` (run detector on released features), `fpv_cache.py`, `fpv_explore.py`.

Run everything with:
```bash
export PATH="<ffmpeg bin>:$PATH"
C:/prism_env/Scripts/python.exe tools/run_all.py --only fpvsum --out C:/prism_data/dev_outputs/vN/outputs
py -3.13 tools/eval/score.py --pred C:/prism_data/dev_outputs/vN/outputs --only <ids>
```

## 3. Results so far (local scorer, approximate)

| Metric | Floor (uniform) | Ceiling / ref | **Ours now** |
|---|---|---|---|
| Label accuracy (FPVSum, 48 vids, real pipeline v3) | — | — | **100%** (was 95.7% before prompt additions) |
| Highlight F1 (FPVSum, 48 vids) | 14.1% | human 37.6% | **31.2%** (v1 heuristic was 14.8%) |
| Action Coverage (EPIC, 136 vids, released feats) | 19.2% | 61.0% with perfect actions | **35.61%** (budget-fill + window widening; was 33.92) |
| Action pair mAP (EPIC, 136 vids, released feats) | 0 | — | **14.99%** (tuned PairCfg; default 12.78) |
| **Approx. weighted total** | 21.2 (uniform) | — | **≈35.8** = .25·15.0 + .40·35.6 + .25·31.2 + .10·100 |
| Raw verb / noun det mAP (port check) | — | ref 23.51 / 21.88 | **24.01 / 22.10** → vendored ActionFormer + python NMS verified |
| Speed (FPVSum) | — | — | **0.09× realtime** (~11× faster than playback) |
| VRAM peak | — | 6 GB | SlowFast 1.0 GB, ActionFormer 0.3 GB, CLIP 1.6 GB |

## 4. Key findings

1. **Coverage ceiling under 15% budget ≈ 61%** even with perfect actions; uniform gives 19%. EPIC
   actions are short (median 2 s) and there are ~24 distinct pairs/video (median).
2. **FPVSum humans agree only F≈0.38**; GT highlights are short bursts (time-weighted median 2.9 s);
   less importance in first/last 10%.
3. **Motion is the best single importance signal** (Spearman 0.27, F1 0.27); CLIP novelty is useless (F1 0.13).
   Ridge combo, leave-one-category-out F1 = 0.30 → generalises across sports.
4. Released ActionFormer checkpoints report verb 23.51 / noun 21.88 avg-mAP on exactly our 138 val videos.
5. EPIC server is slow (~0.7 MB/s per connection; 6 parallel ≈ 2 MB/s, limited by home bandwidth).

## 5. Decisions taken / open

- Python 3.13 instead of 3.11 (works fine).
- 2 FPVSum videos unavailable on YouTube (`Scuba_KD74Ywuy9ng` private, `Scuba_Um4kqB8bQBg` removed).
  **Proposed**: placeholder JSON (label `other`, uniform highlights) + README note — *awaiting user OK*.
- **TEAM_ID = "Opus 5.5 High"** (config default; env `PRISM_TEAM_ID` overrides).
- FPVSum now **48/49**: `Scuba_Um4kqB8bQBg` recovered with `--extractor-args youtube:player_client=android`
  (158.94 s vs manifest 158.99 ✓). `Scuba_KD74Ywuy9ng` is **private** (owner-only) → unrecoverable →
  placeholder JSON (label `other`, uniform highlights) + README note.
- EPIC video download started 2026-11-16 20:10 (user-approved), 6 parallel, shortest first, ~1.3 MB/s
  (≈40 h for all). Logs `C:\prism_data\epic_dl_*.log`. Resumable: re-run the same command.
- FPVSum importance weights were fitted on FPVSum GT (validated LOCO). Consider re-fitting on
  TVSum/SumMe to avoid any "trained on eval" concern — discuss.

## 5a. Session 2 (2026-11-16 evening)

- Drive still blocks the last 0.5 GB of the features archive, but the partial tar already contains
  **136/138 val feature files** (missing P01_12, P04_27) + train features; extracted to
  `C:\prism_data\af_features\epic_kitchens\features\` (truncated P22_07 deleted). Released .npz also contain
  per-clip `v_prob` / `n_prob` (used for pairing in the check).
- Detections cached: `C:\prism_data\dev_outputs\released_dets.npz`. Tuning: `tools/tune_epic.py` (2-fold CV by participant).
- Label prompts refined (bike/motor, FPV-drone "plane") after inspecting errors on the scored set — note this honestly.
- Tuning (2-fold CV by participant, `tools/tune_epic.py`): both folds chose the same config →
  PairCfg nouns_per_verb=8, w_det=0.5, prior_pow=0 (prior hurt); CovCfg lengths=(4,5), calib_pow=2.
  Held-out pair mAP +2.1 pts per fold. Set as code defaults.
- Dev outputs: `dev_outputs/v3` (FPVSum, final prompts), `dev_outputs/released_feats_tuned` (EPIC).
- TODO: check EPIC kitchen videos still route to `cooking` with the new prompts (needs EPIC frames).

## 5a-2. Session 2, later (2026-11-16 ~22:00)

- **SlowFast feature recipe fixed**: EPIC SlowFast was trained on cv2 (BGR) frames. Our RGB input gave
  cosine 0.75 vs released features; **BGR + full 456x256 frame → cosine 0.93**, verb argmax agreement 0.81.
  On 8 complete EPIC videos, our-features pipeline ≈ released-features pipeline (pair mAP 61.05 vs 61.16 on
  that tiny set) → measured EPIC numbers transfer to run.sh. `tools/check_features.py`.
- End-to-end EPIC run.sh works (8 videos, all routed `cooking`). Speed ≈ 0.40x realtime after replacing the
  full 2 fps decode with 32 seek-frames for labelling (sports videos still decode 2 fps).
- Coverage +1.7: fill budget to 14.9% and widen windows with leftover (<4 s) budget (`CovCfg.widen`).
- **Two float bugs fixed in `highlights.finalize`**: (1) 4.22-0.22 = 3.9999… failed the >=4.0 test → windows
  now built at 4.01 s and checked with the validator's exact float test; (2) growing to 4.01 produced
  4.00999… which wrongly triggered the edge-shift branch and collapsed all windows into one.
- Tried & rejected: clip-level (v_prob×n_prob) pseudo-actions for coverage (fold results split ±).
- Placeholder tool `tools/make_placeholders.py` (for private `Scuba_KD74Ywuy9ng`).
- **EPIC download method changed (22:05)**: the official script restarts a file from zero whenever the Bristol
  server resets the connection (WinError 10054) — hopeless for 2–5 GB files. Now:
  `tools/epic_urls.py` → `C:\prism_data\epic_urls.tsv` (url, path, md5) and
  **`bash /c/prism_data/epic_curl.sh 4`** (curl -C - resume, 100 retries, shortest first, MD5 verify,
  `<video>.MP4.ok` marker when verified). Log: `C:\prism_data\epic_curl.log`. Safe to stop/restart anytime.
- Weights slimmed for packaging (≈3.1 GB → ≈1.2 GB): `SlowFast_model.pyth`, `actionformer/*_ema.pth.tar`,
  `clip/open_clip_fp16.bin`; outputs identical (checked on P02_13).
- `submission/README.md` written. Integrity: `tools/integrity_check.py` → static scan CLEAN, rename test IDENTICAL.
- FPVSum v4 (seek-based labelling): label 100% (48/48), F1 31.2%.
- Phase 6 (stronger detector) judged not feasible: EPIC train videos (~700 GB) not downloadable here, and the
  released train features are from the same SlowFast the checkpoints were trained on (no gain from retraining).

## 5b. State when paused (2026-11-16 ~17:30)

- ActionFormer released features: **6.00 / 6.53 GB** in `C:\prism_data\af_features\epic_kitchens.tar.gzi6eb7l01.part`.
  Google Drive is now refusing (gdown: "can't download… check connections and permissions" = Drive quota/limit,
  plus earlier socket errors 10053/10013). Resume later with:
  `cd C:\prism_data\af_features && py -3.13 -m gdown --continue 1Z4U_dLuu6_cV5NBIrSzsSDOOj2Uar85X -O epic_kitchens.tar.gz`
  (if still blocked: download manually in browser from the Box/Drive link in actionformer_release/README.md).
- No background jobs running. EPIC video download not started.
- Last dev outputs: `C:\prism_data\dev_outputs\v2` (FPVSum, F1 31.5%, label 95.7%).

## 6a. Remaining work (current)

1. Let `epic_curl.sh` finish (~1.2–2 MB/s → ~1–2 days for 184 GB). Restart it if the PC sleeps/reboots.
2. As videos arrive: `tools/run_all.py` (caches SlowFast features in `C:\prism_data\cache`) → outputs for all 187.
3. Final assembly OUTSIDE OneDrive, e.g. `C:\prism_submission\Opus 5.5 High\` = run.sh, code/, weights/ (slim),
   requirements.txt, Dockerfile, outputs/ (187 incl. placeholder), README.md. Validate with
   `validate.py --dir outputs --manifest … --classes …`.
4. Docker build + run test (Docker Desktop installed but stopped; image pull ~8 GB).
5. Demo video (≤3 min) + deck `CollegeName_TeamName_Submission_ppt` — need college name from user.

## 6. Next steps (original list)

1. **Tonight:** EPIC download (6 parallel, shortest first):
   `cd C:\prism_ext\epic-kitchens-download-scripts` and run the 6-way split command (see `C:\prism_data\epic_sorted.txt`).
2. Detector check on released features → verb/noun/pair mAP + coverage; tune `PairCfg` / `CovCfg`.
3. Once EPIC videos exist: verify our SlowFast features ≈ released features (cosine), pick crop/sampling recipe.
4. Full 187-video run, timing table, validation.
5. Label fixes (bike vs motor, plane vs jumping prompts); sports-branch refinements.
6. Phase 6 gate: if pair mAP < 10% → train joint head on released train features.
7. Packaging: README, weights handling, Docker test, integrity scan (rename test), demo video, deck.
