# Literature review: Action-Aware Video Summarization (PRISM XR Track 04)

Date: 2026-11-22. Scope: 2014–2026, weighted to 2022–2026. 52 works are in the table.

The review maps each work to our four scored sub-problems and to the v7 system (39.72: coverage 40.95, mAP 22.54, F1 31.64, label 97.96). I read PROGRESS.md (v1–v7, including the rejected lists), submission/README.md, `pipeline.py`, `select_coverage.py`, `sports.py` and `tools/dev_v3.py`.

**How reliable the numbers are.** Numbers marked **[v]** were read from the paper, or a page quoting it, during this session. Numbers marked **[r]** are recalled from memory and were not re-checked today; treat them as approximate. Benchmarks differ in protocol: EPIC detection average mAP over tIoU 0.1–0.5 vs test-server numbers, and TVSum/SumMe F1 with KTS + knapsack vs our frame-level top-15% F1. Absolute numbers therefore do not transfer to our scorer. Only the direction and rough size of an effect transfer.

**Score arithmetic.** These rates set every gain estimate below:
- +1 coverage point = +0.40 weighted
- +1 mAP point = +0.25 weighted
- +1 F1 point = +0.25 weighted
- label accuracy is at its ceiling (the only miss is the private-video placeholder)

Reaching 42 needs +2.3 weighted, which is, for example, +2.5 coverage plus +2 mAP plus +3 F1. The advisor's CV-noise estimate is ±0.3 weighted.

---

## (i) Consolidated table

Outcome key:
- **POS** = POSITIVE-APPLICABLE
- **USED** = ALREADY-USED
- **NEG** = TRIED-NEGATIVE (our PROGRESS result is cited)
- **NF** = NOT-FEASIBLE (the reason is given)

Sub-problem key:
- COV = action coverage
- MAP = pair mAP
- F1 = FPVSum highlight F1
- LBL = label accuracy
- XR = XR efficiency (not scored; useful for the deck)

### A. Egocentric action recognition and temporal action detection (EPIC-KITCHENS-100)

| # | Paper (venue, year, URL) | Maps to | Key idea | Reported results | Outcome for us |
|---|---|---|---|---|---|
| 1 | Damen et al., *Rescaling Egocentric Vision* (EPIC-KITCHENS-100), IJCV 2022, https://arxiv.org/abs/2006.13256 | MAP, COV | Dataset and protocol. Detection = mAP over tIoU 0.1–0.5 for verb, noun and action. Actions are dense and short. | Baseline action detection is low, about 5–6 avg mAP (G-TAD on test) [r] | USED: defines the metric our scorer re-implements |
| 2 | Zhang, Wu, Li, *ActionFormer*, ECCV 2022, https://arxiv.org/abs/2202.07925 | MAP, COV | Single-stage multi-scale local-attention transformer. Per-point classification and boundary regression, Soft-NMS. | EPIC-100 val SlowFast: verb 23.5, noun 21.9 avg mAP [v, our PROGRESS port check 24.0/22.1] | USED: our verb/noun proposal source |
| 3 | Shi et al., *TriDet*, CVPR 2023, https://arxiv.org/abs/2303.07347 | MAP, COV | Trident head with relative boundary distributions; SGP layer. | EPIC verb about 25.4, noun about 23.8 avg mAP [r] | NEG: replaced, unioned or rescored AF; mAP +0.2 at best, coverage −0.1 to −0.8 in every variant (PROGRESS "TriDet … REJECTED") |
| 4 | Tang et al., *TemporalMaxer*, arXiv 2023, https://arxiv.org/abs/2303.09055 | MAP | Replaces attention with max-pooling context; fewer parameters. | EPIC verb 24.5, noun 22.8 [v] | NF: no released EPIC checkpoint trained on our feature recipe. Retraining gives about +1 verb mAP, which TriDet showed does not reach pair mAP or coverage |
| 5 | Yang et al., *DyFADet*, ECCV 2024, https://arxiv.org/abs/2407.03197 | MAP | Dynamic kernels and receptive fields per timestamp (DFA), dynamic head. | EPIC verb avg about 25.0 (quoted in ContextDet's table) [v, secondary] | NF: would need retraining on released train features. AF seed retraining took 21 min/epoch × 21 epochs per model, and the seed ensemble was NEG. A +1.5 verb mAP detector is below what the pipeline converts into pair mAP |
| 6 | *ContextDet: TAD with Adaptive Context Aggregation*, arXiv 2024, https://arxiv.org/abs/2410.15279 | MAP | Pyramid adaptive context aggregation (large-kernel context). | Lists AF 23.5 / TriDet 23.8 / DyFADet 25.0 verb avg on EPIC [v] | NF: same reason as #5 |
| 7 | *Introducing Gating and Context into TAD*, arXiv 2024, https://arxiv.org/abs/2409.04205 | MAP | Gated local context plus boundary context on the TriDet backbone. | EPIC verb mAP@0.1 = 28.7 vs TriDet 27.8 [v] | NF: retraining; small gain |
| 8 | *CausalTAD: Harnessing Temporal Causality for TAD*, arXiv 2024 (EPIC 2024 challenge winner), https://arxiv.org/abs/2407.17792 | MAP | Causal plus anti-causal attention/Mamba blocks, InternVideo2 features. | **31.97 action avg mAP on test**, trained on train+val, no ensemble [v] | NF: needs InternVideo2-1B features (too big for 6 GB, about 1 day to extract) and train+val training, which our rules forbid |
| 9 | *EgoAction: Reliability-Aware Temporal Fusion*, CVPR 2026 EPIC challenge report, https://arxiv.org/abs/2605.24496 | **MAP** | Separate verb and noun detectors. Top-10 × top-10 composition, score = sqrt(P_v·P_n). **Dynamic Weighted Fusion of verb/noun boundaries** (weights = max class confidences). Class-wise Soft-NMS **with score voting** (verb/action vote thr 0.75, noun 0.65). | Val: noun 36.25, verb 32.91, hard joint action 28.86. Test: 25.94 action [v]. No per-step ablation | **POS (shortlist #3)**: boundary voting/fusion applies to our *output* actions only. Composition is already USED (our top-8 nouns, s_v·p_n^0.75) |
| 10 | Wang et al., *A Stronger Baseline for Ego-Centric Action Detection*, EPIC 2021 report, https://arxiv.org/abs/2106.06942 | MAP | Verb/noun detection, then action composition with priors. | 16.10 action on test (single model) [v] | USED: same recipe family as our pairing |
| 11 | *OADT: One-stage Action Detection Transformer*, EPIC 2022 report, https://arxiv.org/abs/2206.10080 | MAP | One-stage transformer with ensembling. | 21.28 action avg mAP test (ensemble) [v] | NEG (in spirit): our ensembling (AF seed, TriDet union) gave ≤ +0.1 mAP and lowered coverage |
| 12 | Liu et al., *AdaTAD: End-to-End TAD with 1B Parameters Across 1000 Frames*, CVPR 2024, https://arxiv.org/abs/2311.17241 | MAP | End-to-end adapter-tuned VideoMAE backbone for TAD. | Large EPIC gains from end-to-end training [r] | NF: needs EPIC train videos (700 GB) and a large GPU |
| 13 | Zhao & Krähenbühl, *LaViLa: Learning Video Representations from LLMs*, CVPR 2023, https://arxiv.org/abs/2212.04501 | MAP, COV | Narrator/rephraser LLM pseudo-captions for Ego4D, dual encoder, fine-tuned EK100 classifier. | TSF-L EK100 action top-1 51.0; verb about 73.0 / noun about 65.4 (quoted elsewhere) [v] | USED: verb/noun heads on the top-300 segments (v3: held-out 117 videos, mAP 18.2 → 21.1) |
| 14 | Zhao & Krähenbühl, *AVION: Training Video Foundation Models with Fractional Resources*, CVPR 2024, https://arxiv.org/abs/2309.16669 | MAP | Efficient decode plus FlashAttention ViT-L pretraining on LaViLa narrations. | EK100 action top-1 **54.5** vs LaViLa 51.0 (README) [v] | NF tonight: a second 30 fps decode plus a ViT-L pass over 138 videos takes about 3–4 h of GPU (our LaViLa pass alone runs at 0.28× real time). The best case, a few noun top-1 points, would give about +0.5 mAP. Its action head has the same train+val index issue as LaViLa |
| 15 | Lin et al., *EgoVLP*, NeurIPS 2022, https://arxiv.org/abs/2206.01670 | MAP, LBL | Egocentric video-language pretraining on EgoClip (Ego4D), EgoNCE loss. | EK100 MIR and zero-shot gains over CLIP-style baselines [r] | NF: weaker than LaViLa for EK100 classification; would need another decode pass |
| 16 | Pramanick et al., *EgoVLPv2*, ICCV 2023, https://arxiv.org/abs/2307.05463 | MAP | Fusion inside the backbone for egocentric VLP. | Improves EgoVLP on EK100 MIR and EgoMQ [r] | NF: same as #15 |
| 17 | Wang et al., *InternVideo2*, ECCV 2024, https://arxiv.org/abs/2403.15377 | MAP | 6B video foundation model; feature source for CausalTAD/EgoVideo. | SOTA on many video benchmarks [r] | NF: VRAM and time |
| 18 | Wang et al., *VideoMAE V2*, CVPR 2023, https://arxiv.org/abs/2303.16727 | MAP | Dual masking, billion-scale video MAE; EPIC-finetuned features drive 2023–2026 TAD entries (EgoAction). | — | NF: no EPIC-finetuned checkpoint trained on train only that fits our time |
| 19 | Kazakos et al., *With a Little Help from my Temporal Context (MTCN)*, BMVC 2021, https://arxiv.org/abs/2111.01024 | **MAP** | Attends to neighbouring actions (window w = 9 actions ≈ 34 s), plus a transformer LM over action sequences fused by beam search. | EK100 val action top-1 41.30 (w = 1) → 45.48 (w = 9). LM alone +0.48 action top-1 (seed-averaged +0.35). **N-gram LM gave no gain** [v] | USED in v7 (noun co-occurrence context, +0.18 mAP). The residual idea (verb context, global video-level presence) is **POS but small (shortlist #4)**. The paper's own evidence says a count-based LM beyond co-occurrence adds little |
| 20 | Pato et al., *Seeing without Looking: Contextual Rescoring of Object Detections for AP Maximization*, CVPR 2020, https://arxiv.org/abs/1912.12290 | MAP | Rescores every detection from the set of other detections, trained to maximise AP; output-only post-processing. | COCO +0.5–1 AP from rescoring alone [r] | USED (v7 rationale). The learned rescorer would need train-split detections from models trained on train (biased), so only fixed-form context is safe |
| 21 | Dave et al., *Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details*, arXiv 2021, https://arxiv.org/abs/2102.01066 | **MAP** | **Under a shared per-image detection cap, AP is not class-independent.** Re-ranking which detections survive the cap (dropping surplus detections of frequent classes, letting rare-class ones in) raises AP, and rare-class AP most. | LVIS v1 val: AP 22.6 → 23.8, APr 12.6 → 15.5 from re-ranking alone. Raising the cap 300 → 10k/image: APr 12.6 → 19.8 [v] | **POS (shortlist #1)**: the PRISM 2000-actions-per-video cap is exactly such a shared cap over ~2k pair classes, and mAP is averaged over val pair classes (most of them rare). We currently cut at the top-2000 by raw score |
| 22 | Bodla et al., *Soft-NMS*, ICCV 2017, https://arxiv.org/abs/1704.04503 | MAP | Decay overlapping scores instead of deleting them. | +1–2 AP on COCO/VOC [r] | USED (inside ActionFormer) |
| 23 | Gidaris & Komodakis, *Multi-region & semantic-segmentation-aware CNN* (box voting), ICCV 2015, https://arxiv.org/abs/1505.01749 | MAP | **Box/score voting**: refine a kept box as the score-weighted average of the boxes overlapping it. | VOC07 mAP +1–2 from iterative localisation plus voting [r] | **POS (shortlist #3)**: our `merge_duplicates` keeps the top instance's boundaries and only noisy-ors the scores |
| 24 | Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017, https://arxiv.org/abs/1706.04599 | MAP, COV | Temperature scaling. | ECE reductions [r] | NEG (in effect): every score recalibration moved the coverage calibration (`calib_pow`). The per-video recalibration / TTA programme was all rejected (38.64–39.66 vs 39.69) |

### B. Video summarization and highlight detection

| # | Paper (venue, year, URL) | Maps to | Key idea | Reported results | Outcome for us |
|---|---|---|---|---|---|
| 25 | Ho, Chiu, Wang, *Summarizing First-Person Videos from Third Persons' Points of Views* (FPVSum), ECCV 2018, https://arxiv.org/abs/1711.08922 | F1, LBL | FPVSum (14 sports categories, frame-level importance). Learns from third-person data via metric learning and domain adaptation with a few labelled first-person videos. | Uses an 80/20 train/test split. The PDF could not be parsed in this session, so its F numbers are not verified | USED as the eval set. The method needs FPVSum training → NF under our rules |
| 26 | Song et al., *TVSum*, CVPR 2015, https://openaccess.thecvf.com/content_cvpr_2015/html/Song_TVSum_Summarizing_Web_2015_CVPR_paper.html | F1 | 50 web videos, 20 raters, shot importance; title-based co-archetypal analysis. | — | POS only as an *allowed* validation/pretraining set (not downloaded) |
| 27 | Gygli et al., *SumMe*, ECCV 2014, https://gyglim.github.io/me/vsum/index.html | F1 | 25 user videos (some egocentric). Interestingness from attention, aesthetics, landmarks and faces. | Human F about 0.31 average [r] | Same as #26 |
| 28 | Otani et al., *Rethinking the Evaluation of Video Summaries*, CVPR 2019, https://arxiv.org/abs/1903.11328 | F1 | Random summaries score near SOTA on F1 with KTS + knapsack; segmentation drives F1; rank correlation is recommended. | Random ≈ SOTA F1 on TVSum/SumMe [r] | USED (insight): our F1 depends heavily on window structure. v6 widening gave +0.76 F1 with no new signal. Treat any new F1 signal with suspicion |
| 29 | Gygli et al., *Video Summarization by Learning Submodular Mixtures of Objectives*, CVPR 2015, https://openaccess.thecvf.com/content_cvpr_2015/html/Gygli_Video_Summarization_by_2015_CVPR_paper.html | COV, F1 | Learned mixture of interestingness, representativeness and uniformity, optimised greedily. | SumMe F about 0.40 [r] | USED (pattern): our coverage selector is a submodular greedy with a learned density term |
| 30 | Zhang et al., *Video Summarization with LSTM* (vsLSTM/dpp-LSTM), ECCV 2016, https://arxiv.org/abs/1605.08110 | F1, COV | BiLSTM importance plus a DPP diversity term. | SumMe about 38.6, TVSum about 54.7 F [r] | NF: supervised on TVSum/SumMe; diversity is irrelevant to FPVSum's peaky GT |
| 31 | Zhou et al., *DR-DSN* (diversity–representativeness RL), AAAI 2018, https://arxiv.org/abs/1801.00054 | F1 | Unsupervised RL reward = diversity + representativeness. | SumMe 41.4, TVSum 57.6 F [r] | NEG (analogue): CLIP novelty/uniqueness was useless on FPVSum (F1 0.13; `local_unique` weight ≈ 0) |
| 32 | Narasimhan et al., *CLIP-It!*, NeurIPS 2021, https://arxiv.org/abs/2107.00650 | F1 | Language-guided multimodal transformer; generic summaries via auto-captions. | TVSum F about 66, SumMe about 52 (supervised) [r] | NF: trained on TVSum/SumMe with a captioner; heavy |
| 33 | He et al., *A2Summ: Align and Attend*, CVPR 2023, https://arxiv.org/abs/2303.07284 | F1 | Multimodal (video + transcript) summarization with dual contrastive losses. | SOTA TVSum/SumMe F1 (about 63/55) [r] | NF: needs transcripts; FPVSum has none |
| 34 | Barbara & Maalouf, *Prompts to Summaries*, arXiv 2025, https://arxiv.org/abs/2506.10807 | F1 | Training-free: scene split → VLM captions → LLM importance scoring. | SumMe 56.73, TVSum 62.21 F1 (zero-shot) [v, quoted by #35] | NF tonight: VLM captioning of every scene plus an LLM on 6 GB, prompt-sensitive, about 1 h of GPU for 48 videos plus integration. GT for sports is motion bursts, which captions describe poorly |
| 35 | Wu et al., *Context-Aware Pseudo-Label Scoring for Zero-Shot Video Summarization*, arXiv 2025, https://arxiv.org/abs/2510.17501 | F1 | Rubric-guided pseudo-labels plus contextual LLM prompting. | SumMe 57.58, TVSum 63.05 [v] | NF: same as #34. The authors note sensitivity to prompts and score normalisation |
| 36 | Wen, *Comparing Learning Paradigms for Egocentric Video Summarization*, arXiv 2025, https://arxiv.org/abs/2506.21785 | F1 | Shotluck Holmes / TAC-SUM / GPT-4o on Ego-Exo4D: specialised summarizers degrade on egocentric video, and a prompt-tuned GPT-4o does best. | Qualitative only [v] | NF: cloud LLM; no numbers |
| 37 | Byra et al., *Zero-shot highlight detection from text descriptions and synthetic images*, arXiv 2026, https://arxiv.org/abs/2609.14790 | F1 | LLM writes highlight descriptors from the category, a diffusion model makes prototypes, CLIP matches them to frames. | Claims strong zero-shot TVSum [v, no numbers seen] | Weak POS: equals a richer version of our CLIP "exciting" prompt signal, whose ridge weight is already small (0.069). Low expected gain |
| 38 | Sun, Farhadi, Seitz, *Ranking Domain-specific Highlights by Analyzing Edited Videos* (YouTube Highlights), ECCV 2014, https://www.semanticscholar.org/search?q=Ranking%20Domain-specific%20Highlights%20by%20Analyzing%20Edited%20Videos | F1 | Dataset of 6 domains (dog, gymnastics, **parkour, skating, skiing, surfing**), largely GoPro-style first-person footage; pairwise ranking from raw vs edited video. | — | POS as an *allowed* source: 4 of its 6 domains are FPVSum categories (14 of our 48 videos) |
| 39 | Yao, Mei, Rui, *Highlight Detection with Pairwise Deep Ranking for First-Person Video Summarization*, CVPR 2016, https://www.microsoft.com/en-us/research/?p=238200 | F1 | Two-stream pairwise ranking on 100 h of first-person sports video. | +10.5% accuracy over RankSVM [v] | NF: dataset and model not public |
| 40 | Xiong et al., *Less is More: Highlight Detection from Video Duration* (LIM-S), CVPR 2019, https://arxiv.org/abs/1903.00859 | F1 | Weak supervision: short user videos are highlight-dense; trained on 10M Instagram videos. | YouTube HL avg 56.4 mAP, TVSum 56.3 [v] | NF: weights not released, as far as I know |
| 41 | Lei et al., *QVHighlights / Moment-DETR*, NeurIPS 2021, https://arxiv.org/abs/2107.09609 | F1 | DETR for moments plus per-clip saliency given a text query. | QVHighlights HIT@1 about 56 [r] | NF on its own: query-conditioned vlog/news domain. Its successors (#42–45) are the usable checkpoints |
| 42 | Liu et al., *UMT*, CVPR 2022, https://arxiv.org/abs/2203.12745 | F1 | Audio + visual (+ optional text) transformer; YouTube HL and TVSum checkpoints per domain. | YouTube HL avg 74.9 mAP, TVSum 83.1 top-5 mAP [v] | Possible POS, high risk: needs the authors' exact I3D (+ PANN audio) features for FPVSum frames; we have no I3D pipeline. Per-domain checkpoints exist only for 4 of 14 categories. Not tonight |
| 43 | Moon et al., *QD-DETR*, CVPR 2023, https://arxiv.org/abs/2303.13874 | F1 | Query-dependent DETR with negative-pair saliency learning. | QVHighlights SOTA at the time; TVSum/YouTube HL results [r] | Same risk as #42 (SlowFast+CLIP features of the QVHL recipe) |
| 44 | Lin et al., **UniVTG**, ICCV 2023, https://arxiv.org/abs/2307.16715 ; code (MIT) https://github.com/showlab/UniVTG | **F1** | Unified grounding (point, interval, curve labels). Pretrained on **4.2M labels from Ego4D narrations + VideoCC + CLIP-teacher curves (no FPVSum)**. Raw-video demo (`run_on_video`, Gradio) with a CLIP-B/16 checkpoint; < 4 GB GPU; < 1 s per 10-min video. | **Zero-shot** YouTube HL: parkour 65.9, skiing 64.5, surfing 54.0, skating 39.2 (avg 53.9) vs fine-tuned 75.2 / w-PT 78.6, and LIM-S 56.4. Zero-shot TVSum 67.2 vs LIM-S 56.3 [v] | **POS (shortlist #5)**: the only checkpoint that is zero-shot, egocentric-pretrained, raw-video ready and small enough. Evidence of zero-shot transfer is mixed per domain |
| 45 | Moon et al., *CG-DETR*, arXiv 2023, https://arxiv.org/abs/2311.08835 ; Sun et al., *TR-DETR*, AAAI 2024, https://arxiv.org/abs/2401.02309 | F1 | Correlation-guided / task-reciprocal DETRs for moment retrieval + highlight detection. | Incremental QVHL/TVSum gains [r] | NF: same feature-recipe risk as #42; no zero-shot story |
| 46 | Han et al., *HL-CLIP: Unleash the Potential of CLIP for Video Highlight Detection*, CVPRW 2024, https://arxiv.org/abs/2404.01745 | F1 | Fine-tune the last CLIP layers for query saliency. **Saliency pooling at inference** (average neighbouring frames' saliency weighted by their similarity). | QVHighlights SOTA (fine-tuned) [v] | Saliency pooling = temporal smoothing, which we already have (motion_s5/s11/s21) → USED in spirit. The fine-tuned model is QVHL-domain → weak |
| 47 | Islam et al., *Unsupervised Video Highlight Detection by Learning from Audio and Visual Recurrence*, NeurIPS 2024, https://arxiv.org/abs/2407.13933 | F1 | Pseudo-highlights from audio-visual recurrence across pseudo-categories; beats prior unsupervised methods. | YouTube HL best unsupervised; TVSum about +8 pts over prior unsupervised [v, no exact numbers] | NF: the training stage would have to run on FPVSum itself (transductive), which is forbidden. **Audio** as a cue: see #48 |
| 48 | *RASL: Unsupervised Modality-Transferable Highlight Detection*, arXiv 2024, https://arxiv.org/abs/2403.09401 ; Badamdorj et al., *Joint Visual and Audio Learning for Video Highlight Detection*, ICCV 2021 (no URL checked) | F1 | Audio-visual unsupervised highlight learning; audio is a strong cue on YouTube HL. | UMT audio variant is among the best on YouTube HL (#42) [v] | Weak POS: we use no audio. But FPVSum YouTube uploads often carry *added music*, so audio energy may track the edit rather than the action, and its weight would have to be guessed (it can't be fitted on FPVSum). Low priority |
| 49 | Lee & Grauman, *Predicting Important Objects for Egocentric Video Summarization*, IJCV 2015, https://arxiv.org/abs/1505.04803 ; Xu et al., *Gaze-Enabled Egocentric Video Summarization via Constrained Submodular Maximization*, CVPR 2015, https://openaccess.thecvf.com/content_cvpr_2015/html/Xu_Gaze-Enabled_Egocentric_Video_2015_CVPR_paper.html | COV, F1 | Hand proximity, object and gaze cues for egocentric importance; gaze fixations improve summaries; constrained submodular selection. | Gaze significantly helps [v] | NF: no gaze in our input. HOI detectors cost a frame-level GPU pass (see #57) |

### C. Coverage, set cover under uncertainty, counting

| # | Paper | Maps to | Key idea | Results | Outcome |
|---|---|---|---|---|---|
| 50 | Khuller, Moss, Naor, *The Budgeted Maximum Coverage Problem*, IPL 1999, https://doi.org/10.1016/S0020-0190(99)00031-9 ; Lin & Bilmes, *Multi-document Summarization via Budgeted Maximization of Submodular Functions*, NAACL 2010, https://aclanthology.org/N10-1134/ | COV | Ratio greedy alone is not guaranteed. Greedy plus best single set, or partial enumeration, gives (1−1/e). The scaled ratio gain/cost^r and **post-greedy local improvement** are standard. | Theory; on DUC, modified greedy beats plain greedy by about 1 ROUGE pt [r] | **POS (shortlist #2)**: our `select()` is a single-pass ratio greedy on a 0.5 s grid with a no-overlap constraint. Swap/shift local search on the same objective is untested |
| 51 | Golovin & Krause, *Adaptive Submodularity*, JAIR 2011, https://arxiv.org/abs/1003.3967 ; Kulesza & Taskar, *Determinantal Point Processes for ML*, 2012, https://arxiv.org/abs/1207.6083 | COV | Stochastic coverage: expected coverage under independent item uncertainty stays submodular; DPPs model diversity. | Theory | USED: E[distinct pairs] = Σ(1−Π(1−p_i)) is exactly the independent-uncertainty coverage. Its weak spot is calibration (calib_pow = 2 fitted) and correlated errors, which are not modelled |
| 52 | Hu et al., *TransRAC: Repetitive Action Counting*, CVPR 2022, https://arxiv.org/abs/2204.01018 ; Lempitsky & Zisserman, *Learning to Count Objects in Images*, NeurIPS 2010 | COV | Density-map regression whose integral gives counts. | — | USED: our singleton-weighted action-density model (coverage 38.55 → 39.34) is this idea for "distinct actions per window" |

### D. XR, egocentric AI and efficiency, 2022–2026

| # | Paper | Maps to | Key idea | Results | Outcome |
|---|---|---|---|---|---|
| 53 | Grauman et al., *Ego4D*, CVPR 2022, https://arxiv.org/abs/2110.07058 ; *Ego-Exo4D*, CVPR 2024, https://arxiv.org/abs/2311.18259 | all | 3,670 h egocentric video; NLQ/MQ/PNR/AV benchmarks. Ego-Exo4D adds keystep recognition and proficiency. | — | USED indirectly (LaViLa and UniVTG pretraining). The Ego4D Moment Queries detector recipe = ActionFormer |
| 54 | Mangalam et al., *EgoSchema*, NeurIPS 2023, https://arxiv.org/abs/2308.09126 ; Chen et al., *Video ReCap*, CVPR 2024, https://arxiv.org/abs/2402.13250 | XR | Very long-form egocentric QA (3-min clips); hierarchical captioning of hour-long egocentric video. | EgoSchema: models far below humans (about 33% vs 76%) [r] | NF for scoring. Good deck motivation: summaries and highlights as the input layer for long-video QA |
| 55 | Bolya et al., *Token Merging (ToMe)*, ICLR 2023, https://arxiv.org/abs/2210.09461 ; Chen et al., *FastV*, ECCV 2024, https://arxiv.org/abs/2403.06764 ; Yang et al., *VisionZip*, CVPR 2025, https://arxiv.org/abs/2412.04467 ; Tao et al., *DyCoke*, CVPR 2025, https://arxiv.org/abs/2411.15024 | XR | Training-free token merging/pruning: about 2× ViT throughput with about 0.2–0.4% accuracy loss (ToMe on video MAE ViT-L); FastV drops 50% of visual tokens after layer 2 (about 45% FLOPs saved). | as stated [r] | XR-efficiency narrative only. Could be applied to the LaViLa TimeSformer pass to speed it up, but it changes predictions → shifts calibration (the recurring failure mode). NF tonight |
| 56 | Shen et al., *LongVU*, 2024, https://arxiv.org/abs/2410.17434 ; Li et al., *EgoPrune*, arXiv 2025, https://arxiv.org/abs/2507.15428 ; Chen et al., *VideoLLM-online*, CVPR 2024, https://arxiv.org/abs/2406.11816 | XR | Spatio-temporal adaptive compression (DINO similarity). EgoPrune: egomotion-aware, perspective-aligned redundancy filtering, >99% accuracy kept on an edge device (Jetson Orin NX). Streaming egocentric video LLM. | as stated [v for EgoPrune] | **Deck/XR transfer**: our 15% action-coverage highlights act as a *task-aware frame pruner* (85% fewer frames, so ≈85% fewer visual tokens at a fixed tokens/frame), complementary to EgoPrune's within-frame pruning. Not scored |
| 57 | Shan et al., *Understanding Human Hands in Contact at Internet Scale* (100DOH), CVPR 2020, https://arxiv.org/abs/2006.06669 ; Zhang et al., *EgoHOS*, ECCV 2022, https://arxiv.org/abs/2208.03826 | COV | Hand-object contact state detector; egocentric hand-object segmentation. Contact transitions mark action boundaries. | — | NF tonight: a frame-level detector pass over about 10+ h of 1080p EPIC val video. Would be the most promising *new position signal* for coverage (GT positions are worth +11.6 coverage) with more time |
| 58 | Lai et al., *GLC: In the Eye of Transformer — Global-Local Correlation for Egocentric Gaze Estimation*, BMVC 2022, https://arxiv.org/abs/2208.04464 ; Engel et al., *Project Aria*, 2023, https://arxiv.org/abs/2308.13561 ; Lv et al., *Aria Everyday Activities*, 2024, https://arxiv.org/abs/2402.13349 ; Gu et al., *EgoLifter*, ECCV 2024, https://arxiv.org/abs/2403.18118 ; Banerjee et al., *HOT3D*, 2024, https://arxiv.org/abs/2411.19167 ; Yang et al., *EgoLife*, CVPR 2025, https://arxiv.org/abs/2503.03803 | XR, F1 | Gaze prediction from video; always-on glasses platform with eye-tracking/SLAM; open-world 3D egocentric segmentation; 3D hand-object tracking; week-long life-logging assistant. | — | NF for scoring (no gaze or 3D in our inputs). Deck: on a real XR headset, eye-tracking gaze (Xu 2015, #49) and SLAM egomotion (EgoPrune) are free signals that would replace our motion proxy |

(Rows 45, 48–52 and 55–58 group 2–6 related works, so the table cites 70+ papers in 58 rows; 52 are primary entries.)

---

## (ii) Map from each scored sub-problem to the best evidence

**Pair mAP (25%), now 22.54.** The bottleneck is noun labels (oracle noun 17.8 → 40.7). But every way to get better noun probabilities needs a new GPU pass over the video:
- AVION (#14) would add about 3.5 action top-1 over LaViLa.
- LaViLa TTA (multi-crop/clip) would also need another pass.
- Both cost about 3–4 h for 138 videos and are infeasible tonight.

Detector swaps and ensembles (#3–8, #11) are empirically dead here: TriDet, the AF seed ensemble and the fine-tune soups each gave ≤ +0.2 mAP and lost coverage.

What remains is **output-only post-processing**, which cannot move coverage because selection builds its own action set (verified in v7):
- (a) **Which 2000 actions survive the cap** (Dave et al. #21: +1.2 AP / +2.9 APr from cap-aware re-ranking on LVIS). This is the strongest untested lever. Our cap is applied by raw score *before* the v7 context rescoring (`pipeline.py` lines 146 and 158–164), so even the context score does not yet influence which actions survive.
- (b) Boundary voting and verb/noun boundary fusion (#9, #23): helps the strict tIoU 0.4–0.5 thresholds.
- (c) More context (#19, #20): mostly used up by v7. MTCN shows that count-based LMs add little.

**Action coverage (40%), now 40.95.** The bottleneck is positions (GT positions → 52.6; GT labels on our detections → +0.7 only). The literature offers two routes:
- (a) A better position/actionness signal, such as hand-object contact (#57) or a stronger detector. Both are infeasible tonight or already NEG.
- (b) A better optimiser for the same objective (#50). Our greedy is single-pass, with no swap/shift refinement, and its only tie to the metric is the calibrated proxy E[distinct pairs] + 1.2·density. The objective is monotone submodular, but the non-overlap constraint plus the 15% budget make it a knapsack-like independence system, where greedy can be noticeably sub-optimal. Local search usually recovers a few % of the objective. Expected metric gain is small, but it is the only selection-side lever left that does not touch the calibrated inputs.

**Highlight F1 (25%), now 31.64; human-vs-consensus 37.6.** FPVSum GT is short motion bursts. Motion is the best single signal (F1 0.27), and semantic novelty is useless. Otani (#28) warns that F1 is dominated by structure.

The only rule-compliant learned signals are highlight models pretrained on non-FPVSum data:
- UniVTG zero-shot (#44, Ego4D/VideoCC-pretrained, raw-video demo, < 4 GB) has domain-level evidence on 4 FPVSum-overlapping YouTube HL domains: zero-shot 54–66 mAP vs LIM-S 49–67.
- UMT/QD-DETR (#42–43) need feature recipes we lack.
- LLM/VLM summarizers (#34–36) are too heavy and prompt-sensitive.

Expected gain is uncertain and could be negative. Any fusion weight evaluated on FPVSum is selection on the test set, so a single pre-declared configuration is the only honest test.

**Integrity flag.** `sports.py` and README §1 say the 11 ridge weights were *fitted on FPVSum human importance curves* (LOCO-validated). That conflicts with the constraint "nothing fitted on FPVSum". This is already in the submission; it is flagged here so the user can decide on the README wording. The clean alternative is a fixed motion-only or hand-set weighting (motion alone F1 0.27, about −1 weighted), or refitting on YouTube Highlights/TVSum (#26, #38), which costs hours of downloads.

**Label (10%), now 97.96.** At the ceiling; the miss is the private-video placeholder. Nothing to gain.

---

## (iii) Novel XR and egocentric approaches 2023–2026, and transfer potential

1. **Egomotion-aware token pruning (EgoPrune 2025, LongVU 2024, DyCoke/VisionZip 2025).** These prune *within* the kept frames, exploiting head-motion continuity. Our method prunes *which* frames are kept, by a task objective (distinct actions). Transfer: deck narrative and an XR-efficiency number ("15% of frames keep 41% of distinct actions, vs 19% for uniform; 2.1× better per token"). It does not transfer to the score.
2. **Streaming egocentric assistants (VideoLLM-online 2024, EgoLife 2025, Video ReCap 2024).** These argue for causal, online summarization. Our density and coverage greedy could run online with a sliding budget. Deck only.
3. **Narration-supervised egocentric foundation models (LaViLa → AVION → EgoVideo/InternVideo2 challenge entries).** These are the main driver of EPIC gains in 2023–2026 (CausalTAD 31.97 test action mAP with InternVideo2 features vs about 22 for SlowFast-era detectors). The transfer is real but needs a new backbone pass. That is the largest legitimate lever for pair mAP and coverage given a day or more of compute, not tonight.
4. **Hand-object contact and keystep structure (100DOH, EgoHOS, Ego-Exo4D keysteps, 2026 contact-transition detection).** Contact onsets and offsets are natural action midpoint anchors. Coverage is position-limited (+11.6 available with GT positions), so this is the most promising *new* coverage signal for future work. It needs a per-frame detector pass.
5. **Gaze and Aria (Xu 2015, GLC 2022, Aria Everyday Activities 2024, EgoEverything 2026).** On real XR hardware, gaze fixations plus SLAM motion are free, high-value importance signals. None are present in FPVSum or EPIC inputs, so there is no transfer to the score.
6. **Zero-shot LLM/VLM summarizers (Prompts to Summaries 2025, CAPS 2025, synthetic-image CLIP 2026).** Strong on TVSum/SumMe, and the 2025 egocentric comparison says GPT-4o-class models beat specialised summarizers on Ego-Exo4D. They are too heavy for a 6 GB GPU tonight, and FPVSum's motion-burst GT favours low-level cues.
7. **Large-vocabulary evaluation lessons (Dave et al., LVIS).** This is old and not XR, but it is the most directly transferable "novel" idea for our scorer: a shared per-video cap makes per-class mAP sensitive to how the cap is allocated.

---

## (iv) Ranked shortlist for tonight (at most 5)

All mAP ideas are **output-only**: they act on `out`, before or at the 2000 cap. Coverage must stay exactly 40.95 [39.68 41.66]; use that as a sanity check. Harness: `tools/dev_v3.py`, where `V.actions(C=…)` gives per-video (out, density, sel) and `V.score(name, res)` prints pooled and per-fold mAP and coverage. Keep a change only if **both** participant folds improve and the pooled weighted gain is above about +0.05. Each idea needs a small config flag in `pipeline.epic_actions` so that dev_v3 and run.sh share one code path, the same pattern as `NOUN_CTX`.

### 1. Cap-aware allocation of the 2000 output actions (Dave et al. 2021, #21)
- **Mechanism.** Today `acts = sorted(by score)[:2000]` runs *before* the noun-context rescoring. Change only *which* candidates survive the cap. The emitted scores stay unchanged, so within-class ranking across videos is untouched. Tested variants:
  - (a) Apply the v7 context rescoring to the full merged candidate set first, then cap.
  - (b) Keep the top-2000 by `score / r^β`, where r is the instance's rank *within its (verb, noun) pair in this video*. Each pair's first instance is unaffected, and surplus repeats of frequent pairs such as (take, plate) make room for rarer pairs. Grid β ∈ {0.1, 0.2, 0.35}.
  - (c) Alternative to (b): keep by `score · (n_train(pair)+10)^(−α)` with train-split pair counts (already in `epic_prior.npz` or the annotations). Grid α ∈ {0.05, 0.1, 0.2}.
- **Why it can work.** Our mAP is the mean over a few thousand val pair classes, mostly rare, so each rare-pair TP is worth roughly 1/N_c of a class. On LVIS, cap-aware re-ranking gave +1.2 AP / +2.9 APr with no model change.
- **Expected gain.** mAP +0.2 to +0.8, so **+0.05 to +0.2 weighted**. CPU only, about 0 GPU.
- **CV design.** Add `C.CAP_MODE`/`C.CAP_BETA`. Run (a) alone, (a)+(b) for 3 β values and (a)+(c) for 3 α values: 7 runs × (about 2–4 min each on cached features and dets). Check that coverage is identical.
- **Kill.** Kill it if either fold's mAP drops for every setting, or if the pooled gain is < +0.15 mAP. Choose β on fold 1 and confirm on fold 2, then the reverse; keep it only if both choices agree or both still improve.

### 2. Local-search refinement of the coverage greedy (Khuller et al. 1999; Lin & Bilmes 2010, #50)
- **Mechanism.** Same inputs, same objective: E[distinct pairs] + dens_w · density mass. After the current greedy, iterate:
  - (i) **shift** each window by ±0.5 … ±2 s;
  - (ii) **swap** each window for the best non-overlapping unused candidate;
  - (iii) **resize** between 4 and 5 s while respecting the budget.
  Accept only strict objective increases, then run `_fill` and `_widen` as now. Variant: also try the Khuller "best of greedy vs greedy started from each of the top-5 single windows".
- **Why it can work.** The ratio greedy with a no-overlap constraint is myopic. A small objective gain turns into a coverage gain only to the extent that the proxy is calibrated; calib_pow = 2 was CV-fitted on this proxy, which is encouraging.
- **Expected gain.** Coverage +0.1 to +0.4, so **+0.04 to +0.16 weighted**. Risk: a better proxy optimum can overfit noise in p. CPU only; selection per video goes from ms to about 1 s.
- **CV design.** Write `SC.select_ls` and monkey-patch `SC.select` in dev_v3. Run `V.score("ls", res, cov={...})` for shift-only, shift+swap and shift+swap+resize. Also log the mean objective gain per video to confirm the optimiser actually improves it.
- **Kill.** Kill it if either fold's coverage drops, if the pooled gain is < +0.2, or if the objective improves by < 1% (nothing to gain).

### 3. Boundary voting in the duplicate merge plus verb/noun boundary fusion (EgoAction 2026 #9; box voting #23; Soft-NMS voting #22)
- **Mechanism, output actions only.**
  - (a) In `merge_duplicates`, set the kept instance's [s, e] to the score-weighted mean of all same-pair members with tIoU ≥ thr_v (thr_v ∈ {0.6, 0.75}). The noisy-or score is unchanged.
  - (b) Where a noun detection of the same noun overlaps the action at tIoU ≥ 0.5, move the boundaries towards it with weight c_n/(c_v+c_n) (EgoAction's DWF).
- **Why.** Our action segments are verb segments; mAP at tIoU 0.4–0.5 is where localisation errors bite (v1 per-threshold mAP fell 16.2 → 11.6 from tIoU 0.1 to 0.5).
- **Expected gain.** mAP +0.1 to +0.3, so **+0.03 to +0.08 weighted**. CPU only.
- **CV design.** Flags `C.MERGE_VOTE` and `C.NOUN_BFUSE`. Run 2 × 2 + 2 settings and check the per-tIoU mAP breakdown: the gain should appear at 0.4–0.5.
- **Kill.** Kill it if either fold drops or if the gain is < +0.1 mAP.

### 4. Context rescoring v2: verb context plus a global video-level presence term (Kazakos BMVC'21 #19; Pato CVPR'20 #20)
- **Mechanism.** Extend v7 with two output-only factors:
  - (a) A verb–verb co-occurrence context from EPIC train annotations, built the same way as M10 in `build_nounctx.py`.
  - (b) A τ → ∞ (whole-video) noun-presence factor, analogous to video-level classification fusion in TAD. For example, `(0.1 + ctx_global)^γ2`, where ctx_global = the video-mean LaViLa/segcls noun probability, γ2 ∈ {0.05, 0.1}.
- **Evidence.** MTCN: the context window gives +4 action top-1, the LM only +0.5 more, and an N-gram LM nothing. v7 already took about +0.2 mAP, so expect diminishing returns.
- **Expected gain.** mAP +0.05 to +0.25, so **+0.01 to +0.06 weighted**. CPU only.
- **CV design.** Grid of 2 × 2 settings plus τ ∈ {10, 30 s} for the existing noun term (if τ was not already gridded). Apply it on top of #1(a), since context then also decides the cap.
- **Kill.** Kill it if either fold drops or if the gain is < +0.1 mAP.

### 5. UniVTG zero-shot saliency as an extra sports-importance signal (UniVTG ICCV'23 #44)
- **Mechanism.** Run the MIT-licensed UniVTG CLIP-B/16 *pretrained-only* checkpoint (Ego4D narrations + VideoCC; no FPVSum, TVSum or YouTube HL fine-tuning) through its `run_on_video` path on each sports video. Query = a fixed template from *our predicted label*, e.g. "an exciting moment of {label}". This is allowed: the label comes from video content, not the video ID. That gives a per-clip saliency curve. Add it to the current importance as `imp + w·z(saliency)`, with **w pre-declared = 0.215 (equal to our largest existing weight, motion_s5)**, no grid.
- **Evidence.** Zero-shot on the FPVSum-overlapping YouTube HL domains: parkour 65.9, skiing 64.5, surfing 54.0, skating 39.2 mAP (fine-tuned 73–84; LIM-S 49–67). TVSum zero-shot 67.2 vs LIM-S 56.3. The evidence is mixed: skating (39.2) is far below every supervised method.
- **Expected gain.** F1 −0.5 to +1.5, so about **−0.1 to +0.4 weighted**; median expectation about +0.1.
- **Cost.** GPU about 10–20 min for 48 videos (< 4 GB, < 1 s inference per video plus CLIP-B/16 features at its clip rate). Implementation 1.5–2 h. Risks: the checkpoint is hosted on Google Drive (our Drive downloads were throttled before), the demo code has to be vendored (third-party code execution needs user approval, as TriDet did), and preprocessing must match.
- **Test design.** dev_v3 is EPIC-only. Instead, use the cached 2 fps frames (`tools/fpv_cache.py`) and `tools/eval/score.py --only <48 FPVSum ids>`. Run **once** with the pre-declared w. Report the per-category F1 change (14 categories) as a pseudo-LOCO check.
- **Kill.** Kill it if the pooled F1 gain is < +0.5, if fewer than 9 of 14 categories improve, or if the setup is not working within 1.5 h. No second weight may be tried on FPVSum; that would be fitting on the eval set.

**Not shortlisted:**
- AVION or LaViLa-TTA noun ensembles (GPU 3–4 h)
- UMT/QD-DETR (feature-recipe risk)
- LLM/VLM summarizers (heavy)
- audio energy (unfittable weight, music overlays)
- hand-object contact anchors (frame-level pass over all of EPIC val)
- any detector retraining (repeatedly NEG)

### Realistic total and the odds of 42

| Idea | Expected weighted gain (realistic range) |
|---|---|
| 1 Cap-aware allocation | +0.05 to +0.20 |
| 2 Coverage local search | +0.00 to +0.16 |
| 3 Boundary voting / fusion | +0.00 to +0.08 |
| 4 Context v2 | +0.00 to +0.06 |
| 5 UniVTG saliency | −0.10 to +0.40 (kept only if it passes) |
| **Sum if every idea lands at its median** | **about +0.2 to +0.4 (→ about 39.9–40.1)** |
| Optimistic (all at the top of their range) | about +0.9 (→ about 40.6) |

Reaching **42.0 (+2.3)** would need about 2.5× the optimistic case. That amounts to a large coverage jump, which the literature ties only to better *positions* (new backbone or HOI passes, not feasible tonight), or a noun-classifier upgrade worth about +5 mAP (needs AVION-class models plus hours of GPU). **Honest odds of 42 tonight: about 2–3%. Odds of 40.0 or more: about 40–50%.**
