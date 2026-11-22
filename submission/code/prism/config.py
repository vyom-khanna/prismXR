"""Central configuration. Everything path-like can be overridden with environment variables."""
from __future__ import annotations

import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))  # submission/ (TEAM_ID/) root

TEAM_ID = os.environ.get("PRISM_TEAM_ID", "Opus 5.5 High")
WEIGHTS_DIR = os.environ.get("PRISM_WEIGHTS", os.path.join(REPO_DIR, "weights"))
CACHE_DIR = os.environ.get("PRISM_CACHE", "")  # optional feature cache ('' = no cache)
CLASSES_JSON = os.path.join(CODE_DIR, "supported_classes.json")

# CLIP (label + routing + sports importance)
CLIP_ARCH = "ViT-L-14"
CLIP_WEIGHTS = os.path.join(WEIGHTS_DIR, "clip", "open_clip_fp16.bin")  # fp16 copy of the DataComp ViT-L/14
SLOWFAST_WEIGHTS = os.path.join(WEIGHTS_DIR, "SlowFast_model.pyth")      # model_state only
LABEL_NUM_FRAMES = 32

# Output rules (TEAM_README)
BUDGET = 0.15
MIN_SEG = 4.0
TARGET_FILL = 0.149  # fill the budget with a margin for 0.01 s rounding fixes (0.1495 tested: worse on CV) fill the budget (CV: fuller budget helps) with a margin for 0.01 s rounding fixes
DECIMALS = 2

# LaViLa (EPIC branch): classify the top distinct verb-detection segments
LAV_TOPK = 300        # segments per video (class-agnostic NMS at LAV_NMS); output actions use all of them
LAV_SEL_TOPK = 200    # highlight selection uses actions built from the top 200 only (2-fold CV: coverage drops with more)
LAV_NMS = 0.7
LAV_MATCH = 0.5       # a detection uses LaViLa's noun probs if it overlaps a classified segment by tIoU >= this
LAV_W = 1.0           # weight of LaViLa noun probs vs segment-classifier noun probs (chosen on a 21-video val subset)
LAV_ALT = 3           # extra actions: LaViLa top-k verbs x top-k nouns per classified segment
LAV_ALT_W = 0.25      # their score = LAV_ALT_W * (p_verb * p_noun) ** 0.75
MERGE_TIOU = 0.8      # same (verb, noun) predictions overlapping >= this are merged (noisy-or score); 2-fold CV

MIN_LEN_PAD = 0.02     # review #4: finalize grows windows to MIN_SEG + this (0.01 could round back to exactly 4.00)
SHORT_GREEDY = True    # review #2: budget < MIN_SEG -> still run the greedy coverage selection on one MIN_SEG window
SPORTS_WIDEN = True    # review #1: sports highlights grow to spend the leftover (< seg_len) budget, as EPIC does
PAIR_MAX = 12000       # review #5: output actions keep this many pairing candidates (the 2000 cap is applied after the merge);
                       # highlight selection keeps the 2000-candidate set (CV: the long tail costs coverage)
NOUN_CTX = (10.0, 0.15, 0.1)  # temporal noun-context rescoring of OUTPUT actions (sigma s, gamma, eps); None = off
CAP_REPEAT_BETA = 0.35  # which output actions survive the 2000 cap: rank by score / r**beta, r = rank of the instance within
                        # its (verb, noun) pair in this video (Dave et al. 2021 cap-aware AP); emitted scores unchanged. 0 = off
SEL_LOCAL_SEARCH = True # post-greedy swap/shift/resize local search on the coverage objective (Khuller et al. 1999)
