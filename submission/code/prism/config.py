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
TARGET_FILL = 0.149  # fill the budget (CV: fuller budget helps) with a margin for 0.01 s rounding fixes
DECIMALS = 2
