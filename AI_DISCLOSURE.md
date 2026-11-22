# AI Disclosure — TIET_Opus5.5high

## AI assistance used during development
This project was developed with heavy use of an AI coding assistant: **Claude Code** (Anthropic), mainly with the
Claude Opus 5.5 and Claude Fable 5.2 models. The team directed the work, set the goals and constraints, and reviewed
and approved the decisions. The assistant was used for:

- writing and refactoring the inference code (`TIET_Opus5.5high/code/`), training scripts and evaluation tools (`tools/`);
- running and analysing experiments (every change was kept only if it improved both folds of a participant-split
  cross-validation; see `PROGRESS.md`, which logs all kept and rejected experiments);
- code review, the literature review (`research/literature_review.md`), the README, and the presentation deck.

## AI models inside the submitted system
The pipeline runs fully locally and calls no external AI service or API. It uses these pretrained models, all listed
in `TIET_Opus5.5high/README.md` with their sources and licences:

- OpenCLIP ViT-L/14 (DataComp.XL), zero-shot video labelling;
- SlowFast-R50 and ActionFormer, trained by their authors on the EPIC-KITCHENS-100 train split;
- LaViLa TimeSformer-L, fine-tuned by its authors on EPIC-KITCHENS-100 train (verb/noun heads only);
- our own small models (segment classifier, action-density model, noun co-occurrence, noun prior), trained by us on
  EPIC-KITCHENS-100 train data only.

## Data and integrity
No ground-truth labels, video-ID lookups or per-video constants are used at inference. The use of the evaluation
annotations during development (FPVSum ridge weights; EPIC hyper-parameter selection) is disclosed in
`TIET_Opus5.5high/README.md`, section 5.
