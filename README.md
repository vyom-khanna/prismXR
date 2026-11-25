# TIET_Opus5.5high — PRISM XR Hackathon 2026, Track 04: Action-Aware Video Summarization

Team **Opus5.5high** · Thapar Institute of Engineering and Technology (TIET)

**The submission package is [`TIET_Opus5.5high/`](TIET_Opus5.5high/)** (TEAM_ID layout: `run.sh`, `code/`, `weights/`,
`requirements.txt`, `outputs/`, `README.md`, `Dockerfile`). Its [README](TIET_Opus5.5high/README.md) has the architecture,
setup, data used, results and the integrity disclosure.

```bash
git clone https://github.com/vyom-khanna/prismXR.git
cd prismXR/TIET_Opus5.5high
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
./run.sh <video.mp4> <out.json>
```

## Video (5 min): architecture, how it works, results

[![Watch the video](demo/thumbnail.png)](demo/TIET_Opus5.5high_video.mp4)

**[▶ Watch the video (demo/TIET_Opus5.5high_video.mp4, 5 min, 1080p)](demo/TIET_Opus5.5high_video.mp4)**:
the final v10 architecture, how each stage works, real outputs, a brief history from v1 to v10, results, and
where the system can be used.

**Demo video (Google Drive):** https://drive.google.com/drive/folders/1gG_aLcU4Mp3Nq-dPiHhkLjqznKbv4hbD?usp=drive_link

Local score on all 187 manifest videos (our re-implementation of the metrics): **weighted total 40.3**
(pair mAP 23.0 · coverage 42.1 · highlight F1 31.6 · label accuracy 98.0).

## How we got to v10

![Weighted total from v1 (35.2) to v10 (40.3)](demo/version_history.png)

Ten measured versions, from a plain detector pipeline (v1, 35.2) to the final system (v10, 40.3). A change was kept only
if it improved both halves of a cross-validation split by participant; the full log, including rejected ideas, is in
[`PROGRESS.md`](PROGRESS.md).

| Folder | Contents |
|---|---|
| `TIET_Opus5.5high/` | the submission: inference code, all model weights (large ones as <100 MB parts, rejoined by run.sh), 187 output JSONs |
| `tools/` | development only: training scripts for our models, evaluation, tuning and integrity checks (never imported at inference) |
| `research/` | literature review behind the design choices |
| `demo/` | the 5-minute video (architecture, how it works, results) and its thumbnail |
| `AI_DISCLOSURE.md` | how AI assistance (Claude Code) was used, and the AI models inside the system |
| `PROGRESS.md` | experiment log (every change, kept or rejected, with its cross-validation result) |
| `TIET_Opus5.5high_Submission.pptx` | presentation deck |
| `Student package_track04/` | the organiser's student kit (manifest, classes, TEAM_README) |
