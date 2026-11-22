"""Train an extra ActionFormer seed on the EPIC-100 TRAIN split (released SlowFast features), for a detector ensemble.

    python tools/train_af_seed.py verb 1     -> C:/prism_data/weights/actionformer_seed1/epic_slowfast_verb_ema.pth.tar
    python tools/train_af_seed.py verb 11 ft -> fine-tune FROM the official checkpoint (short, low-lr; for weight-space soups)
                                                -> C:/prism_data/weights/actionformer_ft11/...

Mirrors the official ActionFormer recipe (AdamW, 5 warm-up epochs + cosine, EMA 0.999, grad clip 1.0, random crop/truncation)
using the vendored model code in submission/code/prism/actionformer. Validation videos are never read.
"""
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

from prism.actionformer.core.config import DEFAULTS, load_config
from prism.actionformer.modeling import make_meta_arch
from prism.actionformer.modeling.blocks import AffineDropPath, LayerNorm, MaskedConv1D, Scale

AF = r"C:/prism_data/af_features/epic_kitchens"
CFG = os.path.join(os.path.dirname(__file__), "..", "submission", "code", "prism", "actionformer", "configs")

kind, seed = sys.argv[1], int(sys.argv[2])
FT = len(sys.argv) > 3 and sys.argv[3] == "ft"
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
cfg = load_config(os.path.join(CFG, f"epic_slowfast_{kind}.yaml"), copy.deepcopy(DEFAULTS))
ds, opt = cfg["dataset"], cfg["opt"]
if FT:  # fine-tune: 1 warm-up + 4 cosine epochs at lr/4
    opt = dict(opt, warmup_epochs=1, epochs=4, learning_rate=opt["learning_rate"] / 4)
db = json.load(open(os.path.join(AF, "annotations", f"epic_kitchens_100_{kind}.json")))["database"]
items = []
for vid, v in db.items():
    if v["subset"] != "training" or not os.path.exists(os.path.join(AF, "features", vid + ".npz")):
        continue  # 10 of 495 train feature files are missing from the partial download
    with np.load(os.path.join(AF, "features", vid + ".npz")) as z:
        f = torch.from_numpy(np.ascontiguousarray(z["feats"].astype(np.float32).T))
    seg = np.array([a["segment"] for a in v["annotations"]], np.float32)
    lab = np.array([a["label_id"] for a in v["annotations"]], np.int64)
    items.append((vid, f, seg, lab, v["duration"]))
print(len(items), "train videos", flush=True)

FS, NF, FPS = ds["feat_stride"], ds["num_frames"], ds["default_fps"]
OFF = 0.5 * NF / FS


def sample(it):
    vid, f, seg, lab, dur = it
    segs = torch.from_numpy(seg * FPS / FS - OFF)
    labs = torch.from_numpy(lab)
    T, L = f.shape[1], ds["max_seq_len"]
    if T <= L:
        L = random.randint(max(round(ds["crop_ratio"][0] * T), 1), min(round(ds["crop_ratio"][1] * T), T))
    if L < T:
        for _ in range(200):
            st = random.randint(0, T - L)
            left = torch.clamp(segs[:, 0], min=st - OFF)
            right = torch.clamp(segs[:, 1], max=st + L + OFF)
            ratio = (right - left).clamp(min=0) / (segs[:, 1] - segs[:, 0]).abs()
            keep = ratio >= ds["trunc_thresh"]
            if keep.any():
                break
        f = f[:, st:st + L].clone()
        segs = torch.stack((left[keep], right[keep]), 1) - st
        labs = labs[keep]
    return {"video_id": vid, "feats": f, "segments": segs, "labels": labs, "fps": FPS, "duration": dur,
            "feat_stride": FS, "feat_num_frames": NF}


model = make_meta_arch(cfg["model_name"], **cfg["model"]).cuda()
if FT:
    sd = torch.load(os.path.join(r"C:/prism_data/weights/actionformer", f"epic_slowfast_{kind}_ema.pth.tar"),
                    map_location="cpu", weights_only=False)
    sd = sd.get("state_dict_ema", sd.get("state_dict", sd))
    model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}, strict=True)
decay, no_decay = set(), set()
for mn, m in model.named_modules():
    for pn, _ in m.named_parameters(recurse=False):
        fpn = f"{mn}.{pn}" if mn else pn
        if pn.endswith("bias") or pn.endswith("rel_pe"):
            no_decay.add(fpn)
        elif pn.endswith("weight") and isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)):
            decay.add(fpn)
        elif pn.endswith("weight") and isinstance(m, (LayerNorm, torch.nn.GroupNorm)):
            no_decay.add(fpn)
        elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
            no_decay.add(fpn)
P = dict(model.named_parameters())
assert set(P) == decay | no_decay and not decay & no_decay, set(P) - (decay | no_decay)
optim = torch.optim.AdamW([{"params": [P[n] for n in sorted(decay)], "weight_decay": opt["weight_decay"]},
                           {"params": [P[n] for n in sorted(no_decay)], "weight_decay": 0.0}], lr=opt["learning_rate"])
B = cfg["loader"]["batch_size"]
ipe = len(items) // B                                   # drop_last, as in the official loader
W, E = opt["warmup_epochs"] * ipe, (opt["warmup_epochs"] + opt["epochs"]) * ipe
sched = torch.optim.lr_scheduler.LambdaLR(
    optim, lambda s: s / W if s < W else 0.5 * (1 + math.cos(math.pi * (s - W) / (E - W))))
ema = copy.deepcopy(model).eval()
clip = cfg["train_cfg"]["clip_grad_l2norm"]

for ep in range(opt["warmup_epochs"] + opt["epochs"]):
    model.train()
    order = list(range(len(items)))
    random.shuffle(order)
    t0, tot = time.time(), 0.0
    for b in range(ipe):
        batch = [sample(items[i]) for i in order[b * B:(b + 1) * B]]
        optim.zero_grad(set_to_none=True)
        loss = model(batch)["final_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step(); sched.step()
        with torch.no_grad():
            for e_, m_ in zip(ema.state_dict().values(), model.state_dict().values()):
                e_.copy_(0.999 * e_ + 0.001 * m_)
        tot += loss.item()
    print(f"{kind} {'ft' if FT else 'seed'}{seed} epoch {ep} loss {tot / ipe:.4f} {time.time() - t0:.0f}s", flush=True)

out = rf"C:/prism_data/weights/actionformer_{'ft' if FT else 'seed'}{seed}"
os.makedirs(out, exist_ok=True)
torch.save({"state_dict_ema": ema.state_dict()}, os.path.join(out, f"epic_slowfast_{kind}_ema.pth.tar"))
print("saved", flush=True)
