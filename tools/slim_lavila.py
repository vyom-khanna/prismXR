"""Slim the LaViLa EK100-CLS TimeSformer-L checkpoint: keep the visual backbone + verb/noun heads in fp16.
The action head (indexes LaViLa's train+val pair list) is dropped on purpose.

    C:/prism_env/Scripts/python.exe tools/slim_lavila.py
"""
import pathlib

import torch

pathlib.PosixPath = pathlib.WindowsPath  # checkpoint args were pickled on Linux

SRC = r"C:/prism_data/weights/lavila/tsf_l_ek100_cls.pth"
DST = r"C:/prism_data/weights/lavila/tsf_l_ek100_vn_fp16.pth"

ck = torch.load(SRC, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
keep = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()
        if k.startswith("visual.") or k.startswith("fc_cls.0.") or k.startswith("fc_cls.1.")}
print("dropped:", sorted(k for k in sd if k not in keep))
print("heads:", {k: tuple(v.shape) for k, v in keep.items() if k.startswith("fc_cls")})
torch.save({"state_dict": keep}, DST)
print("saved", DST, sum(v.numel() for v in keep.values()) / 1e6, "M params")
