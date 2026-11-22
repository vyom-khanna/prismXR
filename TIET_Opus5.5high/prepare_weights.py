"""Download the public third-party checkpoints and convert them to the exact files the pipeline loads.

    pip install gdown            # only needed for the two ActionFormer files on Google Drive
    python prepare_weights.py    # writes into ./weights (or $PRISM_WEIGHTS)

Our own small models (epic_prior.npz, epic_segcls.pt, epic_density.pt, epic_nounctx.npz) are committed in weights/.
Third-party files are too large for GitHub, so they are fetched from their authors' public links:

  SlowFast-R50 EPIC-100   (epic-kitchens/epic-kitchens-slowfast, Dropbox)  -> SlowFast_model.pyth          (model_state only)
  ActionFormer EPIC verb/noun (happyharrycn/actionformer_release, Drive)    -> actionformer/*_ema.pth.tar   (EMA weights only)
  OpenCLIP ViT-L/14 DataComp.XL (Hugging Face, laion)                       -> clip/open_clip_fp16.bin      (fp16 copy)
  LaViLa TimeSformer-L EK100-CLS (facebookresearch/LaViLa)                  -> lavila/tsf_l_ek100_vn_fp16.pth (visual + verb/noun heads, fp16)

The conversions only drop unused entries and cast to fp16; no values are changed otherwise.
Run `python prepare_weights.py --check` to compare with weights/MD5SUMS.txt (the checksums of the files we used;
torch pickles can differ byte-wise across torch versions, so a mismatch there is not by itself an error).
"""
import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.environ.get("PRISM_WEIGHTS", os.path.join(HERE, "weights"))
RAW = os.path.join(W, "_downloads")

SLOWFAST_URL = "https://www.dropbox.com/s/uxb6i2xkn91xqzi/SlowFast.pyth?dl=1"
AF_DRIVE = {"verb": "1Ta4ggKSj2YcszSrDbePlHe1ECF1CFKK4", "noun": "1OTlxeiWj8JE9n1-LsRYogHmqgUdsE5PR"}
CLIP_URL = "https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/resolve/main/open_clip_pytorch_model.bin"
LAVILA_URL = ("https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_cls/"
              "clip_openai_timesformer_large.ft_ek100_cls.ep_0090.md5sum_4a2509.pth")


def fetch(url, dst):
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("download", url)
    urllib.request.urlretrieve(url, dst + ".part")
    os.replace(dst + ".part", dst)
    return dst


def fetch_drive(fid, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.check_call([sys.executable, "-m", "gdown", "-q", fid, "-O", dst])
    return dst


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)  # authors' public checkpoints


def save(obj, dst):
    import torch
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(obj, dst)
    print("wrote", dst)


def half(sd):
    return {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}


def slowfast(src, dst):
    save({"model_state": load(src)["model_state"]}, dst)


def actionformer(zip_path, dst):
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith(".pth.tar")][0]
        tmp = os.path.join(RAW, os.path.basename(name))
        with open(tmp, "wb") as f:
            f.write(z.read(name))
    save({"state_dict_ema": load(tmp)["state_dict_ema"]}, dst)


def clip(src, dst):
    save(half(load(src)), dst)


def lavila(src, dst):
    import pathlib
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath  # checkpoint args were pickled on Linux
    ck = load(src)
    sd = ck.get("state_dict", ck)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    keep = {k: v for k, v in sd.items()
            if k.startswith("visual.") or k.startswith("fc_cls.0.") or k.startswith("fc_cls.1.")}
    save({"state_dict": half(keep)}, dst)


def check():
    ok = True
    for line in open(os.path.join(W, "MD5SUMS.txt")):
        md5, name = line.split(maxsplit=1)
        p = os.path.join(W, name.strip().lstrip("*").lstrip("./"))
        if not os.path.exists(p):
            print("MISSING ", p); ok = False; continue
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        print("OK      " if h.hexdigest() == md5 else "DIFFERS ", p)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only compare files with MD5SUMS.txt")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    out = lambda *p: os.path.join(W, *p)
    if not os.path.exists(out("SlowFast_model.pyth")):
        slowfast(fetch(SLOWFAST_URL, os.path.join(RAW, "SlowFast.pyth")), out("SlowFast_model.pyth"))
    for k, fid in AF_DRIVE.items():
        if not os.path.exists(out("actionformer", f"epic_slowfast_{k}_ema.pth.tar")):
            actionformer(fetch_drive(fid, os.path.join(RAW, f"af_epic_{k}.zip")),
                         out("actionformer", f"epic_slowfast_{k}_ema.pth.tar"))
    if not os.path.exists(out("clip", "open_clip_fp16.bin")):
        clip(fetch(CLIP_URL, os.path.join(RAW, "open_clip_pytorch_model.bin")), out("clip", "open_clip_fp16.bin"))
    if not os.path.exists(out("lavila", "tsf_l_ek100_vn_fp16.pth")):
        lavila(fetch(LAVILA_URL, os.path.join(RAW, "lavila_tsf_l_ek100_cls.pth")), out("lavila", "tsf_l_ek100_vn_fp16.pth"))
    print("done; raw downloads in", RAW, "can be deleted")
    check()


if __name__ == "__main__":
    main()
