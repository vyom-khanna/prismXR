"""CLIP wrapper: frame embeddings + zero-shot video label (also used to route EPIC vs sports)."""
from __future__ import annotations

import numpy as np
import torch

from . import config as C

# Several first-person phrasings per class; averaged text embeddings (prompt ensembling).
CLASS_PROMPTS = {
    "cooking": ["a first-person view of hands cooking in a kitchen", "an egocentric video of preparing food on a kitchen counter",
                "a point-of-view shot of washing dishes at a kitchen sink", "a head-mounted camera view of a person cooking at a stove"],
    "bike": ["a first-person view of riding a mountain bike on a trail", "a helmet camera view of cycling downhill",
             "a point-of-view shot of riding a bicycle on a road",
             "a first-person view of mountain biking with handlebars of a bicycle visible",
             "a downhill mountain bike race on a dirt track"],
    "bikepolo": ["a first-person view of playing bike polo", "a helmet camera view of bicycle polo with mallets and a ball on a court",
                 "a point-of-view shot of a hardcourt bike polo game"],
    "boxing": ["a first-person view of boxing sparring", "a point-of-view shot of a boxing match with gloves",
               "a head camera view of punching an opponent in a boxing ring",
               "a first-person view of martial arts sparring training in a gym with boxing gloves"],
    "horse": ["a first-person view of riding a horse", "a point-of-view shot from horseback showing the horse's head and ears",
              "a helmet camera view of horse riding and jumping"],
    "jumping": ["a first-person view of skydiving", "a point-of-view shot of a parachute or base jump",
                "a helmet camera view of free fall from an airplane", "a first-person view of a bungee jump"],
    "longboarding": ["a first-person view of longboarding down a road", "a point-of-view shot of riding a longboard skateboard downhill on asphalt"],
    "motor": ["a first-person view of riding a motorcycle", "a helmet camera view of motocross or motorbike racing",
              "a point-of-view shot of riding a dirt bike",
              "a first-person view of a motorcycle with the speedometer and fuel tank visible",
              "a motorbike racing on a road at high speed"],
    "parkour": ["a first-person view of parkour", "a point-of-view shot of freerunning and jumping over rooftops and walls"],
    "plane": ["a first-person view from an airplane cockpit", "a point-of-view shot of flying a small plane or glider",
              "a cockpit camera view of piloting an aircraft",
              "a first-person view from an FPV drone or RC model plane flying low over a park",
              "fpv drone footage flying fast between trees over grass",
              "a view through the windshield of a small airplane cockpit with instrument panel"],
    "rockclimbing": ["a first-person view of rock climbing", "a point-of-view shot of climbing a cliff face with hands on rock"],
    "scuba": ["a first-person view of scuba diving underwater", "a point-of-view shot of diving near coral reef and fish"],
    "skate": ["a first-person view of skateboarding in a skate park", "a point-of-view shot of doing skateboard tricks on a ramp",
              "a first-person view of ice skating"],
    "ski": ["a first-person view of skiing down a snowy slope", "a helmet camera view of skiing or snowboarding in the mountains"],
    "surfing": ["a first-person view of surfing a wave", "a point-of-view shot from a surfboard in the ocean"],
}
# 'other' is never predicted directly; it's used when no class is confident (see predict_label).

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class Clip:
    def __init__(self, device: str | None = None):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = open_clip.create_model_and_transforms(C.CLIP_ARCH, pretrained=C.CLIP_WEIGHTS)
        self.model = model.eval().to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.tok = open_clip.get_tokenizer(C.CLIP_ARCH)
        self._class_emb = None

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray, batch: int = 64) -> np.ndarray:
        """frames uint8 [N, 224, 224, 3] -> L2-normalised float32 [N, D]."""
        outs = []
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i:i + batch]).permute(0, 3, 1, 2).float().div_(255)
            x = ((x - MEAN) / STD).to(self.device)
            if self.device == "cuda":
                x = x.half()
            f = self.model.encode_image(x).float()
            outs.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        return torch.cat(outs).numpy() if outs else np.zeros((0, 768), np.float32)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        t = self.tok(texts).to(self.device)
        f = self.model.encode_text(t).float()
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    def class_embeddings(self):
        if self._class_emb is None:
            names, embs = [], []
            for k, ps in CLASS_PROMPTS.items():
                e = self.encode_text(ps).mean(0)
                embs.append(e / np.linalg.norm(e))
                names.append(k)
            self._class_emb = (names, np.stack(embs))
        return self._class_emb

    def predict_label(self, frame_emb: np.ndarray, other_thresh: float = 0.0) -> tuple[str, float, dict]:
        """Average per-frame softmax over class prompts -> (name, score, all_probs)."""
        names, ce = self.class_embeddings()
        logit_scale = float(self.model.logit_scale.detach().exp())
        logits = logit_scale * frame_emb @ ce.T
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        mean = p.mean(0)
        i = int(np.argmax(mean))
        if mean[i] < other_thresh:
            return "other", float(1 - mean[i]), dict(zip(names, mean.tolist()))
        return names[i], float(mean[i]), dict(zip(names, mean.tolist()))
