"""Channel mean / std of the released EPIC-100 SlowFast features over the TRAIN split -> weights/epic_train_featstats.npz
(reference statistics for per-video input recalibration in prism.tta)."""
import json, os
import numpy as np
AF = r"C:/prism_data/af_features/epic_kitchens"
db = json.load(open(os.path.join(AF, "annotations", "epic_kitchens_100_verb.json")))["database"]
s = ss = n = 0
for vid, v in db.items():
    p = os.path.join(AF, "features", vid + ".npz")
    if v["subset"] != "training" or not os.path.exists(p):
        continue
    f = np.load(p)["feats"].astype(np.float64)
    s = s + f.sum(0); ss = ss + (f ** 2).sum(0); n += len(f)
mu = s / n; sd = np.sqrt(np.maximum(ss / n - mu ** 2, 1e-12))
np.savez(r"C:/prism_data/weights/epic_train_featstats.npz", mean=mu.astype(np.float32), std=sd.astype(np.float32))
print(n, mu.mean(), sd.mean())
