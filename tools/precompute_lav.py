"""DEV: extend the LaViLa cache to the top-K NMS'd verb segments for all 138 EPIC val videos (incremental).

    C:/prism_env/Scripts/python.exe tools/precompute_lav.py 300
"""
import glob
import sys
import time

import epic_dev as E
import dev_v3 as V
from prism import config as C, pipeline as PL
from prism.lavila_cls import nms_segments

K = int(sys.argv[1])
for i, v in enumerate(E.EPIC):
    vid = glob.glob(rf"C:/prism_data/epic/EPIC-KITCHENS/{v.split('_')[0]}/videos/{v}.MP4")[0]
    vs, vsc, _ = V._det[(v, "verb")]
    segs = nms_segments(vs, vsc, C.LAV_NMS, K)
    t = time.time()
    PL.epic_lavila(vid, segs)
    print(f"[{i}] {v} {len(segs)} segs {time.time() - t:.1f}s", flush=True)
