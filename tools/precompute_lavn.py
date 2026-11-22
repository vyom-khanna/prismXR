"""DEV: LaViLa on the top-K noun-detection segments that do not overlap the verb segments (cache tag 'lavn').

    C:/prism_env/Scripts/python.exe tools/precompute_lavn.py 100
"""
import glob
import sys
import time

import epic_dev as E
import dev_v3 as V
from prism import pipeline as PL

K = int(sys.argv[1])
for i, v in enumerate(E.EPIC):
    vid = glob.glob(rf"C:/prism_data/epic/EPIC-KITCHENS/{v.split('_')[0]}/videos/{v}.MP4")[0]
    segs = PL.noun_segments(V._det[(v, "noun")], V.lav_of(v)[0], K)
    t = time.time()
    PL.epic_lavila(vid, segs, tag="lavn")
    print(f"[{i}] {v} {len(segs)} segs {time.time() - t:.1f}s", flush=True)
